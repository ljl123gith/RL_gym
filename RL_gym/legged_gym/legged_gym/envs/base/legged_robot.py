
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg  #存储配置
        self.sim_params = sim_params #仿真参数
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)  #提取并预处理重要的配置值 (如 dt、观测缩放、奖励缩放、回合长度)
         
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers() # 初始化各种 PyTorch 张量，用于存储仿真状态 (根状态、DOF 状态、接触力) 和其他派生量 (速度、指令、奖励)。这对于 GPU 上的高效批量处理至关重要。
        self._prepare_reward_function() # 基于配置创建单个奖励函数及其权重的列表，从而设置奖励计算。
        self.init_done = True 
    def _init_buffers(self): #这是设置所有 PyTorch 张量的关键方法，这些张量将保存仿真数据和派生量。
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors 
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim) #获取根刚体状态的张量 (位置、旋转、线速度、角速度)。
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  #获取关节状态的张量 (位置、速度)。
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim) #获取接触力的张量。
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # 完整的根状态张量
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) #完整的 DOF 状态张量
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0] # 形状为 (num_envs, num_dof) 的 DOF 位置张量
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1] # 形状为 (num_envs, num_dof) 的 DOF 速度张量
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on 初始化其他缓冲区:
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0) 
    def _parse_cfg(self, cfg):  #解析和处理一个配置对象 (cfg)，并根据这些配置计算或设置类的各种内部参数
      
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # 从配置中提取控制频率和仿真时间步长，计算并存储实际的时间步长 (dt)。
       # 在机器学习，特别是强化学习中，通常需要将原始观察值（如机器人关节角度、速度等）进行归一化或缩放，使其落在特定范围（例如 -1 到 1），以提高训练的稳定性和效率。
        self.obs_scales = self.cfg.normalization.obs_scales  
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)   #定义奖励函数中各个组成部分的相对重要性
        self.command_ranges = class_to_dict(self.cfg.commands.ranges) #确保命令或动作在物理上合理且安全
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s #最大回合（episode）长度，以秒为单位。
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)  #将以秒为单位的最大回合长度除以我们前面计算出的控制步长 self.dt，得到理论上的步数。
                                #numpy.ceil 函数向上取整
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt) 
    def create_sim(self):   #设置核心的 Isaac Gym 仿真、地形和包含机器人的环境。
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()  #创建程序生成的高度图地形
        elif mesh_type=='trimesh':
            self._create_trimesh()  #创建由三角网格定义的地形。
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()   # 用机器人执行器填充仿真环境。  调用 _create_envs 方法，将机器人模型实例化并放置到已设置好的仿真器和地形中，完成环境的搭建。
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """ #如果 self.debug_viz 为真，此方法可用于在 Isaac Gym 查看器中绘制自定义可视化 (例如，在高度测量点绘制球体)。
        # draw height lines
        if not self.terrain.cfg.measure_heights: 
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
    def _create_ground_plane(self): #向仿真中添加一个大地平面，并根据配置设置摩擦和恢复系数。
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams() 
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) #设置地面的法线向量
        plane_params.static_friction = self.cfg.terrain.static_friction     #设置地面的静态摩擦力
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction   #设置地面的动态摩擦力
        plane_params.restitution = self.cfg.terrain.restitution #设置地面的恢复系数
        self.gym.add_ground(self.sim, plane_params)  #将地面添加到仿真中
    def _create_heightfield(self):  #其功能是在 Isaac Gym 仿真中创建一个基于高度图 (heightfield) 的复杂地形
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale #设置高度图的水平缩放因子
        hf_params.row_scale = self.terrain.cfg.horizontal_scale    #设置高度图的垂直缩放因子
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale #设置高度图的垂直缩放因子
        hf_params.nbRows = self.terrain.tot_cols #设置高度图的行数
        hf_params.nbColumns = self.terrain.tot_rows  #设置高度图的列数
        hf_params.transform.p.x = -self.terrain.cfg.border_size  #设置高度图的 x 坐标偏移
        hf_params.transform.p.y = -self.terrain.cfg.border_size  #设置高度图的 y 坐标偏移
        hf_params.transform.p.z = 0.0                            #设置高度图的 z 坐标偏移
        hf_params.static_friction = self.cfg.terrain.static_friction   #设置高度图的静态摩擦力
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction #设置高度图的动态摩擦力
        hf_params.restitution = self.cfg.terrain.restitution           #设置高度图的恢复系数

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)  #将高度图添加到仿真中
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device) 
    def _create_trimesh(self):  #向仿真中添加一个三角网格地形，并根据配置设置参数。
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams() #创建一个三角形网格参数对象
        tm_params.nb_vertices = self.terrain.vertices.shape[0] #设置三角形网格的顶点数量
        tm_params.nb_triangles = self.terrain.triangles.shape[0] #设置三角形网格的三角形数量
 
        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction #设置三角形网格的静态摩擦力
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction 
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)    #将三角形网格添加到仿真中 
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device) #将高度图数据转换为 PyTorch 张量
    def _init_height_points(self): #初始化一组三维点，这些点用于在机器人的局部坐标系中进行高度测量。
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False) #从配置中获取 Y 轴方向上的测量点坐标数组（或列表
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False) #创建一个包含测量点 x 坐标的张量
        grid_x, grid_y = torch.meshgrid(x, y) #这是一个 PyTorch 函数，用于从两个一维张量 x 和 y 中创建二维网格坐标。

        self.num_height_points = grid_x.numel() #计算测量点的数量
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False) #创建一个形状为 (num_envs, self.num_height_points, 3) 的张量，用于存储测量点的坐标
        points[:, :, 0] = grid_x.flatten() #将 grid_x 展平为一维张量。这个一维张量包含了所有网格点的 X 坐标
        points[:, :, 1] = grid_y.flatten()
        return points


    #create robot envs
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props：一个列表，包含了当前资产（例如机器人）中每个刚体形状的属性对象。
            env_id：当前正在创建的环境的 ID。这允许开发者为每个环境应用不同的随机化参数。

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction: #如果 cfg.domain_rand.randomize_friction 为真，则随机化刚体形状的摩擦力。它会预先计算摩擦力锥，以便在环境间一致应用或按环境随机化。
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64 #摩擦力值将被离散化为 64 个可能的预定义值
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props 
    def _process_dof_props(self, props, env_id):  #从 URDF/MJCF 资源属性中存储 DOF 位置限制、速度限制和力矩限制
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
            #它将原始范围 r 乘以一个 soft_dof_pos_limit 因子（该因子通常小于 1），从而缩小了有效的操作范围，使其在中心点 m 周围。
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  #计算当前 DOF 关节位置限制的中心点 (midpoint)。
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0] # 计算当前 DOF 关节位置限制的范围 (range)。
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit 
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props 
    def _process_rigid_body_props(self, props, env_id): #如果 cfg.domain_rand.randomize_base_mass 为真，则随机化基座质量
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props 
    def _create_envs(self): # 加载机器人资源 (asset) 并创建多个实例 (环境) ，Isaac Gym 环境设置的最后也是最重要的一步
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR) #从配置中获取机器人资源文件的路径
        asset_root = os.path.dirname(asset_path) #提取路径中的目录部分
        asset_file = os.path.basename(asset_path)
        #1.加载机器人模型 (URDF/MJCF)
        asset_options = gymapi.AssetOptions() #创建一个 AssetOptions 对象，用于配置加载的机器人资产的属性。
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode  #自由度 (DOF) 的默认驱动模式（例如，位置控制、力矩控制）。
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints    #是否将固定关节 (fixed joints) 折叠为单个关节。
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule  #是否将圆柱体 (cylinders) 替换为胶囊体 (capsules)。
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments  #是否翻转视觉附件的方向。
        asset_options.fix_base_link = self.cfg.asset.fix_base_link     #是否固定机器人的基座链接 (base link)。
        asset_options.density = self.cfg.asset.density 
        asset_options.angular_damping = self.cfg.asset.angular_damping #角阻尼系数，用于模拟关节的旋转运动。 
        asset_options.linear_damping = self.cfg.asset.linear_damping   #线阻尼系数，用于模拟关节的平移运动。
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity 
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature        #关节的机械臂ature，用于模拟关节的机械特性。
        asset_options.thickness = self.cfg.asset.thickness      #形状的厚度，用于模拟形状的厚度。
        asset_options.disable_gravity = self.cfg.asset.disable_gravity #是否禁用重力，即机器人是否受到重力的影响。

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options) 
        #2. 提取机器人资产的属性:
        self.num_dof = self.gym.get_asset_dof_count(robot_asset) #获取机器人资产的自由度 (DOF) 数量。
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset) #获取机器人资产的刚体数量。
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset) #获取机器人资产的 DOF 属性 (位置、速度限制、力矩限制等)。
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset) #获取机器人资产的刚体形状属性 (摩擦、恢复系数等)。

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        #3.识别特定身体部位的索引
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]  # 包含机器人“脚”的名称。用于奖励函数（例如“空中时间”）和接触检测。
        penalized_contact_names = [] #机器人上那些与地形接触时会受到惩罚的身体部位的名称（例如，身体部分不应接触地面）。
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = [] #机器人上那些与地形接触时会导致回合终止的身体部位的名称
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        #4.从配置中组合机器人基座的初始状态：位置 (x, y, z)、姿态 (四元数)、线速度和角速度。
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform() #用于定义机器人实例在每个环境中的初始位置和姿态。
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3]) 
        #5. 确定环境起始位置
        self._get_env_origins() # 确定网格/地形中每个机器人的起始位置
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        #6.创建每个环境和机器人实例
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))) #创建一个独立的环境。
            #随机化起始位置:
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            #处理和设置形状属性    
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i) # 回调函数，用于可能随机化每个环境形状的摩擦力
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props) # 在环境中创建一个机器人执行器。
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i) # 回调函数，用于存储/修改 DOF 属性 (限制等)。
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i) # 回调函数，用于可能随机化刚体属性 (例如基座质量)。
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 7.存储关键身体部位的索引，用于奖励函数和接触检测
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)): #足部索引，用于奖励函数 (空中时间) 和接触检测
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
                #  接触会产生惩罚的身体部位索引。
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
            #接触会导致回合终止的身体部位索引
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i]) 
    def _get_env_origins(self): #定义每个机器人环境的起始 (x, y, z) 位置。
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """  #如果使用带课程的高度图/三角网格地形，则从预定义的地形平台位置采样原点。
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:  #机器人将放置在一个简单的网格上，间距由 cfg.env.env_spacing 定义。
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0. 
    def _get_heights(self, env_ids=None): #计算机器人周围特定测量点相对于地形的实际高度。
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        
        #计算世界坐标系下的测量点位置，这部分将机器人在其局部坐标系中定义的测量点 (self.height_points) 转换到世界坐标系。
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
        # 将世界坐标转换为高度图网格索引
        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        #从高度样本中采样高度（近似双线性插值）
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale 
           


    #explore
    def _push_robots(self): #随机推动机器人。 通过对机器人施加随机的初始速度（模拟外力冲击），来增加训练出的强化学习策略的鲁棒性
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        """
        [:, 7:9]：这是PyTorch的切片操作：
                :：选择所有行，意味着对所有并行环境中的机器人进行操作。
            7:9：选择从索引7开始到索引9（不包括9）的列。在Isaac Gym等常见的机器人根状态表示中，各个索引的含义通常如下：
            0-2：位置 (x, y, z)
             -6：方向 (quaternion: qx, qy, qz, qw)
            7-9：线速度 (vx, vy, vz)
          10-12：角速度 (wx, wy, wz)
        因此，7:9 精确地对应于机器人的线速度的X和Y分量（vx 和 vy），与行末的
        """
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y  
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states)) 
    def _update_terrain_curriculum(self, env_ids): # 一种在强化学习中常用的（Domain Randomization）和Curriculum Learning）结合的技术，旨在提高策略的泛化能力和训练效率
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1) # 计算机器人与起始位置之间的距离
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up # 少于期望的一半距离
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down # 机器人根据距离的变化调整地形级别
        # Robots that solve the last level are sent to a random one    判断目前处于那一个难度级别 ，做高级别的话就随机选一个难度，并确保不在最低级别
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level, 
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]] 
        #更新了这些环境的初始位置（self.env_origins），使它们在下一个回合中会出生在对应新难度级别和地形类型的地方。
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # if : 当前正在重置的这些机器人，它们在上一回合中平均每步获得的线性速度追踪奖励，超过了理论最大追踪奖励的80%，那么就认为机器人表现非常出色，可以增加命令的难度。
             #  修改上下限  command_ranges[ 0  1 ]
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
    def _get_noise_scale_vec(self, cfg): #通过在训练数据中引入噪声，使训练出的策略对真实世界中传感器读数的不确定性或噪声具有更强的鲁棒性。

        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """  #: 创建一个向量，定义添加到观测的每个分量的噪声尺度 (如果启用了噪声)。
        noise_vec = torch.zeros_like(self.obs_buf[0])

        self.add_noise = self.cfg.noise.add_noise # 开关
        noise_scales = self.cfg.noise.noise_scales # 输入参数
        noise_level = self.cfg.noise.noise_level  

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel # 线性速度（0 1 2 ） #为线速度观测分量设置噪声尺度
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel # 角速度（3 4 5） #为角速度观测分量设置噪声尺度
        noise_vec[6:9] = noise_scales.gravity * noise_level  #重力（6 7 8）            #为重力观测分量设置噪声尺度
        noise_vec[9:12] = 0.                                                          # commands (9 10 11) #为命令观测分量设置噪声尺度
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # 关节位置（12 13 14 15 16 17 18 19 20 21 22 23） #为关节位置观测分量设置噪声尺度
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # 关节速度（24 25 26 27 28 29 30 31 32 33 34 35） #为关节速度观测分量设置噪声尺度
        noise_vec[36:48] = 0.                                                           # previous actions (36 37 38 39 40 41 42 43 44 45 46 47） #为前一个动作观测分量设置噪声尺度
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec 
    def check_termination(self): #检查终止 ，它包含了两种常见的终止条件：接触力过大（模拟跌倒或剧烈撞击）和达到最大回合长度（时间耗尽）。
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
    def reset_idx(self, env_ids):  #强化学习环境重置的核心函数，用于初始化环境状态、重置机器人、更新地形等。
        #它负责处理指定一批环境（env_ids）的回合结束逻辑，包括更新课程（如果启用）、重置机器人状态、生成新命令、清空内部缓冲区以及记录回合统计信息
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]):  定义了一个名为 reset_idx 的方法，接收一个 env_ids 列表，表示需要重置的环境的ID。
        """
        if len(env_ids) == 0:
            return
        # 1.update curriculum 课程学习的动态调整：根据表现提升或降低地形/命令难度。
        if self.cfg.terrain.curriculum:  #  如果地形课程处于活动状态，则可能更改这些环境的地形难度
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs

        #这个条件确保了命令课程只在每当一个完整的最大回合长度周期过去时（例如，每200步），才去尝试更新一次命令难度。这避免了重复计算和不必要的全局变量修改。
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0): # 如果指令课程处于活动状态，则可能更改指令难度。
            self.update_command_curriculum(env_ids)
        
        # 2.reset robot states 机器人状态的初始化：将机器人放置到新的起始位置和姿态，清零速度等。
        self._reset_dofs(env_ids) #重置关节位置 (在默认值附近随机化) 和速度 (置零)
        self._reset_root_states(env_ids) #重置基座位置 (到原点 + 偏移，可能来自地形课程) 和速度 (随机化)。

        #3.新的目标命令：为机器人设定新的运动指令。
        self._resample_commands(env_ids)  #分配新的随机指令。 

        # 4.reset buffers 内部缓冲区清理：重置各种计数器和历史数据。
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        #5. fill extras 这部分代码负责记录上一回合的统计信息，特别是奖励的平均值
        self.extras["episode"] = {} 
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        #6. log additional curriculum info 这部分记录了课程学习相关的额外信息，以便在训练日志中追踪课程的进展。
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm 决定是否将超时信息发送给强化学习算法
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    def _reset_dofs(self, env_ids): #重置关节位置 (在默认值附近随机化) 和速度 (置零)
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        #self.dof_pos ：这是一个PyTorch张量，存储了所有环境中所有机器人的关节位置。
        # 其形状通常是 (num_envs, num_dof)，其中 num_dof 是机器人自由度（关节）的数量。
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device) # 随机范围（0.5，1.5）
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32) 
        #通过这一行代码，物理模拟器被告知使用 self.dof_pos 和 self.dof_vel 中新设置的值来更新对应机器人的关节状态。
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):  #重置基座位置 (到原点 + 偏移，可能来自地形课程) 和速度 (随机化)
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
          self.root_states ：   结构是 (num_envs, 13)，其中包含：
                        0-2 ：位置 (x, y, z)
                        3-6 ：方向 (四元数: qx, qy, qz, qw)
                        7-9 ：线速度 (vx, vy, vz)
                      10-12 ：角速度 (wx, wy, wz)

        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _resample_commands(self, env_ids):  #重采样指令，为指定环境生成新的随机期望线速度和角速度，
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """  #从 self.command_ranges 中定义的范围采样速度。
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)    
   
    #progess
    def step(self, actions):  #RL 智能体的主要交互步骤。应用动作，仿真物理，计算结果。 
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        #1. 动作裁剪
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # 2.step physics and render each frame 渲染 
        self.render()
        #3.物理仿真循环 (Decimation Loop)
        for _ in range(self.cfg.control.decimation): #降采样循环
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) # 将这些力矩应用于机器人。
            self.gym.simulate(self.sim) #创建主仿真对象。
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True) #将物理仿真推进一个步骤
            self.gym.refresh_dof_state_tensor(self.sim) #用仿真中的新值更新 DOF 状态张量
        # 4.在物理步骤完成后执行计算
        self.post_physics_step() #在物理步骤完成后执行计算

        #5. 观测裁剪和返回 (Observation Clipping and Return)
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        #6.RL接口: 按照RL训练框架的约定，返回更新后的观测、奖励、重置标志和额外信息
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras 
    def post_physics_step(self):  #物理步进后处理  目的: 在一个智能体步骤内，物理仿真之后的主要计算
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        #1.刷新 Isaac Gym 张量
        self.gym.refresh_actor_root_state_tensor(self.sim)  #刷新 Isaac Gym 张量:
        self.gym.refresh_net_contact_force_tensor(self.sim)
        #2.更新计数器
        self.episode_length_buf += 1  #更新 episode_length_buf, common_step_counter。
        self.common_step_counter += 1

        # 3.prepare quantities 物理量
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        #4.
        self._post_physics_step_callback() # 占位符方法，允许在更复杂的环境中添加额外的、在物理步之后执行的自定义计算

        # 5.计算观测、奖励、重置等
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward() #计算所有环境的奖励
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten() #: 重置已终止的环境
        self.reset_idx(env_ids) #计算下一个智能体步骤的观测。
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        #6. 保存上一步状态 (Save last states)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # 7.调试可视化 (Debug Visualization)
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    def _post_physics_step_callback(self): #在 post_physics_step 中执行的附加逻辑的回调。
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids) #定期为环境重新采样目标指令 (例如，期望的前进速度)。
        if self.cfg.commands.heading_command: #如果 cfg.commands.heading_command 为真，则计算一个偏航角速度指令 (self.commands[:, 2]) 以将机器人导向目标航向 (self.commands[:, 3])。
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights: # 如果 cfg.terrain.measure_heights 为真，则采样机器人周围的地形高度
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()  #如果 cfg.domain_rand.push_robots 为真，则定期向机器人施加随机力/冲量
    def _compute_torques(self, actions):  #将 RL 智能体的动作转换为施加到机器人关节的力矩
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type 
        if control_type=="P":  # P(位置控制 Position Control): 动作是目标关节位置。力矩为 P_gain * (目标位置 - 当前位置) - D_gain * 当前速度。
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V": # V(速度控制 Velocity Control): 动作是目标关节速度。力矩为 P_gain * (目标速度 - 当前速度) - D_gain * (速度误差导数)
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T": # T(力矩控制 Torque Control): 动作直接解释为缩放后的力矩。
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    def compute_observations(self): # 构建 RL 智能体的观测向量
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:  #如果 cfg.terrain.measure_heights，则附加裁剪和缩放后的高度测量值。
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:  # 如果 cfg.noise.add_noise，则添加按 self.noise_scale_vec 缩放的随机噪声。
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


  #reward  
    def compute_reward(self): #计算每个环境的总奖励。
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        #调用每个奖励函数 (例如 _reward_lin_vel_z())，乘以其缩放因子 (self.reward_scales[name])，然后加到 self.rew_buf。
        self.rew_buf[:] = 0. #当前时间步的总奖励
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:  #终止奖励是在所有其他奖励计算并可能被剪裁之后才添加的 
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew    
    def _prepare_reward_function(self): #初始化和配置奖励计算机制。它在模拟环境开始运行之前被调用，
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()): #在循环中直接修改（例如使用 pop()) 你正在迭代的字典的键是危险的，
            scale = self.reward_scales[key]         #通过将其转换为列表，我们创建了一个键的静态副本，从而可以安全地迭代和修改原始字典。
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt #在强化学习环境中，许多奖励函数可能被设计为计算“每秒”的奖励率
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name) #添加
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums 初始化 self.episode_sums 字典以跟踪用于日志记录的累积奖励。
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    #------------ reward functions----------------  task  smooth safety beauty 
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) 
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)  
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) 
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target) 
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1) 
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1) 
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1) 
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1) 
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) 
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf 
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1) 
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1) 
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1) 
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma) 
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) 
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime     
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1) 
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1) 
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
