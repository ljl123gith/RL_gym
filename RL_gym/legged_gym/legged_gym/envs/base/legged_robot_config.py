# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096         #同时运行的仿真环境数量
        num_observations = 235  # 每个环境的观察维度 

        #用于非对称训练，即 critic 具有更多信息，而 actor 只有标准观测
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12 #每个环境的动作空间维度
        env_spacing = 3.  # not used with heightfields/trimeshes  环境之间的间距，用于在网格中排列环境
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds  每个环境的最大时间步长

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh      地形类型 
        horizontal_scale = 0.1 # [m]  地形网格的水平
        vertical_scale = 0.005 # [m]    地形网格的垂直缩放因子
        border_size = 25 # [m]   地形边界的大小
        curriculum = True #  是否使用 curriculum 地形（逐渐增加地形难度）
        static_friction = 1.0  # 地形摩擦 
        # 动态摩擦系数，用于模拟物体在接触时的摩擦力。
        dynamic_friction = 1.0 #  动态摩擦系数，用于模拟物体在接触时的摩擦力。
        restitution = 0. # 地形恢复系数
        # rough terrain only:
        measure_heights = True  #   是否在机器人周围测量地形高度（作为观测的一部分）
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] #定义机器人基座坐标系下测量地形高度的点的 x 和 y 坐标网格。
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state #课程学习的起始难度级别。
        terrain_length = 8. # [m]  地形的长度
        terrain_width = 8. # [m]  地形的宽度
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2] # 地形类型的比例，用于 curriculum 地形。
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces   斜坡阈值，用于将斜坡校正为垂直表面。

    class commands:
        curriculum = False # curriculum learning enabled
        max_curriculum = 1. # maximum percentage of levels in the command
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s] # 命令改变的频率（秒）
        heading_command = True # if true: compute ang vel command from heading error 如果为真，偏航角速度指令将从目标朝向误差重新计算。
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state: 
        # 机器人基座的初始位置、旋转（四元数）、线速度和角速度。这些参数用于定义机器人的初始状态。
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad] P
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad] D
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5 # 动作缩放因子。RL 策略的输出动作会乘以这个因子，然后加上默认关节角度来得到目标角度。
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 # 动作更新频率。控制动作每 4 个物理时间步更新一次。

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []  #列表，指定哪些部位的接触会导致惩罚。
        terminate_after_contacts_on = [] #列表，指定哪些部位的接触会导致回合终止。
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True    #随机化地面摩擦力。
        friction_range = [0.5, 1.25] #地面摩擦力的范围。
        randomize_base_mass = False  #随机化机器人基座的质量。
        added_mass_range = [-1., 1.] #添加到机器人基座质量的范围。
        push_robots = True           #是否随机推动机器人。
        push_interval_s = 15         #推动机器人的间隔时间（秒）。
        max_push_vel_xy = 1.         #最大的水平推动速度。

    class rewards:
        class scales:
            termination = -0.0         # 终止奖励的缩放因子。
            tracking_lin_vel = 1.0     # 跟踪线速度的奖励缩放因子。
            tracking_ang_vel = 0.5     # 跟踪角速度的奖励缩放因子。
            lin_vel_z = -2.0           # 垂直线速度的奖励缩放因子。
            ang_vel_xy = -0.05         # 水平角速度的奖励缩放因子。
            orientation = -0.          # 朝向的奖励缩放因子。
            torques = -0.00001         # 扭矩的奖励缩放因子。
            dof_vel = -0.              # 关节速度的奖励缩放因子。
            dof_acc = -2.5e-7          # 关节加速度的奖励缩放因子。
            base_height = -0.          # 基座高度的奖励缩放因子。
            feet_air_time =  1.0       # 脚在空中的时间的奖励缩放因子。
            collision = -1.            # 碰撞的奖励缩放因子。
            feet_stumble = -0.0        # 脚的 stumble 奖励缩放因子。
            action_rate = -0.01        # 动作速率的奖励缩放因子。
            stand_still = -0.          # 站立的奖励缩放因子。
       
        # 如果为 True，所有负奖励都被裁剪为零，以避免早期终止问题。
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25        # tracking reward = exp(-error^2/sigma) 跟踪奖励的平滑参数
        soft_dof_pos_limit = 1.      # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.      #关节位置、速度和扭矩的软限制，超出这些限制会受到惩罚。
        soft_torque_limit = 1.       # 关节位置、速度和扭矩的软限制，超出这些限制会受到惩罚。
        base_height_target = 1.      # 期望的机器人基座高度。
        max_contact_force = 100.     # forces above this value are penalized # 最大接触力，超出这个值会受到惩罚。
 
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100. # 对观察值进行裁剪的阈值。
        clip_actions = 100.      # 对动作进行裁剪的阈值。

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005                # 物理时间步长 [s] 仿真时间步长（秒）
        substeps = 1               # 子步长，每个物理时间步长内的子步长数量。
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1                # 0 is y, 1 is z

        class physx: #hysX 物理引擎的详细配置。
            num_threads = 10  # 物理引擎使用的线程数。
            solver_type = 1   # 0: pgs, 1: tgs # 物理引擎的求解器类型。
            num_position_iterations = 4  # 位置迭代次数。
            num_velocity_iterations = 0  # 速度迭代次数。
            contact_offset = 0.01  # [m] 接触偏移量。
            rest_offset = 0.0      # [m]  恢复偏移量。
            bounce_threshold_velocity = 0.5  #0.5 [m/s] 反弹阈值速度。
            max_depenetration_velocity = 1.0 #1.0 [m/s] 最大分离速度。
            max_gpu_contact_pairs = 2**23    #2**24 -> needed for 8000 envs and more   # 最大 GPU 接触对数量。
            default_buffer_size_multiplier = 5 # 物理引擎的默认缓冲区大小乘数。
            contact_collection = 2       # 0: never, 1: last sub-step, 2: all sub-steps (default=2)  # 接触集合方式。

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1 # 随机种子
    runner_class_name = 'OnPolicyRunner'  # 运行器类名
    class policy:  # 策略配置
        init_noise_std = 1.0                # 初始化噪声标准差
        actor_hidden_dims = [512, 256, 128] # 策略网络隐藏层维度
        critic_hidden_dims = [512, 256, 128] # 价值网络隐藏层维度
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:  
        # training params
        value_loss_coef = 1.0          # 价值损失在总损失中的权重。
        use_clipped_value_loss = True  #是否使用裁剪的价值损失
        clip_param = 0.2               # PPO 裁剪参数。
        entropy_coef = 0.01            # 熵损失在总损失中的权重。 熵损失的权重，用于鼓励探索。
        num_learning_epochs = 5   #每个数据收集周期内学习的 epoch 数
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches  将收集到的数据分成多少个 mini-batch 进行梯度更新。
        learning_rate = 1.e-3 #5.e-4 # 学习率。
        schedule = 'adaptive' # could be adaptive, fixed # 学习率调度策略。
        gamma = 0.99   # 折扣因子。
        lam = 0.95     # GAE 折扣因子。
        desired_kl = 0.01  # 目标 KL 散度。
        max_grad_norm = 1. # 梯度裁剪的最大范数。

    class runner:
        policy_class_name = 'ActorCritic' # 策略类名
        algorithm_class_name = 'PPO' # 算法类名
        num_steps_per_env = 24 # per iteration # 每个环境在每次迭代中收集的步数。
        max_iterations = 1500 # number of policy updates # 最大迭代次数。

        # logging
        save_interval = 50 # check for potential saves every this many iterations # 每隔多少次迭代检查是否需要保存模型
        experiment_name = 'test' # 实验名称。
        run_name = '' # 运行名称。
        # load and resume 
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt