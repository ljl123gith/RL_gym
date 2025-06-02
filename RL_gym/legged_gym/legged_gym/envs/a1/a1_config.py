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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):  #继承了基础类中关于初始状态的所有设置，并可能在这里覆盖或添加一些。
        pos = [0.0, 0.0, 0.42] # x,y,z [m] 
        default_joint_angles = { # = target angles [rad] when action = 0.0  
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]  #表示每偏离目标角度 1 弧度产生 20 N*m 的力矩。
        damping = {'joint': 0.5}     # [N*m*s/rad] PD 控制器的微分 (D) 增益 ,表示每有 1 弧度/秒的速度变化，产生 0.5 N*m 的阻尼力矩
        # action scale: target angle = actionScale *  action + defaultAngle 
        # 最终发送给 PD 控制器的目标角度 = action_scale * 策略输出动作 + default_joint_angle。
        action_scale = 0.25 #策略输出的动作值通常在 [-1, 1] 之间。这个参数定义了策略输出与实际目标关节角度变化量之间的缩放关系。
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 ## 定义控制频率与仿真频率的关系
        # sim DT 是仿真器的步长 (simulation delta time)，policy DT 是策略的步长 (policy delta time)。
        # decimation = policy DT / sim DT。
        # 这里 decimation = 4 意味着每执行 4 个仿真步，策略才会产生一个新的动作。
        # 在这 4 个仿真步内，低层控制器会使用由上一个策略动作计算出的目标角度进行控制。


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
         # 配置机器人自身各个部分之间的碰撞检测。
         # 根据注释，1 表示禁用自碰撞检测，0 表示启用。忽略自碰撞可以简化仿真和训练，但可能不完全真实。
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            # 对关节力矩的惩罚权重。力矩越大，惩罚越大。负号表示惩罚。
            # 鼓励机器人使用更小的力矩，可能对应更节能或平稳的运动。
            dof_pos_limits = -10.0

class A1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01 
        # 熵系数。在 PPO 的损失函数中，会加入一个与策略熵相关的项，并乘以这个系数。
        # 熵越高，策略越随机。增加熵奖励鼓励策略进行更多的探索。0.01 是一个常见的值。


    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_a1'

  