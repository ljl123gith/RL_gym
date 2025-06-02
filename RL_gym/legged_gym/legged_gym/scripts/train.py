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

import numpy as np
import os
from datetime import datetime # 导入 datetime，在本段代码中未直接使用，但在日志记录中会被用到 (TaskRegistry.make_alg_runner 内部)
 
import isaacgym
from legged_gym.envs import * # 从 legged_gym.envs 包中导入所有内容。这会触发 legged_gym.envs.__init__.py 的执行，从而注册所有的机器人任务到 task_registry 中。
from legged_gym.utils import get_args, task_registry  # 从 legged_gym.utils 包中导入 get_args 函数和 task_registry 对象。
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)  # 使用 task_registry 创建环境
    # 返回创建的环境实例 (env) 和最终使用的环境配置对象 (env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args) # 使用 task_registry 创建算法运行器 (这里是 PPO 运行器)
    # 返回创建的运行器实例 (ppo_runner) 和最终使用的训练配置对象 (train_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True) # 启动训练过程
    # num_learning_iterations=train_cfg.runner.max_iterations: 指定训练迭代次数，从训练配置中获取
    # init_at_random_ep_len=True: (可能是) 在开始训练时，让环境从随机的回合长度位置开始，而不是总是从第0步开始，有助于探索状态空间

if __name__ == '__main__':
    args = get_args()
    train(args)
    