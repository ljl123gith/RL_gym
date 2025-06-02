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

#用于处理配置、随机性、加载模型以及将策略导出为 TorchScript
import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil 

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
  
  # 
def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj  #直接返回对象本身
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []  # 初始化一个列表，用于处理属性值是列表的情况
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return
    #用于设置多个随机数生成器的种子，包括 Python 内置的 random、NumPy 和 PyTorch (CPU 和 CUDA)。
    # 确保在多次运行中获得相同的随机行为，这对实验的可复现性至关重要。当传入 -1 时，它会生成一个随机种子
def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
     
    random.seed(seed) # 设置 Python 内置 random 模块的种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# parse_sim_params 负责将项目内的配置转换为 Isaac Gym 所需的仿真参数格式。
def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams() # 创建一个空的 Isaac Gym SimParams 对象

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params
# get_load_path 提供了一种查找已训练模型检查点的标准方式。
def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

#根据解析后的命令行参数来覆盖环境配置 (env_cfg) 和训练配置 (cfg_train) 中的特定属性。
# 这使得用户可以在不修改配置文件的情况下通过命令行调整一些重要的参数（如环境数量、种子、最大迭代次数、恢复训练设置等）。这个函数直接修改传入的配置对象。
def update_cfg_from_args(env_cfg, cfg_train, args):
    # 定义一个函数 update_cfg_from_args，接收三个参数：
    # env_cfg: 环境配置对象 (可能来自 task_registry.make_env 的输出)
    # cfg_train: 训练配置对象 (可能来自 task_registry.make_alg_runner 的输出)
    # args: 命令行参数对象 (来自 get_args() 的输出)

    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs # 更新环境配置中的 num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed  # # 如果 args.seed 不为空，则将 cfg_train 对象中 seed 的值更新为 args.seed 的值。
                                          # 命令行指定的随机种子将覆盖配置文件中的设置。
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations #   如果 args.max_iterations 不为空，则更新 cfg_train.runner.max_iterations 的值。
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [ # 定义自定义命令行参数列表  ，--task 指定任务名称 ，--resume 检查点恢复训练 (布尔标志) ，--experiment_name 实验名称，--run_name 运行名称， --load_run  恢复训练时要加载的运行名称，--checkpoint
        {
            "name": "--task",
            "type": str,
            "default": "anymal_c_flat",  # You might want to change this default
            "help": "Name of the task to run. Overrides config file if provided.",
        },
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},  # 是否在无头模式下运行 (无图形界面)
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},  # 是否使用 Horovod 进行多 GPU 训练 
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},# RL 算法使用的设备
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."}, # 环境数量
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},  # 随机种子
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."}, # 最大训练迭代次数
    ]
    # parse arguments
    args = gymutil.parse_arguments(  # 调用 Isaac Gym 的实用函数解析命令行参数
        description="RL Policy", 
        custom_parameters=custom_parameters
        )

    # name allignment 
    args.sim_device_id = args.compute_device_id # 将 compute_device_id 赋值给 sim_device_id
    args.sim_device = args.sim_device_type # 将 sim_device_type (如 'cuda' 或 'cpu') 赋值给 sim_device
    if args.sim_device=='cuda':    # 如果 sim_device 是 'cuda'
        args.sim_device += f":{args.sim_device_id}"  # 在后面加上设备 ID，格式化为 "cuda:0", "cuda:1" 等
    return args 

#将训练好的策略模型（特别是其中的 actor 网络）导出为 TorchScript 格式。
# TorchScript 允许在没有 Python 解释器的情况下运行 PyTorch 模型，便于部署。
# 它区分处理带有循环模块（如 LSTM）和不带循环模块的策略。对于循环策略，它使用专门的 PolicyExporterLSTM 类
def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

# 这是一个 PyTorch Module，用于包装包含 LSTM 模块的策略，以便正确地将其导出为 TorchScript。
# 它关键在于使用 register_buffer 来管理 LSTM 的内部隐藏状态和细胞状态。
# 这样，导出的 TorchScript 模型就能在每次调用 forward 时自动维护并更新这些状态，并在需要时通过导出的 reset_memory 方法进行重置。
class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
