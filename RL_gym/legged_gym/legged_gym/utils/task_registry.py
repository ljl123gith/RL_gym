

import os  # 导入 os 模块，用于文件路径操作
from datetime import datetime # 导入 datetime 模块，用于处理日期和时间（特别是生成日志目录名称）
from typing import Tuple  
import torch
import numpy as np 

from rsl_rl.env import VecEnv # 从 rsl_rl.env 导入 VecEnv 类，表示向量化环境接口
from rsl_rl.runners import OnPolicyRunner   # 导入 OnPolicyRunner 类

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR # 导入项目根目录和环境目录
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}   # 初始化一个字典，用于存储任务名称到环境类的映射
        self.env_cfgs = {}       # 用于存储任务名称到环境配置对象的映射
        self.train_cfgs = {}     # 用于存储任务名称到 PPO 训练配置对象的映射
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO): #用于注册一个新任务。
        self.task_classes[name] = task_class # 将任务名称映射到环境类并存储
        self.env_cfgs[name] = env_cfg         # 将任务名称映射到环境配置对象并存储
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:  #根据任务名称获取注册的环境类。
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]: #根据任务名称获取注册的环境配置和训练配置。
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed   将训练配置中的 seed 复制到环境配置中 ，这很重要，可以确保每次训练或测试时，环境的初始化
        #（如地形生成、随机状态）和智能体训练的随机性（如网络权重初始化、采样）都基于同一个种子，从而保证实验的可复现性。
        env_cfg.seed = train_cfg.seed   # 将训练配置中的随机种子复制到环境配置中，确保仿真和训练使用相同的种子
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None: # 如果没有提供命令行参数 args，则调用辅助函数 get_args() 获取命令行参数
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes: # 检查提供的名称是否已在注册器中存在， 如果存在，获取对应的环境类
            task_class = self.get_task_class(name)
        else: 
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None: # 如果没有提供要覆盖的环境配置 env_cfg
            # load config files
            env_cfg, _ = self.get_cfgs(name)  # 则从注册器中加载默认的环境配置 (并忽略训练配置)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args) # 根据命令行参数 args 更新/覆盖 env_cfg 中的值 (只更新环境配置，训练配置为 None)
        set_seed(env_cfg.seed)  # 设置随机种子，确保环境初始化（如地形、随机状态）的可复现性
        # parse sim params (convert to dict first)   
        sim_params = {"sim": class_to_dict(env_cfg.sim)}  # 解析并准备传递给 Isaac Gym create_sim 函数的仿真参数 (sim_params)
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm) 
             env: Isaac Gym 环境实例。
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
             name: 任务名称 (字符串)，用于从注册表中加载配置。
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
             args: 命令行参数对象 (来自 get_args())。
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file 
        """
        # if no args passed get command line arguments 如果没有传入，则调用 get_args() 函数获取命令行参数。
        if args is None:                              # 这使得 make_alg_runner 可以在没有在外部调用 get_args() 的情况下独立运行。
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)    # 如果没有传入 train_cfg 但提供了 name，则调用 self.get_cfgs(name) 方法加载与该名称相关的配置。
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
                 # 如果同时传入了 train_cfg 和 name，打印一个警告信息。
                 # 说明在这种情况下，传入的 name 参数将被忽略，因为 train_cfg 具有更高的优先级。
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args) # 调用 update_cfg_from_args 函数，用命令行参数 args 更新 train_cfg。

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = class_to_dict(train_cfg)  #将最终使用的训练配置对象转换为字典格式
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)  # 实例化 OnPolicyRunner，传入环境、字典格式的配置、日志目录和 RL 设备
         # 实例化 OnPolicyRunner 类，创建训练运行器对象。
         # 传入以下参数：
         # env: 之前创建的环境实例。
         # train_cfg_dict: 字典格式的训练配置。
         # log_dir: 计算得到的日志目录路径 (或 None)。
         # device=args.rl_device: 从命令行参数 args 中获取 RL 算法使用的设备。
        
        
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume: # 如果需要恢复训练
            # load previously trained model     # 调用辅助函数获取要加载模型的路径
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg #返回创建的 OnPolicyRunner 实例和最终使用的训练配置对象

# make global task registry
task_registry = TaskRegistry()