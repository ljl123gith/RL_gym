import os
import sys
import numpy as np
sys.path.append("/opt/openrobots/lib/python3.10/site-packages")
# sys.path.append("/opt/openrobots/lib/python3.8/site-packages")

import pinocchio
import crocoddyl
from pinocchio.robot_wrapper import RobotWrapper
try:
    from pinocchio.visualize import MeshcatVisualizer
except ImportError:
    print("无法导入 pinocchio.visualize 模块，请检查 pinocchio 安装。")
    raise

current_directory = os.getcwd()
print("上层路径：", current_directory)

# change path ??
modelPath = os.path.join(current_directory, 'humanoid-gym','resources', 'robots', 'g1_description', 'urdf')
# 获取 meshes 目录的绝对路径
meshes_dir = os.path.join(os.path.dirname(modelPath), 'meshes')
#modelPath = current_directory + '/resources/robots/g1_description/urdf/'
# modelPath = current_directory + '/resources/robots/XBot/'
# 移除多余的反斜杠
URDF_FILENAME = "g1_23dof.urdf"

urdf_path = os.path.join(modelPath, URDF_FILENAME)

# Load the full model
# 直接将 meshes_dir 添加到搜索路径列表中，移除 package_dirs 参数
rrobot = RobotWrapper.BuildFromURDF(urdf_path, [modelPath, meshes_dir], pinocchio.JointModelFreeFlyer())  # Load URDF file
rmodel = rrobot.model

rightFoot = 'right_ankle_roll_joint' # 修改左右脚名字
leftFoot = 'left_ankle_roll_joint'

try:
    # 移除已弃用的 frameNames 参数
    display = crocoddyl.MeshcatDisplay(rrobot)
except Exception as e:
    print(f"创建 MeshcatDisplay 实例时出错: {e}")
    raise

q0 = pinocchio.utils.zero(rrobot.model.nq)
display.display([q0])
print("---------------initial pos-----------")

rdata = rmodel.createData()
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)

rfId = rmodel.getFrameId(rightFoot)
lfId = rmodel.getFrameId(leftFoot)

rfFootPos0 = rdata.oMf[rfId].translation
lfFootPos0 = rdata.oMf[lfId].translation

comRef = pinocchio.centerOfMass(rmodel, rdata, q0)

print("--------------compute com--------------")

# initialAngle = np.array([0.3, 0.1, 0.3, -0.5, -0.2, 0.0,
#                          0.3, 0.1, 0.3, -0.5, -0.2, 0.0,
#                          0.0, 0.00, 0.0,
#                          0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
#                          0.0, 0.0, 0.0,
#                          0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
# q0 = pinocchio.utils.zero(rrobot.model.nq)
# q0[6] = 1  # q.w
# q0[2] =0.5848  # z
# q0[ 7:39] = initialAngle
# display.display([q0])

for i in range(rrobot.model.nq-7):
    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[i+7] = 1
    display.display([q0])

for i in range(rrobot.model.nq-7-6): #same time
    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[i+7] = 1
    q0[i+13] = 1
    display.display([q0])
    print("-----左右关节同时转动对比------")

print("--------------start to play--------------")
for i in range(10000):
    phase = i * 0.001
    sin_pos = np.sin(2 * np.pi * phase)
    sin_pos_l = sin_pos.copy()
    sin_pos_r = sin_pos.copy()

    ref_dof_pos = np.zeros((1,12))
    scale_1 = 0.17
    scale_2 = 2 * scale_1
    # left foot stance phase set to default joint pos
    if sin_pos_l > 0 :
        sin_pos_l = sin_pos_l * 0
    ref_dof_pos[:, 0] = sin_pos_l * scale_1
    ref_dof_pos[:, 3] = -sin_pos_l * scale_2
    ref_dof_pos[:, 5] = sin_pos_l * scale_1
    # right foot stance phase set to default joint pos
    if sin_pos_r < 0:
        sin_pos_r = sin_pos_r * 0
    ref_dof_pos[:, 6] = -sin_pos_r * scale_1
    ref_dof_pos[:, 9] = sin_pos_r * scale_2
    ref_dof_pos[:, 11] = -sin_pos_r * scale_1
    # Double support phase
    ref_dof_pos[np.abs(sin_pos) < 0.1] = 0

    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    # 修正赋值操作，将二维数组转换为一维
    q0[7:7+12] = ref_dof_pos.flatten()
    display.display([q0])
print("-------------finish-----------------")