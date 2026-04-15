import os
import sys

SDK_ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sdk')
sys.path.append(SDK_ROOT_DIR)

from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.load_write_yaml import LoadWriteYaml

# 初始化配置
yaml = LoadWriteYaml()
setting = yaml.load_setting_yaml()
hand_joint = setting['LINKER_HAND']['RIGHT_HAND']['JOINT']
modbus = setting['LINKER_HAND']['RIGHT_HAND']['MODBUS']

# 初始化 API
api = LinkerHandApi(hand_joint=hand_joint, hand_type="right", modbus=modbus, can=None)

# 定义函数控制手
def move_hand(positions):
    """positions: list of int, 每个关节目标值"""
    api.finger_move(positions)

def set_speed(val):
    api.set_speed([val] * 6)

def set_torque(val):
    api.set_torque([val] * 6)


# 使用示例
set_speed(50)      # 所有关节速度
set_torque(25)      # 所有关节扭矩
move_hand([0, 255, 255, 255, 255, 255])
#print(read_touch_matrix())