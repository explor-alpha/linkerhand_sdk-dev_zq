import os
import sys
import time

SDK_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
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
def move_hand(target):
    """positions: list of int, 每个关节目标值"""
    api.finger_move(target)

def set_speed(speed):
    api.set_speed(speed)

def set_torque(torque):
    api.set_torque(torque)

def get_state():
    # 获取手当前状态
    return api.get_state()


# 测试
def real_time_monitor(interval=0.1, middle_mcp_pitch=0):
    """
    测试 LinkerHandApi.get_state
    实时打印手部状态
    :param interval: 刷新间隔（秒），默认0.1秒（10Hz）
    """
    print("开始实时监控 (按 Ctrl+C 停止)...")

    move_hand([250, 250, 250, middle_mcp_pitch, 250, 250])

    try:
        while True:
            hand_state = api.get_state()
            
            # 使用 \r 实现单行覆盖输出，或者配合 clear 清屏
            # 如果 hand_state 返回的是字典，我们格式化一下
            output = f"\rCurrent State: {hand_state}"
            
            # 打印并立即刷新缓冲区
            sys.stdout.write(output)
            sys.stdout.flush()
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n监控已停止。")


# --- 使用示例 ---
if __name__ == "__main__":
    set_speed(speed=[200, 50, 200, 50, 50, 50])
    set_torque(torque=[25, 200, 200, 25, 25, 25])

    # 调用实时监控
    real_time_monitor(interval=0.05, middle_mcp_pitch=0) # 0.05秒刷新一次, 示例控制中指

    # 移动到一个位置
    # move_hand(target=[250, 250, 250, 0, 250, 250])

    current_state = get_state()
    print (current_state)