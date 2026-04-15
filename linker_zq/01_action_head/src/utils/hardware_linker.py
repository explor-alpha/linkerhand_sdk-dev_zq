# src/utils/hardware_linker.py
import sys
import os
# 导入配置，确保环境变量已经注入
import config 

# 现在可以安全地导入外层 SDK 了
from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.load_write_yaml import LoadWriteYaml

class LinkerHandController:
    def __init__(self, speed=config.DEFAULT_SPEED, torque=config.DEFAULT_TORQUE):
        """初始化灵巧手并应用默认配置"""
        print("🔧 正在初始化 LinkerHand 灵巧手...")
        
        # 读取官方 YAML 配置
        yaml_loader = LoadWriteYaml()
        setting = yaml_loader.load_setting_yaml()
        
        # 根据右手或左手获取配置 (按你代码要求写死 RIGHT_HAND，也可通过 config 灵活传)
        hand_joint = setting['LINKER_HAND']['RIGHT_HAND']['JOINT']
        modbus = setting['LINKER_HAND']['RIGHT_HAND']['MODBUS']
        
        # 实例化官方 API
        self.api = LinkerHandApi(hand_joint=hand_joint, hand_type=config.HAND_TYPE, modbus=modbus, can=None)
        
        # 设置初始速度和扭矩
        self.set_speed(speed)
        self.set_torque(torque)
        print(f"✅ 灵巧手初始化成功! [速度:{speed}, 扭矩:{torque}]")

    def set_speed(self, val):
        """统一设置 6 个关节的速度"""
        self.api.set_speed([val] * 6)

    def set_torque(self, val):
        """统一设置 6 个关节的扭矩"""
        self.api.set_torque([val] * 6)

    def safety_check(self, positions):
        """
        架构师的底线：硬编码安全锁 (防止 AI 输出导致物理干涉)
        positions 顺序: [拇指弯曲, 拇指侧摆, 食指, 中指, 无名指, 小指]
        """
        thumb_flex = positions[0]
        thumb_yaw = positions[1]
        index_flex = positions[2]
        
        # 【示例规则】如果拇指往掌心收(yaw小) 且 食指重度弯曲(flex大)，可能会打架
        # 这里你需要根据真机实际情况修改阈值！
        if thumb_yaw < 50 and index_flex > 200:
            print(f"⚠️ 触发安全锁：大拇指(yaw:{thumb_yaw})与食指(flex:{index_flex})存在碰撞风险！")
            # 强制修改姿态避险 (例如让大拇指退回到安全的 100)
            positions[1] = 100 
            print(f"🔄 已自动修正下发指令为: {positions}")
            
        return positions

    def move_hand(self, positions):
        """
        接收 AI 传来的 6 自由度指令 (0-255 的整数 list)，下发给硬件
        """
        # 1. 经过安全网过滤
        safe_positions = self.safety_check(positions)
        
        # 2. 调用官方 API 执行
        print(f"🤖 正在执行动作: {safe_positions}")
        self.api.finger_move(safe_positions)

    def reset_to_open(self):
        """快捷指令：恢复默认初始状态 250/255=0.980392"""
        self.move_hand([0.980392, 0.980392, 0.980392, 0.980392, 0.980392, 0.980392])