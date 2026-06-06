"""
自定义轻量化 test:
不同于isaac lab自带的测试:
- 轻量化
- print 关节名称; up & low limit; 测试 limit
- 测试各个参数的影响
- 连续规则测试

注意：
    1. 如果指令`"rh_thumb_cmc_pitch": 0.50`超限，会抽搐
"""


# 导入Isaac Sim核心模块
import omni
import numpy as np
import time
import argparse
import torch

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# 物理与控制器配置 (HAND_CONFIG)
HAND_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/qunz/projects_linux/own/linkerhand_sdk-dev_zq/o6_usd/linkerhand_o6_right/linkerhand_o6_right.usd",  # TODO
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Key！自碰撞开启; False: 调试urdf，动作调顺后开启自碰撞
            enabled_self_collisions=True,  
            # "软骨病"
            solver_position_iteration_count=12,
            # 速度迭代次数;"鬼畜抖动";（主要影响物体碰撞后的反弹、阻尼和摩擦力）
            solver_velocity_iteration_count=2,
        ),
    ),
    # 初始化手掌位置悬空 0.2 米
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={".*": 0.0},
        pos=(0.0, 0.0, 0.2),
    ),
    # 驱动器 (Actuators):ImplicitActuatorCfg，给每个关节挂 PD 控制器（弹簧阻尼系统）
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=30,
            velocity_limit_sim=30,
            stiffness=2000,#弹簧
            damping=200,#阻尼
        ),
    },
)

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """
    场景组装 (NewRobotsSceneCfg): 大地板、穹顶光（提供全局照明）和机器手
    """
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    # 穹顶光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # 生成一张桌子 (长 0.8m, 宽 0.8m, 高 0.2m)
    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True), # 桌子固定不动
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)), # 木头颜色
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)), # 桌子中心点在 0.1，顶部刚好是 0.2 高度
    )

    # 你的手（注意 pos 里的 z 轴可能需要加高一点，避免和桌子穿模）
    hand = HAND_CONFIG.replace(prim_path="{ENV_REGEX_NS}/LinkerhandO6")


def main():
    # --- Simulation ---
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    eye_pos = [0.3, 0.3, 0.4]       # eye_pos: 从 GUI 抄来的 Translate 坐标 (你的眼睛在哪)
    target_pos = [0.0, 0.0, 0.2]      # target_pos: 你的手在哪，就填什么 (你想看什么)
    sim.set_camera_view(eye_pos, target_pos)

    # --- Scene / Robot ---
    scene_cfg = NewRobotsSceneCfg(num_envs=1, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    o6_hand = scene["hand"]

    # 第一次把关节/刚体等数据同步出来（很重要）
    o6_hand.reset()
    o6_hand.write_data_to_sim()
    sim.step(render=True)
    o6_hand.update(sim_cfg.dt)

    # Key!关节映射,打印关节名
    print("=== Joint names ===")
    for i, n in enumerate(o6_hand.data.joint_names):
        print(i, n)
    print("===================")

    print("=== Joint Limits (Radians) ===")
    # soft_joint_pos_limits 的 shape 是 (num_envs, num_joints, 2)
    # [:, :, 0] 是下限，[:, :, 1] 是上限
    limits = o6_hand.data.soft_joint_pos_limits[0] 
    
    for i, n in enumerate(o6_hand.data.joint_names):
        lower = limits[i, 0].item()
        upper = limits[i, 1].item()
        print(f"{i:2d} {n:25s} | Lower: {lower:+.3f} rad | Upper: {upper:+.3f} rad")
    print("==============================")
    
    # ---------------------------
    # 1) 定义“关键帧”基础手势
    # ---------------------------
    # 完全张开（全 0）
    pose_open = {
        "rh_thumb_cmc_yaw": 0.0,     
        "rh_thumb_cmc_pitch": 0.0,   
        "rh_index_mcp_pitch": 0.0,   
        "rh_middle_mcp_pitch": 0.0,  
        "rh_ring_mcp_pitch": 0.0,    
        "rh_pinky_mcp_pitch": 0.0    
    }

    # 握拳/弯曲
    pose_close = {
        "rh_thumb_cmc_yaw": 0.5,     
        "rh_thumb_cmc_pitch": 0.4,   
        "rh_index_mcp_pitch": 1.2,   
        "rh_middle_mcp_pitch": 1.2,  
        "rh_ring_mcp_pitch": 1.2,    
        "rh_pinky_mcp_pitch": 1.2    
    }

    # 串联动作序列：先弯曲 -> 再打开 -> 再弯曲
    keyframes = [pose_close, pose_open, pose_close, pose_open, pose_close]

    # ---------------------------
    # 2) 将字典转换为底层目标 Tensor 列表
    # ---------------------------
    device = o6_hand.device
    num_joints = o6_hand.num_joints
    name_to_index = {n: i for i, n in enumerate(o6_hand.data.joint_names)}
    
    q_start = o6_hand.data.joint_pos.clone()  # 当前关节位置（作为插值起点） # shape: (1, num_joints)
    q_goals = []   # 目标关节位置（默认保持当前不变，只改你指定的那几个关节）

    for pose in keyframes:
        q = q_start.clone()

        # 1. 只设置主动关节
        for jname, jpos in pose.items():
            if jname not in name_to_index:
                raise RuntimeError(f"关节名 '{jname}' 不存在！请用上面打印的 joint names 里的名字。")
            if jname in name_to_index:
                q[0, name_to_index[jname]] = float(jpos)

        q_goals.append(q)

    # ---------------------------
    # 3) 时间轴与状态机插值控制
    # ---------------------------
    dt = sim_cfg.dt
    phase_duration = 3.0  # 每个动作阶段耗时 2 秒 (总共 3 个动作 = 6 秒)
    phase_steps = max(1, int(phase_duration / dt))

    # 先切到 position target 模式：用 set_joint_position_target + write_data_to_sim
    # （不要再用 write_joint_position_to_sim 每帧写0）
    step_idx = 0

    while simulation_app.is_running():
        # 更新内部状态（读取仿真结果）
        o6_hand.update(dt)

        # 计算当前处于第几个阶段 (0=第一次弯曲, 1=打开, 2=第二次弯曲)
        phase_idx = step_idx // phase_steps

        if phase_idx < len(q_goals):
            # --- 序列动作还在执行中 ---
            q_target = q_goals[phase_idx]
            
            # 关键：寻找当前这段动作的起点。
            # 如果是第一阶段，起点是原始状态；否则起点是上一个动作的终点。
            q_initial = q_start if phase_idx == 0 else q_goals[phase_idx - 1]

            # 计算当前阶段内部的插值进度 (alpha 从 0.0 -> 1.0)
            local_step = step_idx % phase_steps
            alpha = local_step / phase_steps

            # 线性插值
            q_cmd = (1 - alpha) * q_initial + alpha * q_target
            
            # 写入关节目标位置
            o6_hand.set_joint_position_target(q_cmd)

        else:
            # --- 动作序列全部执行完毕，死死锁住最后一个姿态避免回弹 ---
            o6_hand.set_joint_position_target(q_goals[-1])

        # 把目标写进仿真并推进一步
        o6_hand.write_data_to_sim()
        sim.step(render=True)
        
        step_idx += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
    # 关闭仿真应用
    simulation_app.close()