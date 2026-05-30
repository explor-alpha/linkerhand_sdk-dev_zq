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
        usd_path="./o6_sim/o6_urdf/linkerhand_o6_right/linkerhand_o6_right.usd",  # "./o6_urdf/linkerhand_o6_right/linkerhand_o6_right.usd"
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=0,
            solver_velocity_iteration_count=0,
            # enabled_self_collisions=True,  #自碰撞开启
            # solver_position_iteration_count=12,
            # solver_velocity_iteration_count=2,
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
    eye_pos = [0.6, 0.3, 0.4]       # eye_pos: 从 GUI 抄来的 Translate 坐标 (你的眼睛在哪)
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


    # ---------------------------
    # 1) 定义你要的“手势目标”;修改 O6 右手真实的关节名
    # ---------------------------
    gesture_targets = {
        "rh_thumb_cmc_yaw": 0.2,     # 拇指横摆
        "rh_thumb_cmc_pitch": 0.4,   # 大拇指根部俯仰
        "rh_index_mcp_pitch": 0.4,   # 食指根部弯曲
        "rh_middle_mcp_pitch": 0.4,  # 中指根部弯曲
        "rh_ring_mcp_pitch": 0.4,    # 中指根部弯曲
        "rh_pinky_mcp_pitch": 0.4    # 小指
    }

    # 如果你还没填任何关节名，给一个提示并保持不动
    if len(gesture_targets) == 0:
        print("[WARN] gesture_targets 为空：请把你要控制的关节名/角度填进去。窗口会保持运行。")

    # ---------------------------
    # 2) 把“手势目标”写成全关节的目标向量
    # ---------------------------
    device = o6_hand.device  # 通常是 cuda
    num_joints = o6_hand.num_joints

    # 当前关节位置（作为插值起点）
    # shape: (1, num_joints)
    q_start = o6_hand.data.joint_pos.clone()

    # 目标关节位置（默认保持当前不变，只改你指定的那几个关节）
    q_goal = q_start.clone()

    name_to_index = {n: i for i, n in enumerate(o6_hand.data.joint_names)}
    for jname, jpos in gesture_targets.items():
        if jname not in name_to_index:
            raise RuntimeError(f"关节名 '{jname}' 不存在！请用上面打印的 joint names 里的名字。")
        q_goal[0, name_to_index[jname]] = float(jpos)

    # ---------------------------
    # 3) 平滑过渡到目标手势（例如 2 秒）
    # ---------------------------
    move_duration = 3.0
    dt = sim_cfg.dt
    steps = max(1, int(move_duration / dt))

    # 先切到 position target 模式：用 set_joint_position_target + write_data_to_sim
    # （不要再用 write_joint_position_to_sim 每帧写0）
    step_idx = 0

    while simulation_app.is_running():
        # 更新内部状态（读取仿真结果）
        o6_hand.update(dt)

        if len(gesture_targets) > 0 and step_idx <= steps:
            alpha = step_idx / steps  # 0->1
            q_cmd = (1 - alpha) * q_start + alpha * q_goal  # shape (1, num_joints)

            # 写入“关节目标位置”
            o6_hand.set_joint_position_target(q_cmd)

            # 把目标写进仿真
            o6_hand.write_data_to_sim()

            step_idx += 1
        else:
            # 达到目标后保持（持续写 target，避免回弹）
            if len(gesture_targets) > 0:
                o6_hand.set_joint_position_target(q_goal)
                o6_hand.write_data_to_sim()

        sim.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
    # 关闭仿真应用
    simulation_app.close()