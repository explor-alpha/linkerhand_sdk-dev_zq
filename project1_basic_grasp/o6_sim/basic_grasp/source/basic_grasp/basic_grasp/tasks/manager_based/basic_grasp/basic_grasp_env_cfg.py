# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# 使用 isaac lab - Manager-Based; single-agent 模块化模板 - skrl算法库
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg  # 获取状态信息
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

from . import mdp


##
# Scene definition
##

# Linkerhand o6 right： 6 个主动关节
O6_CONTROL_JOINTS=[    
    "rh_thumb_cmc_yaw", "rh_thumb_cmc_pitch",
    "rh_index_mcp_pitch", "rh_middle_mcp_pitch", 
    "rh_ring_mcp_pitch", "rh_pinky_mcp_pitch",
    ]


# [TODO]
O6_FINGERTIP_NAMES=[
        "rh_thumb_distal",
        "rh_index_distal",
        "rh_middle_distal",
        "rh_ring_distal",
        "rh_pinky_distal",
    ]


@configclass
class BasicGraspSceneCfg(InteractiveSceneCfg):
    """Configuration for the dexterous hand grasping scene.
    加载—每个组件独立加载：
        1. 加载方式/Asset Type:
            AssetBaseCfg (ground; table; dome_light)
            ArticulationCfg (带关节: robot)
            RigidObjectCfg (不带关节的刚体: ball)
        2. 独立加载原因：
            Domain Randomization: 可以精准地定位小球并给它施加 EventTerm; 否则很难把 ball 从 scene 中独立
            Asset Type Mismatch
            Reward Function: 独立加载的 SceneEntity, 才能直接获取对应的状态
    加载—workflow: 
        1. usd 保存：分别选中灵巧手的根节点、小球的根节点，右键点击 "Save Selected"（保存选中项），把它们存为 hand.usd 和 ball.usd
        2. prim_path, usd_path加载
            1. prim_path="{ENV_REGEX_NS}/Robot"=/World/envs/{env_1-4095}/Robot
            2. 读取 usd_path(hand.usd) 文件，挂载/克隆到 prim_path (每个并行环境的 Robot 虚拟文件夹下)
        3. init_state 初始化
    Details:
        1. 将 rh_hand_base_link 作为 Articulation Root 在代码里何处体现？— 封装在 usd 中
        2. usd 导入信息：
            可导入：
                网格形状、外观材质颜色、
                物理属性（质量、密度、动/静摩擦力、弹性）
                关节树结构（Articulation Root、所有 Joint 的限制范围、Mimic 联动关系）
                相机内部参数（FOV、焦距等）
            可覆盖（InitialStateCfg 或 EventCfg (重置逻辑) 覆盖）：
                初始世界坐标；初始关节角度
        3. 并行-加载: 
            Global Assets: ground & dome_light — /World/ground
            其他 — {ENV_REGEX_NS}/Robot
        4. 并行-防止干扰：
            逻辑防线（碰撞过滤 Collision Filtering）
                底层 PhysX: 每个环境分配一个独立的碰撞组; 即使 env_spacing=0, 也不会碰撞！
                全局资产（如 prim_path="/World/ground" 的地面）被设置为可以与所有碰撞组交互。
            物理防线（环境间距 env_spacing）
                仅用于可视化
        5. Torque Control or Position Control
            Torque Control: 
                即，神经网络直接输出每个电机的扭矩；
                Agent 更能建立物理世界的理解
                优点：动作极其平滑，上限极高
                贵(万元/十万元级): 精密力矩传感器；精确输出；减速器摩擦力必须极小等等
            Position Control: 
                即，神经网络输出目标角度（比如让食指弯曲 30 度）；
                Agent 只能理解"需要到什么位置"，不能理解“需要施加多少力“（由真实电机 PID 控制器负责）
                定位：mdp.JointPositionActionCfg
                便宜(百元级): 简单的电位器（测角度）和廉价芯片
        6. 视觉模块：
            先 RL 训练低维 observation (3 维已知 ball 位置)
            再 RL 训练高维 observation (CNN, 图像-ball 位置)
            以此降低 reward 调参时间成本
        7. hand自碰撞参数 `enabled_self_collisions`
            关闭意味着它的中指和食指可以互相穿透！
            有时关闭是因为开源的灵巧手 URDF/USD 碰撞网格做得很烂，一旦手指靠得太近，物理引擎就会判定它们卡在一起疯狂弹开。关闭自碰撞是一种“逃课”的稳妥做法。如果你在跑你的项目时发现手指莫名其妙炸飞，你也要改成 False。
    """

    # 1. 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 2. 桌子 (匹配你 PDF 中的尺寸: 0.8x0.8x0.2, 高度设为 0.1 刚好贴地)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.02),
            rigid_props=None,  # 不添加刚体属性，这样它就会像你设定的一样固定在空中
            collision_props=sim_utils.CollisionPropertiesCfg(), # 开启碰撞
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)), # 棕色木头色
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)),
    )

    # 3. 待抓取的小球 (匹配 PDF 设定: 悬浮、高密度、高摩擦)
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.027,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # kinematic_enabled=True,  # 核心: 运动学固定  # TODO
                disable_gravity=True, # 禁用重力
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=10.0), # 密度 10
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0, dynamic_friction=2.0 # 摩擦力 2.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05501, 0.00733, 0.14303)), # 初始化
    )

    # 4. 灵巧手 (Articulation)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/qunz/projects_linux/own/linkerhand_sdk-dev_zq/o6_usd/linkerhand_o6_right/linkerhand_o6_right.usd",     # TODO
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),   # TODO，max_depenetration_velocity=5.0,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,   # TODO，TRUE？ 不加？
                # Key！自碰撞开启; False: 调试urdf，动作调顺后开启自碰撞
                enabled_self_collisions=True,  
                # "软骨病"
                solver_position_iteration_count=12,
                # 速度迭代次数;"鬼畜抖动";（主要影响物体碰撞后的反弹、阻尼和摩擦力）
                solver_velocity_iteration_count=2,
            ), # TODO
            activate_contact_sensors=True,  # 核心: 开启接触传感器用于奖励计算
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.1, 0.0, 0.02), # PDF 中设定的手掌初始位姿
            #rot=(1.0, 0.0, 0.0, 0.0),  # TODO
            #joint_pos={".*": 0.0},  # 强制所有关节默认初始位置为 0.0 # TODO
        ),

        # TODO: 关节-电机控制器；需要调参
        actuators={
            "active_fingers": ImplicitActuatorCfg(
                joint_names_expr=O6_CONTROL_JOINTS, # 显式指定你的 6 个主动关节，与 USD 中完全一致！
                
                # --- 1. 模拟底层位置闭环 ---
                # 对应 PD 控制器的 Kp 和 Kd
                # 这些值需要你根据仿真表现微调。Kp 设大一些保证位置追踪。
                stiffness=2.0,  # 调低增强接触柔顺性; 较高的刚度，确保关节紧紧追随 RL 策略输出的目标位置
                damping=0.1,     # 适中的阻尼，防止手指动作像弹簧一样震荡; 提高用于吸收震荡
                
                # --- 2. 模拟 set_torque=[94]*6 的物理约束 ---
                # 【核心对应 set_torque】
                # 限制物理引擎输出的最大扭矩（单位 N*m）
                # 假设查阅模型参数后，关节极限扭矩为 1.5 N·m，94/255 大约是 37%
                # 在此限制物理引擎计算出的 PD 扭矩上限，使得仿真具备柔顺性
                effort_limit_sim=0.55,  # 10, if fingers don't move, try 20~50
                
                # --- 3. 模拟 set_speed=[120]*6 的物理约束 ---
                # 【核心对应 set_speed】
                # 限制最大角速度（单位 rad/s）
                # 假设电机空载极限转速为 6.0 rad/s，120/255 大约是 47%
                # 物理引擎底层将通过约束冲量(Constraint Impulse)强制钳位角速度
                velocity_limit_sim=2.8, # 20
            )
        },
    )

    # 5. D455 相机 (直接挂载在环境中或手掌上)
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1, # 10Hz
        height=480,
        width=640,
        data_types=["distance_to_image_plane"], # 提取深度图用于强化学习
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)
        ),
        # 对应 PDF 中的坐标
        offset=CameraCfg.OffsetCfg(pos=(0.1358, 0.02347, 0.28703)), 
    )

    # 6. 光源
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    ArticulationActionCfg，将强化学习网络输出的[-1,1]映射为关节真实的物理弧度
    usd 中设置并调整Mimic，这里只控制主动关节
    linkerhand o6 right 为位置控制 (Position Control) 
    """
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=O6_CONTROL_JOINTS, 
        scale=1.0, 
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # 1. 关节位置 (11维)
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=O6_CONTROL_JOINTS)})
        # 2. 关节速度 (11维)
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=O6_CONTROL_JOINTS)})
        # 3. 指尖到小球相对距离向量 (15维)
        tip_rel = ObsTerm(
            func=mdp.tip_to_ball_rel, # [TODO]
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=O6_FINGERTIP_NAMES),
                "ball_cfg": SceneEntityCfg("ball")
            }
        )

        # 注意：深度相机的图像数据不能直接和上面的低维数据连结 (concatenate)，
        # 如果要用相机，需要在算法库 (如 skrl) 层面进行特征提取分离，这里暂且用低维状态。
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events (Domain Randomization & Resets)."""
    # Managed 中小球是 kinematic 的，不需要重置位置，默认即可。[TODO]
    # 每回合重置小球到一个随机位置
    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}, # z轴保持悬浮高度
            "velocity_range": {},
        },
    )

    # 重置手部关节
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0), # <--- 偏移量设为 0，代表回到默认位置
            "velocity_range": (0.0, 0.0), # <--- 速度重置为 0
        },
    )

@configclass
# [TODO]
class RewardsCfg:
    """Reward terms for the MDP."""
    # 1. 指尖距离奖励 (系数 3.0)
    tip_dist = RewTerm(
        func=mdp.fingertip_distance_reward,
        weight=3.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=O6_FINGERTIP_NAMES),
            "ball_cfg": SceneEntityCfg("ball")
        }
    )
    """
    # 2. 接触力奖励 (系数 1.0)
    contact = RewTerm(
        func=mdp.fingertip_contact_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=O6_FINGERTIP_NAMES),
            "threshold": 0.5
        }
    )    
    """

    # 3. 动作平滑度惩罚 (系数 0.001) - 惩罚动作的 L2 范数
    action_penalty = RewTerm(func=mdp.action_l2, weight=-0.001)

@configclass
# [TODO]
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) 如果小球掉落出界则结束 (虽然禁用了重力，但防止手把它拍飞)
    ball_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("ball")},
    )

    # success
    success = DoneTerm(
        func=mdp.success_termination, # [TODO]
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=O6_FINGERTIP_NAMES),
            "ball_cfg": SceneEntityCfg("ball"),
            "dist_thresh": 0.03,
            "contact_thresh": 0.5,
            "min_contacts": 3
        }
    )
##
# Environment configuration
##

"""
scene:
        replicate_physics=True,
        filter_collisions=True,
不用声明观测空间？
奖励参数？
"""
@configclass
class BasicGraspEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BasicGraspSceneCfg = BasicGraspSceneCfg(num_envs=256, env_spacing=1.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2 # 控制频率
        self.episode_length_s = 6.0 # 每回合 6 秒
        # viewer settings
        self.viewer.eye = (0.5, 0.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        # simulation settings
        self.sim.dt = 1 / 120 
        self.sim.render_interval = self.decimation