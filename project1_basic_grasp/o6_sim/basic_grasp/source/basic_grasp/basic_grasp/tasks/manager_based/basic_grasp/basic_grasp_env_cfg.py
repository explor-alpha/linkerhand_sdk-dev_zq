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
# 本模块参数设置：导入 isaac sim scene 中测试好的参数
##

# Linkerhand o6 right： 6 个主动关节
O6_CONTROL_JOINTS=[    
    "rh_thumb_cmc_yaw", "rh_thumb_cmc_pitch",
    "rh_index_mcp_pitch", "rh_middle_mcp_pitch", 
    "rh_ring_mcp_pitch", "rh_pinky_mcp_pitch",
    ]

# TODO
# Linkerhand o6 right： 5 指尖刚体，用于距离奖励设计
O6_FINGERTIP_NAMES=[
        "rh_thumb_distal",
        "rh_index_distal",
        "rh_middle_distal",
        "rh_ring_distal",
        "rh_pinky_distal",
    ]


@configclass
class BasicGraspSceneCfg(InteractiveSceneCfg):
    """Configuration for the dexterous hand grasping scene. """
    # 1. 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 2. 桌子
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.02), # scene
            rigid_props=None,  # 不添加刚体属性，这样它就会像你设定的一样固定在空中
            collision_props=sim_utils.CollisionPropertiesCfg(), # 碰撞属性
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)), # 棕色木头色
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)), # scene
    )

    # 3. 待抓取的 Ball (Rigid)
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.027, # scene
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # kinematic_enabled=True,  # Ture 则小球被固定在原位，不会被推动
                disable_gravity=True, # 禁用重力
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=10.0), # 密度 10
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0, dynamic_friction=2.0 # 摩擦力 2.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(), # 碰撞属性
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.055, 0.007, 0.143)), # scene
    )

    # 4. 灵巧手 (Articulation)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/qunz/projects_linux/own/linkerhand_sdk-dev_zq/o6_usd/linkerhand_o6_right/linkerhand_o6_right.usd",  # mimic 等等在 usd 中已完成调整
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, # 启动重力
                max_depenetration_velocity=5.0,  # 防止穿模崩溃，最大反穿模速度
            ),  # TODO
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,   # 固定 linkerhand 基座
                enabled_self_collisions=True,  # 开启自碰撞; PS：调试 urdf 关节联动时可以 False
                solver_position_iteration_count=12,  # 计算的精细程度-位置迭代次数（决定能不能抗住并传递 PD 的力而不发生穿模或散架）。 stiffness若大，此数值需大（关节连接处越“硬”），否则，散架/"软骨病"
                solver_velocity_iteration_count=2,  # 计算的精细程度-速度迭代次数（主要影响碰撞计算的精细程度：反弹、阻尼和摩擦力）;过小，"鬼畜抖动";
            ), # TODO
            activate_contact_sensors=True,  # 开启接触传感器用于奖励计算
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.1, 0.0, 0.02), # scene
            rot=(1.0, 0.0, 0.0, 0.0),  # scene，0 旋转
            joint_pos={".*": 0.0},  # 正则表达，所有关节初始化 0 rad
        ),

        # 关节电机底层 PD 控制器
        
        actuators={
            "active_fingers": ImplicitActuatorCfg(
                joint_names_expr=O6_CONTROL_JOINTS, # 显式指定 6 个主动关节

                # --- 1. PD-Kp/Kd ---
                stiffness=80,  # PD-Kp；Kp 调大保证位置追踪（否则重力都克服不了）；调低增强接触柔顺性
                damping=8,     # PD-Kd；Kd 调大吸收震荡
                
                # --- 2. 模拟 set_torque=[94]*6 的物理约束 ---
                # 【核心对应 sdk：set_torque】 限制物理引擎输出的 PD 最大扭矩（单位 N*m）
                # 官方产品手册关节极限扭矩为 1.5 N·m
                # sdk 上限对应 94/255 大约是 37%
                # 即 sdk 中 set_torque=94 对应 仿真 effort_limit_sim = 94/255*1.5
                effort_limit_sim=0.55,  # 10, if fingers don't move, try 20~50   

                # --- 3. 模拟 set_speed=[120]*6 的物理约束 ---
                # 【核心对应 set_speed】限制最大角速度（单位 rad/s）
                # 假设电机空载极限转速为 6.0 rad/s，120/255 大约是 47%
                # 物理引擎底层将通过约束冲量(Constraint Impulse)强制钳位角速度
                velocity_limit_sim=2.8, # 20
            ) # TODO: 需根据仿真表现微调参数；
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
    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)}, # Domain Randomization，相对 ball 初始位置
            "velocity_range": {}, # 不随机速度，且=0
        },
    )

    # 重置手部关节
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0), # 不随机手的位置，且严格回到 0 rad，仅随机 ball 位置
            "velocity_range": (0.0, 0.0), # 不随机速度，且=0
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