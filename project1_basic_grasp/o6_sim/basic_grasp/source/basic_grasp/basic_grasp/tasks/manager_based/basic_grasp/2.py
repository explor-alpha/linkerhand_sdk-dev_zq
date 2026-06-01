# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 使用 isaac lab - Manager-Based; single-agent 模块化模板 - skrl算法库
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import mdp
# TODO: 定义 mdp

##
# Scene definition
##

@configclass
class BasicGraspSceneCfg(InteractiveSceneCfg):
    """Configuration for the dexterous hand grasping scene."""

    # 1. 地面
    # TODO: 地板100x100容纳的下并行训练？
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 2. 桌子 (匹配你 PDF 中的尺寸: 0.8x0.8x0.2, 高度设为 0.1 刚好贴地)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.2),
            rigid_props=None,  # 不添加刚体属性，这样它就会像你设定的一样固定在空中
            collision_props=sim_utils.CollisionPropertiesCfg(), # 开启碰撞
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)), # 棕色木头色
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )

    # 3. 待抓取的小球 (匹配 PDF 设定: 悬浮、高密度、高摩擦)
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.03, # 对应你说的 scale 为 0.1 时的半径
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, # 禁用重力，如你 PDF 中所设
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=10.0), # 密度 10
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0, dynamic_friction=2.0 # 摩擦力 2.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05233, 0.0, 0.3)), # PDF 中的坐标
    )

    # 4. 灵巧手 (Articulation)
    # TODO: 请将 usd_path 替换为你导出的手部 usd 文件的实际绝对路径或服务器路径！
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/qunz/path_to_your_linker_06_hand.usd", 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.1, 0.0, 0.2), # PDF 中设定的手掌初始位姿
        ),
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
    """Action specifications for the MDP."""
    # 灵巧手控制：因为你设置了 Mimic，这里只控制主动关节 (MCP 和 Thumb 根部)
    # 我们使用位置控制 (Position Control) 而不是力矩控制，更稳定。
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        # TODO: 核对这些主动关节名称是否与你 USD 中的完全一致
        joint_names=[
            "rh_index_mcp_pitch", 
            "rh_middle_mcp_pitch", 
            "rh_pinky_mcp_pitch", 
            "rh_ring_mcp_pitch",
            "rh_thumb_cmc_yaw",
            "rh_thumb_cmc_pitch"
        ], 
        scale=1.0, 
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # 1. 关节位置与速度
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # 2. 目标小球的位置 (如果是盲抓)
        ball_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball")})
        
        # 注意：深度相机的图像数据不能直接和上面的低维数据连结 (concatenate)，
        # 如果要用相机，需要在算法库 (如 skrl) 层面进行特征提取分离，这里暂且用低维状态。
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events (Domain Randomization & Resets)."""
    # 每回合重置小球到一个随机位置
    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)}, # z轴保持悬浮高度
            "velocity_range": {},
        },
    )
    
    # 重置手部关节
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # (1) 靠近奖励: 手掌距离小球越近，奖励越高
    approach_ball = RewTerm(
        func=mdp.position_command_error, # 这是一个替代函数，需要你后续在 mdp/rewards.py 中自定义
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["rh_hand_base_link"]), "command_name": "ball_pos"}
    )
    # (2) 动作惩罚: 避免手部乱动抽搐
    action_penalty = RewTerm(func=mdp.action_l2, weight=-0.01)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) 如果小球掉落出界则结束 (虽然禁用了重力，但防止手把它拍飞)
    ball_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("ball")},
    )

##
# Environment configuration
##

@configclass
class BasicGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: BasicGraspSceneCfg = BasicGraspSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2 # 控制频率
        self.episode_length_s = 5 # 每回合 5 秒
        self.viewer.eye = (0.5, 0.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.sim.dt = 1 / 60 # 60Hz 物理仿真
        self.sim.render_interval = self.decimation