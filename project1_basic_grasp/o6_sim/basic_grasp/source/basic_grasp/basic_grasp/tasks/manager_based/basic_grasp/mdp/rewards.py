# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def tip_to_ball_rel(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[Observation] 计算指尖到小球的相对位置向量 (N, 15)"""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    # 获取球的世界坐标 [N, 1, 3]
    ball_pos_w = ball.data.root_pos_w.unsqueeze(1)
    # 获取指尖的世界坐标 [N, 5, 3]
    tip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    
    # 相对向量计算
    rel = ball_pos_w - tip_pos_w
    return rel.view(env.num_envs, -1)  # 展平为 [N, 15]


def root_pos_out_of_bounds(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    max_distance: float
) -> torch.Tensor:
    """
    [Done] 越界终止：判定目标球是否偏离初始位置超过阈值。
    利用 env_origins 修复了 Isaac Lab 并行环境下的 World vs Local 坐标系匹配问题。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 1. 获取当前小球的绝对世界坐标 [num_envs, 3]
    current_pos_w = asset.data.root_pos_w
    
    # 2. 获取小球的默认【局部坐标】(即相对于各自环境中心点的偏移) [num_envs, 3]
    default_pos_local = asset.data.default_root_state[:, :3]
    
    # 3. 获取这 1024 个环境各自的中心点世界坐标 [num_envs, 3]
    env_origins = env.scene.env_origins
    
    # 4. 真实默认世界坐标 = 环境中心点坐标 + 局部偏移坐标
    default_pos_w = env_origins + default_pos_local
    
    # 5. 计算当前世界坐标与默认世界坐标的真实物理欧式距离 [num_envs]
    distance = torch.norm(current_pos_w - default_pos_w, dim=-1)
    
    # 6. 判断位移是否大于阈值
    return distance > max_distance

def fingertip_distance_reward(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    ball_cfg: SceneEntityCfg,
    bounds: tuple[float, float] = (0.000, 0.020),  # 核心区：指尖距离小球 2cm 内即视为满分
    margin: float = 0.070,                        # 引导区：7cm 
    sigmoid: str = "gaussian"                  
) -> torch.Tensor:
    """[Reward] 基于 Tolerance 映射的指尖到小球距离奖励 (输出范围 [0, 1])"""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    ball_pos_w = ball.data.root_pos_w.unsqueeze(1)
    tip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    
    # 计算欧式距离 [N, 5]
    d = torch.norm(ball_pos_w - tip_pos_w, dim=-1)
    
    # [新增] 将距离丢入 tolerance 映射 
    # 指尖到小球的距离越接近 bounds，reward_per_finger 越接近 1.0
    reward_per_finger = tolerance_torch(
        x=d,
        bounds=bounds,
        margin=margin,
        sigmoid=sigmoid
    )
    
    # 返回 5 根手指的平均 tolerance 得分
    return reward_per_finger.mean(dim=-1)

def antipodal_grasp_reward(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    ball_cfg: SceneEntityCfg,
    thumb_name: str = "rh_thumb_distal",  # 大拇指的刚体名称
    other_fingers: list[str] = ["rh_index_distal", "rh_middle_distal", "rh_ring_distal"] # 其他手指
) -> torch.Tensor:
    """
    [Reward] 对立抓取（Antipodal Grasp）奖励。
    计算大拇指到球心的向量，以及其他手指（质心）到球心的向量。
    通过向量的点积（Cosine Similarity），强制要求它们从球的【两端】夹击。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    # 1. 获取小球的世界坐标 [num_envs, 3]
    ball_pos_w = ball.data.root_pos_w
    
    # 2. 找到大拇指和其他手指在 body_names 中的索引
    body_names = robot.data.body_names
    thumb_idx = body_names.index(thumb_name)
    other_idx = [body_names.index(name) for name in other_fingers]
    
    # 3. 获取手指的世界坐标 [num_envs, 3]
    thumb_pos_w = robot.data.body_pos_w[:, thumb_idx, :]
    other_pos_w = robot.data.body_pos_w[:, other_idx, :] # [num_envs, 3, 3]
    
    # 4. 计算其他手指的“几何中心”（质心）坐标 [num_envs, 3]
    # 这样就把食指、中指、无名指当成一个“大板子”来和拇指对抗
    other_center_w = other_pos_w.mean(dim=1)
    
    # 5. 计算指向球心的方向向量
    # 向量：大拇指 -> 球心
    vec_thumb_to_ball = ball_pos_w - thumb_pos_w
    # 向量：其他手指中心 -> 球心
    vec_other_to_ball = ball_pos_w - other_center_w
    
    # 6. 向量归一化 (转换为单位向量，长度为 1)
    dir_thumb = torch.nn.functional.normalize(vec_thumb_to_ball, p=2, dim=-1)
    dir_other = torch.nn.functional.normalize(vec_other_to_ball, p=2, dim=-1)
    
    # 7. 计算点积 (Cosine Similarity) [-1.0, 1.0]
    # 如果大拇指和其他手指在球的同一侧（搓球），方向相同，点积接近 1.0
    # 如果它们在球的两侧（完美对立抓取），方向相反，点积接近 -1.0
    cos_sim = torch.sum(dir_thumb * dir_other, dim=-1)
    
    # 8. 奖励映射：点积越接近 -1.0，得分越接近 1.0
    # 我们用 tolerance，把目标设为 -1.0（完美对立）
    reward = tolerance_torch(
        x=cos_sim,
        bounds=(-1.0, -0.8),  # 核心区：只要夹角大于 143 度，就给满分
        margin=1.8,           # 引导区：从 1.0 (同侧) 一路降到 -0.8
        sigmoid="linear"      # 线性引导它走向对立面
    )
    
    return reward

def fingertip_distance_variance_reward(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    ball_cfg: SceneEntityCfg,
    margin: float = 0.05  # 引导区：标准差在 5cm 内时开始给分
) -> torch.Tensor:
    """[Reward] 指尖距离方差奖励 (促使多指同时到达、对称合围)"""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    ball_pos_w = ball.data.root_pos_w.unsqueeze(1)
    tip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    
    # 1. 计算每个指尖到球心的欧式距离 -> 形状 [N, 5]
    d = torch.norm(ball_pos_w - tip_pos_w, dim=-1)
    
    # 2. 计算各手指距离的标准差 -> 形状 [N]
    # 标准差越小，说明各手指距离球的远近越一致，合围姿态越完美
    d_std = torch.std(d, dim=-1)
    
    # 3. 将标准差丢入 tolerance 映射为 [0, 1] 的正向奖励
    reward = tolerance_torch(
        x=d_std,
        bounds=(0.0, 0.0),    # 核心区：标准差为 0 是最完美的对称
        margin=margin,        # 缓冲区：标准差越接近 0，得分从 0 飙升到 1
        sigmoid="gaussian"    # 高斯曲线顶部平滑，鼓励精细微调
    )
    
    return reward

def fingertip_contact_reward(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float
) -> torch.Tensor:
    """[Reward] 接触力奖励 (利用 tolerance 提供平滑的力引导)"""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # 获取每个指尖的合力大小 [N, 5]
    fmag = torch.norm(forces, dim=-1)
    
    # 之前是阶跃函数 (0 或 1)，现在用 tolerance 给出连续梯度
    # 目标区：力 >= threshold (闭区间上界设为一个极大的值 10000.0)
    # 缓冲区：力从 0 到 threshold 的过程
    reward_per_finger = tolerance_torch(
        x=fmag,
        bounds=(threshold, 10000.0), # 只要力达到阈值，就给满分 1.0
        margin=threshold,            # 让 0 到 threshold 之间有平滑递增的梯度
        sigmoid="linear"             # 使用线性递增，直接且明确地鼓励施加力
    )
    return reward_per_finger.mean(dim=-1)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[Penalty] 动作平滑度惩罚 (映射至 [-1.0, 0.0])"""
    # 获取当前步输出的动作 (通常在 [-1, 1] 之间) [N, num_actions]
    actions = env.action_manager.action
    action_l2 = torch.norm(actions, dim=-1)
    
    # perfect_score 在 [0, 1] 之间：动作 L2 越接近 0，得分越接近 1.0
    perfect_score = tolerance_torch(
        x=action_l2,
        bounds=(0.0, 0.0),
        margin=2.0,           # 动作 L2 范数的容忍边缘
        sigmoid="quadratic"   # 抛物线衰减，对小动作宽容，对大动作惩罚剧增
    )
    # 减 1 将其转化为纯惩罚，防止智能体为了拿这部分分而原地挂机
    return perfect_score - 1.0

"""

def joint_vel_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    max_vel: float = 2.0
) -> torch.Tensor:
    # [Penalty] 关节速度惩罚 (映射至 [-1.0, 0.0])
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取受控关节的角速度 [N, num_joints]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_l2 = torch.norm(joint_vel, dim=-1)
    
    perfect_score = tolerance_torch(
        x=vel_l2,
        bounds=(0.0, 0.0),
        margin=max_vel,       # 超过 max_vel 后，perfect_score 逼近 0
        sigmoid="gaussian"    # 高斯曲线，越靠近 0 越平滑
    )
    return perfect_score - 1.0

"""


def success_termination(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    ball_cfg: SceneEntityCfg, 
    sensor_cfg: SceneEntityCfg, 
    contact_thresh: float, 
    min_contacts: int = 4,      # 4 指
    max_ball_vel: float = 0.05  # 核心补丁：小球线速度必须极小 (即静力平衡)
) -> torch.Tensor:
    """[Done] 成功条件：4指接触力达标 且 小球处于静力平衡(速度极小)"""
    ball: RigidObject = env.scene[ball_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name] 

    # 1. 接触力判断
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    fmag = torch.norm(forces, dim=-1)
    # 计算有几根手指的力大于阈值
    contact_ok = (fmag > contact_thresh).sum(dim=-1) >= min_contacts
    
    # 2. 静力平衡判断 (防止“拍击”作弊)
    # 获取小球当前的世界线速度 [N, 3]
    ball_vel_w = ball.data.root_vel_w[:, :3] 
    ball_speed = torch.norm(ball_vel_w, dim=-1)
    # 速度必须极其微小，证明球被稳稳“锁”住了
    velocity_ok = ball_speed < max_ball_vel
    
    return contact_ok & velocity_ok


def success_bonus(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    ball_cfg: SceneEntityCfg, 
    sensor_cfg: SceneEntityCfg, 
    contact_thresh: float, 
    min_contacts: int = 4,      
    max_ball_vel: float = 0.05  
) -> torch.Tensor:
    """
    [Reward] 成功补偿奖励：当满足成功终止条件时，给予单次巨额奖励。
    底层直接复用 success_termination，保证判定逻辑的绝对统一。
    """
    # 直接调用我们之前写好的 success_termination 函数 (复用其张量计算)
    is_success = success_termination(
        env=env,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        sensor_cfg=sensor_cfg,
        contact_thresh=contact_thresh,
        min_contacts=min_contacts,
        max_ball_vel=max_ball_vel
    )
    
    # 将 bool 张量转换为 float 张量 (True -> 1.0, False -> 0.0)
    return is_success.float()


def tolerance_torch(
    x: torch.Tensor,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    """
    PyTorch 版本的 Tolerance 奖励函数，支持 GPU 并行运算。
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("lower bound must be less than upper bound")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    # 检查 x 是否落在指定的 [lower, upper] 闭区间内
    in_bounds = torch.logical_and(lower <= x, x <= upper)
    
    if margin == 0.0:
        return torch.where(in_bounds, torch.ones_like(x), torch.zeros_like(x))

    # 计算超出边界的距离并根据 margin 归一化
    d = torch.where(x < lower, lower - x, x - upper) / margin

    # Sigmoid 平滑计算
    if sigmoid == "gaussian":
        scale = math.sqrt(-2 * math.log(value_at_margin))
        v = torch.exp(-0.5 * (d * scale) ** 2)
    elif sigmoid == "long_tail":
        scale = math.sqrt(1 / value_at_margin - 1)
        v = 1 / ((d * scale) ** 2 + 1)
    elif sigmoid == "tanh_squared":
        scale = math.atanh(math.sqrt(1 - value_at_margin))
        v = 1 - torch.tanh(d * scale) ** 2
    elif sigmoid == "hyperbolic":
        scale = math.acosh(1 / value_at_margin)
        v = 1 / torch.cosh(d * scale)
    elif sigmoid == "linear":
        scale = 1 - value_at_margin
        scaled_d = d * scale
        v = torch.where(torch.abs(scaled_d) < 1, 1 - scaled_d, torch.zeros_like(x))
    elif sigmoid == "quadratic":
        scale = math.sqrt(1 - value_at_margin)
        scaled_d = d * scale
        v = torch.where(torch.abs(scaled_d) < 1, 1 - scaled_d**2, torch.zeros_like(x))
    else:
        raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")

    # 在边界内直接给 1.0，边界外按衰减曲线给分
    return torch.where(in_bounds, torch.ones_like(x), v)