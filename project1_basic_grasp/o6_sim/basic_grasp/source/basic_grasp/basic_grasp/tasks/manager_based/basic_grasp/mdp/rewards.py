# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
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

def fingertip_distance_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[Reward] 惩罚指尖到小球的距离 (返回负的平均距离)"""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    ball_pos_w = ball.data.root_pos_w.unsqueeze(1)
    tip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    
    # 计算欧式距离 [N, 5]
    d = torch.norm(ball_pos_w - tip_pos_w, dim=-1)
    return -d.mean(dim=-1)

def fingertip_contact_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """[Reward] 奖励大于给定阈值的指尖接触数量比例"""
    robot: Articulation = env.scene[robot_cfg.name]
    # 获取接触力 [N, 5, 3]
    forces = robot.data.net_contact_forces_w[:, robot_cfg.body_ids, :]
    fmag = torch.norm(forces, dim=-1)
    # 计算接触数并取平均
    contacts = (fmag > threshold).float()
    return contacts.mean(dim=-1)

def success_termination(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    ball_cfg: SceneEntityCfg, 
    dist_thresh: float, 
    contact_thresh: float, 
    min_contacts: int
) -> torch.Tensor:
    """[Done] 成功条件：平均距离达标 且 接触力达标指尖数足够"""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    # 距离判断
    ball_pos_w = ball.data.root_pos_w.unsqueeze(1)
    tip_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    d = torch.norm(ball_pos_w - tip_pos_w, dim=-1)
    dist_ok = d.mean(dim=-1) < dist_thresh
    
    # 接触力判断
    #forces = robot.data.net_contact_forces_w[:, robot_cfg.body_ids, :]
    #fmag = torch.norm(forces, dim=-1)
    #contact_ok = (fmag > contact_thresh).sum(dim=-1) >= min_contacts
    
    return dist_ok #& contact_ok