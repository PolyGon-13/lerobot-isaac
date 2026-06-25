from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .observations import ball_in_target_square

_REST_POSE_DEG = {
    "base": 0.0,
    "shoulder": 0.0,
    "elbow": 0.0,
    "wrist_pitch": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
_REST_POSE_TOLERANCE_DEG = 30.0


def _robot_at_rest_pose(joint_pos: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    """Check the custom arm against its default all-zero rest pose."""
    is_reset = torch.ones(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)
    joint_pos_deg = joint_pos / torch.pi * 180.0

    for joint_name, target_deg in _REST_POSE_DEG.items():
        if joint_name not in joint_names:
            continue
        joint_idx = joint_names.index(joint_name)
        lower = target_deg - _REST_POSE_TOLERANCE_DEG
        upper = target_deg + _REST_POSE_TOLERANCE_DEG
        is_reset = torch.logical_and(
            is_reset,
            torch.logical_and(joint_pos_deg[:, joint_idx] > lower, joint_pos_deg[:, joint_idx] < upper),
        )
    return is_reset


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    ball_cfg: SceneEntityCfg,
    target_center: tuple[float, float] = (0.30, 0.12),
    target_half_size: float = 0.075,
    height_range: tuple[float, float] = (0.010, 0.060),
    require_rest: bool = True,
) -> torch.Tensor:
    """Return True when the ball is inside the black square and, optionally, the arm is at rest."""
    done = ball_in_target_square(
        env,
        ball_cfg=ball_cfg,
        target_center=target_center,
        target_half_size=target_half_size,
        height_range=height_range,
    )

    if require_rest:
        joint_pos = env.scene["robot"].data.joint_pos
        joint_names = env.scene["robot"].data.joint_names
        done = torch.logical_and(done, _robot_at_rest_pose(joint_pos, joint_names))

    return done
