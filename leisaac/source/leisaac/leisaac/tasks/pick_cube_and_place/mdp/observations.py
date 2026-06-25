import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


_GRASP_TOOL_CENTER_Z_ABOVE_CUBE = 0.062
_GRASP_XY_THRESHOLD = 0.035
_GRASP_Z_THRESHOLD = 0.030


def _target_frame_pos_w(ee_frame: FrameTransformer, frame_name: str) -> torch.Tensor:
    frame_names = list(ee_frame.data.target_frame_names)
    if frame_name not in frame_names:
        raise ValueError(f"FrameTransformer target frame '{frame_name}' not found: {frame_names}")
    return ee_frame.data.target_pos_w[:, frame_names.index(frame_name), :]


def ball_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    diff_threshold: float = 0.045,
    grasp_threshold: float = -0.75,
) -> torch.Tensor:
    """Check if the cube is close to the jaw frame while the gripper is closed."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ball: RigidObject = env.scene[object_cfg.name]

    ball_pos = ball.data.root_pos_w
    end_effector_pos = _target_frame_pos_w(ee_frame, "jaw")
    delta = ball_pos - end_effector_pos
    xy_diff = torch.linalg.vector_norm(delta[:, :2], dim=1)
    z_error = torch.abs(delta[:, 2] + _GRASP_TOOL_CENTER_Z_ABOVE_CUBE)
    close_to_grasp_pose = torch.logical_and(xy_diff < _GRASP_XY_THRESHOLD, z_error < _GRASP_Z_THRESHOLD)

    return torch.logical_and(close_to_grasp_pose, robot.data.joint_pos[:, -1] > grasp_threshold)


def ball_in_target_square(
    env: ManagerBasedRLEnv | DirectRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_center: tuple[float, float] = (0.30, 0.12),
    target_half_size: float = 0.075,
    height_range: tuple[float, float] = (0.010, 0.060),
) -> torch.Tensor:
    """Check if the ball center is inside the black square marker."""
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_x = ball.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    ball_y = ball.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    ball_z = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    in_x = torch.logical_and(ball_x > target_center[0] - target_half_size, ball_x < target_center[0] + target_half_size)
    in_y = torch.logical_and(ball_y > target_center[1] - target_half_size, ball_y < target_center[1] + target_half_size)
    in_z = torch.logical_and(ball_z > height_range[0], ball_z < height_range[1])
    return torch.logical_and(torch.logical_and(in_x, in_y), in_z)


def ball_released_in_target_square(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_center: tuple[float, float] = (0.30, 0.12),
    target_half_size: float = 0.075,
    grasp_threshold: float = -0.75,
) -> torch.Tensor:
    """Check if the cube is inside the square after the gripper has opened."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball_inside = ball_in_target_square(
        env,
        ball_cfg=ball_cfg,
        target_center=target_center,
        target_half_size=target_half_size,
    )
    gripper_open = robot.data.joint_pos[:, -1] < grasp_threshold
    return torch.logical_and(ball_inside, gripper_open)
