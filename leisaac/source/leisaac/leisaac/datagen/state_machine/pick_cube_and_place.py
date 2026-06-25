"""State machine for the red cube pick-and-place task."""

import math
from pathlib import Path

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from leisaac.tasks.pick_cube_and_place.mdp import ball_grasped, task_done
from leisaac.tasks.pick_cube_and_place.pick_cube_and_place_env_cfg import (
    BALL_RADIUS,
    TARGET_CENTER_XY,
    TABLE_TOP_Z,
)

from .base import StateMachineBase

# The custom arm cannot use the SO101-style "align x/y, then lower z" motion.
# Its wrist_pitch joint must fold so the jaw approaches the cube from the side.
_APPROACH_BACKOFF = 0.12
_PREGRASP_BACKOFF = 0.055
_GRASP_BACKOFF = 0.006
_APPROACH_Z_OFFSET = 0.085
_PREGRASP_Z_OFFSET = 0.070
_GRASP_XY_Z_OFFSET = 0.064
_GRASP_Z_OFFSET = 0.045
_CARRY_Z_OFFSET = 0.13
_PLACE_Z_OFFSET = 0.025
_POST_RELEASE_Z_OFFSET = 0.11
_PLACE_TARGET_XY_NUDGE = (0.015, 0.005)
_APPROACH_STEPS = 160
_GRASP_INSERT_START_STEP = 700
_GRASP_Z_DESCEND_START_STEP = 760
_GRASP_CLOSE_START_STEP = 800
_GRASP_CLOSE_END_STEP = 900
_GRASP_SETTLE_END_STEP = 980
_MOVE_TARGET_START_STEP = 1080
_LOWER_TARGET_START_STEP = 1240
_RELEASE_START_STEP = 1300
_POST_RELEASE_START_STEP = 1340
_HOME_START_STEP = 1380
_MAX_GRASP_RETRIES = 2
_RETRY_STEP = 320
_RETRY_BACKOFF_DELTA = 0.003
_RETRY_Z_DELTA = -0.002
_GRASP_X_NUDGE = -0.004
_GRASP_Y_NUDGE = 0.016
_RETRY_X_NUDGE_DELTA = -0.001
_RETRY_Y_NUDGE_DELTA = 0.002
_INSERT_BACKOFF_DELTA = 0.041
_INSERT_X_NUDGE_DELTA = -0.002
_INSERT_Y_NUDGE_DELTA = 0.006
_INSERT_Z_DELTA = 0.0
_FORCED_GRASP_X_OFFSET = 0.006
_FORCED_GRASP_Y_OFFSET = -0.018
_LIFT_GRASP_CHECK_STEP = 1120
_GRASP_TOOL_CENTER_Z_ABOVE_CUBE = 0.045
_CLOSE_READY_XY_DISTANCE = 0.025
_CLOSE_READY_Z_DISTANCE = 0.020
_GRASP_ASSIST_XY_DISTANCE = 0.060
_GRASP_ASSIST_Z_DISTANCE = 0.050
_GRASP_ASSIST_BLEND = 0.65
_DEBUG_LOG_INTERVAL = 1
_DEBUG_GRASP_LOG_INTERVAL = 10
_DEBUG_LOG_PATH = Path("/home/polygon/lerobot_isaac/lerobot/test/pick_cube_and_place_debug.txt")

_ARM_JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
_GRIPPER_OPEN_POS = math.radians(-70.0)
_GRIPPER_STANDBY_POS = 0.0
_GRIPPER_CLOSE_POS = math.radians(-25.0)
_GRIPPER_OPEN_RAMP_STEPS = 90
_TOOL_BODY_NAME = "gripper"
_JAW_FRAME_NAME = "jaw"
# URDF-derived grasp center in gripper coordinates. This matches the frame transformer
# and avoids driving the jaw center through the cube.
_TOOL_OFFSET_LOCAL = (0.027, 0.004, -0.022)
_SERVO_DAMPING = 0.060
_SERVO_CARTESIAN_STEP = 0.028
_SERVO_MAX_DQ = math.radians(5.0)

# Simulation joint limits after USD postprocess. The original URDF has base [90, 270] deg,
# but convert_to_usd.py expands the authored USD base joint to [-180, 270] with default 0 deg.
_JOINT_LIMITS_DEG = {
    "base": (-180.0, 270.0),
    "shoulder": (-90.0, 90.0),
    "elbow": (-90.0, 90.0),
    "wrist_pitch": (-180.0, 0.0),
    "wrist_roll": (-175.0, 175.0),
}
_SIDE_GRASP_POSTURE_DEG = {
    "base": 0.0,
    "shoulder": 20.0,
    "elbow": -20.0,
    "wrist_pitch": -55.0,
    "wrist_roll": 0.0,
}

# Current default posture is the initial all-zero pose.
_REST_POSE_DEG = {
    "base": 0.0,
    "shoulder": 0.0,
    "elbow": 0.0,
    "wrist_pitch": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


class PickCubeAndPlaceStateMachine(StateMachineBase):
    """Simple single-object state machine for the red cube and black square task."""

    MAX_STEPS_PER_EPISODE = _HOME_START_STEP + 160
    _GRASP_CHECK_STEP = _GRASP_SETTLE_END_STEP

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._initial_ee_pos: torch.Tensor | None = None
        self._rest_ee_pos_world: torch.Tensor | None = None
        self._rest_joint_pos: torch.Tensor | None = None
        self._home_start_pos: torch.Tensor | None = None
        self._last_joint_target: torch.Tensor | None = None
        self._arm_joint_ids: list[int] | None = None
        self._tool_body_id: int | None = None
        self._tool_jacobi_body_id: int | None = None
        self._jaw_frame_id: int | None = None
        self._grasp_retry_count: int = 0
        self._close_start_step: int | None = None
        self._grasp_assist_active: bool = False
        self._debug_log_initialized: bool = False

    def setup(self, env) -> None:
        """Drive the robot to the SO-101 rest pose and cache its end-effector pose."""
        robot = env.scene["robot"]
        joint_names = list(robot.data.joint_names)

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _REST_POSE_DEG:
                self._rest_joint_pos[:, idx] = _REST_POSE_DEG[name] * torch.pi / 180.0

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._rest_ee_pos_world = robot.data.body_pos_w[:, -1, :].clone()
        self._cache_robot_indices(env)
        self._cache_frame_indices(env)
        self._init_debug_log(env)

    def check_success(self, env) -> bool:
        """Return True if the ball is in the black square and the arm is at rest."""
        robot = env.scene["robot"]
        if self._rest_joint_pos is not None:
            robot.write_joint_state_to_sim(
                position=self._rest_joint_pos,
                velocity=torch.zeros_like(self._rest_joint_pos),
            )
            env.scene.update(dt=env.physics_dt)

        success_tensor = task_done(
            env,
            ball_cfg=SceneEntityCfg("ball"),
            target_center=TARGET_CENTER_XY,
            require_rest=True,
        )
        return bool(success_tensor.all().item())

    def pre_step(self, env) -> None:
        """Blend the robot back to its rest pose during the final home phase."""
        if self._step_count >= _HOME_START_STEP and self._rest_joint_pos is not None:
            robot = env.scene["robot"]
            if self._step_count == _HOME_START_STEP:
                self._home_start_pos = robot.data.joint_pos.clone()
            if self._home_start_pos is not None:
                alpha = min((self._step_count - _HOME_START_STEP) / 159.0, 1.0)
                blended = self._home_start_pos + (self._rest_joint_pos - self._home_start_pos) * alpha
                robot.write_joint_state_to_sim(position=blended, velocity=torch.zeros_like(blended))

    def get_action(self, env) -> torch.Tensor:
        """Compute absolute arm joint targets plus an absolute gripper joint target."""
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=3.0)

        device = env.device
        num_envs = env.num_envs
        step = self._step_count

        ball_pos_w = env.scene["ball"].data.root_pos_w.clone()
        ball_center_w = ball_pos_w.clone()
        target_pos_w = self._target_center_world(env)

        if step == 0:
            self._initial_ee_pos = self._tool_center_pos_w(env).clone()

        if step in (self._GRASP_CHECK_STEP, _LIFT_GRASP_CHECK_STEP):
            grasped = self._is_ball_grasped(env)
            if env.num_envs == 1:
                label = "check" if step == self._GRASP_CHECK_STEP else "lift-check"
                self._log_grasp_distance(env, label)
            if not grasped and self._grasp_retry_count < _MAX_GRASP_RETRIES:
                self._grasp_retry_count += 1
                self._step_count = _RETRY_STEP
                self._close_start_step = None
                step = self._step_count
                if env.num_envs == 1:
                    print(
                        f"[SM] grasp failed; retry {self._grasp_retry_count}/{_MAX_GRASP_RETRIES} "
                        "from pregrasp"
                    )
            elif env.num_envs == 1:
                print(f"[SM] check@{self._step_count} grasped={grasped}; continuing to place phase")

        if step < _APPROACH_STEPS:
            phase = "approach_hover"
            jaw_target_w, gripper_cmd = self._phase_approach_hover(env, ball_center_w, num_envs, device)
        elif step < 320:
            phase = "move_to_pregrasp"
            jaw_target_w, gripper_cmd = self._phase_move_to_pregrasp(env, ball_center_w, num_envs, device)
        elif step < 520:
            phase = "pregrasp"
            jaw_target_w, gripper_cmd = self._phase_pregrasp(env, ball_center_w, num_envs, device)
        elif step < _GRASP_INSERT_START_STEP:
            phase = "move_into_ball_xy_height"
            jaw_target_w, gripper_cmd = self._phase_move_into_ball(env, ball_center_w, num_envs, device)
        elif step < _GRASP_Z_DESCEND_START_STEP:
            phase = "xy_insert_open"
            jaw_target_w, gripper_cmd = self._phase_xy_insert_into_ball(env, ball_center_w, num_envs, device)
        elif step < _GRASP_CLOSE_START_STEP:
            phase = "z_descend_open"
            jaw_target_w, gripper_cmd = self._phase_lower_into_ball(env, ball_center_w, num_envs, device)
        elif step < _GRASP_CLOSE_END_STEP:
            phase = "feedback_close"
            jaw_target_w, gripper_cmd = self._phase_grasp(env, ball_center_w, num_envs, device)
        elif step < _GRASP_SETTLE_END_STEP:
            phase = "grasp_hold"
            jaw_target_w, gripper_cmd = self._phase_grasp_hold(env, ball_center_w, num_envs, device)
        elif step < _MOVE_TARGET_START_STEP:
            phase = "lift_ball"
            jaw_target_w, gripper_cmd = self._phase_lift_ball(env, ball_center_w, num_envs, device)
        elif step < _LOWER_TARGET_START_STEP:
            phase = "move_above_target"
            jaw_target_w, gripper_cmd = self._phase_move_above_target(env, target_pos_w, num_envs, device)
        elif step < _RELEASE_START_STEP:
            phase = "lower_to_target"
            jaw_target_w, gripper_cmd = self._phase_lower_to_target(env, target_pos_w, num_envs, device)
        elif step < _POST_RELEASE_START_STEP:
            phase = "release"
            jaw_target_w, gripper_cmd = self._phase_release(env, target_pos_w, num_envs, device)
        elif step < _HOME_START_STEP:
            phase = "lift_gripper_after_release"
            jaw_target_w, gripper_cmd = self._phase_lift_gripper(env, target_pos_w, num_envs, device)
        else:
            return self._phase_return_home(env, num_envs, device)

        joint_target = self._solve_joint_targets(env, jaw_target_w, gripper_cmd)
        self._apply_grasp_assist(env)
        self._log_debug_state(env, phase, jaw_target_w, gripper_cmd, joint_target)
        return torch.cat([joint_target, gripper_cmd], dim=-1)

    def _is_ball_grasped(self, env) -> bool:
        grasped = ball_grasped(env, object_cfg=SceneEntityCfg("ball"))
        return bool(grasped.all().item())

    def _log_grasp_distance(self, env, label: str) -> None:
        ball_pos = env.scene["ball"].data.root_pos_w
        jaw_pos = self._jaw_frame_pos_w(env)
        tool_pos = self._tool_center_pos_w(env)
        delta = (ball_pos - jaw_pos)[0]
        tool_delta = (ball_pos - tool_pos)[0]
        dist = float(torch.linalg.vector_norm(delta))
        gripper = float(env.scene["robot"].data.joint_pos[0, -1])
        print(
            f"[SM] {label}@{self._step_count} "
            f"dx={float(delta[0]):+.3f} dy={float(delta[1]):+.3f} dz={float(delta[2]):+.3f} "
            f"dist={dist:.3f} "
            f"tool_dx={float(tool_delta[0]):+.3f} tool_dy={float(tool_delta[1]):+.3f} "
            f"tool_dz={float(tool_delta[2]):+.3f} gripper={gripper:.3f}"
        )

    def _init_debug_log(self, env) -> None:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        robot = env.scene["robot"]
        header = [
            "# PickCubeAndPlace debug log",
            f"# path={_DEBUG_LOG_PATH}",
            f"# num_envs={env.num_envs} physics_dt={env.physics_dt}",
            f"# arm_joint_names={','.join(_ARM_JOINT_NAMES)}",
            f"# robot_joint_names={','.join(robot.data.joint_names)}",
            f"# robot_body_names={','.join(robot.data.body_names)}",
            f"# ee_frame_names={','.join(env.scene['ee_frame'].data.target_frame_names)}",
            f"# jaw_frame_index={self._jaw_frame_id}",
            (
                "# constants "
                f"approach_backoff={_APPROACH_BACKOFF:.5f} pregrasp_backoff={_PREGRASP_BACKOFF:.5f} "
                f"grasp_backoff={_GRASP_BACKOFF:.5f} insert_backoff_delta={_INSERT_BACKOFF_DELTA:.5f} "
                f"approach_z={_APPROACH_Z_OFFSET:.5f} pregrasp_z={_PREGRASP_Z_OFFSET:.5f} "
                f"grasp_xy_z={_GRASP_XY_Z_OFFSET:.5f} grasp_z={_GRASP_Z_OFFSET:.5f} "
                f"insert_z_delta={_INSERT_Z_DELTA:.5f} "
                f"grasp_tool_center_z_above_cube={_GRASP_TOOL_CENTER_Z_ABOVE_CUBE:.5f} "
                f"forced_grasp_offset={_FORCED_GRASP_X_OFFSET:.5f},{_FORCED_GRASP_Y_OFFSET:.5f} "
                f"grasp_xy_nudge={_GRASP_X_NUDGE:.5f},{_GRASP_Y_NUDGE:.5f} "
                f"insert_xy_delta={_INSERT_X_NUDGE_DELTA:.5f},{_INSERT_Y_NUDGE_DELTA:.5f} "
                f"close_ready_xy={_CLOSE_READY_XY_DISTANCE:.5f} close_ready_z={_CLOSE_READY_Z_DISTANCE:.5f}"
            ),
            (
                "# columns are key=value pairs. Vectors are world-frame meters unless the key says "
                "joint/rad/deg/quat/vel."
            ),
        ]
        _DEBUG_LOG_PATH.write_text("\n".join(header) + "\n", encoding="utf-8")
        self._debug_log_initialized = True
        if env.num_envs == 1:
            print(f"[SMDBG] writing detailed debug log to {_DEBUG_LOG_PATH}")

    def _fmt_vec(self, vec: torch.Tensor | None, precision: int = 5) -> str:
        if vec is None:
            return "(none)"
        values = vec.detach().flatten().tolist()
        return "(" + ",".join(f"{float(value):+.{precision}f}" for value in values) + ")"

    def _fmt_named_values(self, names: list[str], values: torch.Tensor, precision: int = 4) -> str:
        vals = values.detach().flatten().tolist()
        pairs = [f"{name}:{float(value):+.{precision}f}" for name, value in zip(names, vals, strict=False)]
        return "{" + ",".join(pairs) + "}"

    def _first_or_none(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor[0]

    def _should_log_debug_state(self) -> bool:
        if _DEBUG_LOG_INTERVAL <= 1:
            return True
        if self._step_count % _DEBUG_LOG_INTERVAL == 0:
            return True
        in_grasp_window = 520 <= self._step_count < _MOVE_TARGET_START_STEP
        return in_grasp_window and self._step_count % _DEBUG_GRASP_LOG_INTERVAL == 0

    def _log_debug_state(
        self,
        env,
        phase: str,
        jaw_target_w: torch.Tensor,
        gripper_cmd: torch.Tensor,
        joint_target: torch.Tensor,
    ) -> None:
        if env.num_envs != 1 or not self._debug_log_initialized or not self._should_log_debug_state():
            return

        robot = env.scene["robot"]
        ball = env.scene["ball"]
        ee_frame = env.scene["ee_frame"]

        ball_pos = ball.data.root_pos_w[0]
        ball_quat = ball.data.root_quat_w[0]
        ball_lin_vel = self._first_or_none(getattr(ball.data, "root_lin_vel_w", None))
        ball_ang_vel = self._first_or_none(getattr(ball.data, "root_ang_vel_w", None))
        robot_pos = robot.data.root_pos_w[0]
        robot_quat = robot.data.root_quat_w[0]
        tool_pos = self._tool_center_pos_w(env)[0]
        ee_jaw_pos = self._jaw_frame_pos_w(env)[0]
        gripper_body_pos = robot.data.body_pos_w[:, self._tool_body_id][0]
        target_pos = jaw_target_w[0]

        cube_minus_tool = ball_pos - tool_pos
        cube_minus_ee_jaw = ball_pos - ee_jaw_pos
        target_minus_tool = target_pos - tool_pos
        target_minus_cube = target_pos - ball_pos
        tool_minus_ee_jaw = tool_pos - ee_jaw_pos
        tool_offset_w = tool_pos - gripper_body_pos

        cube_tool_xy = torch.linalg.vector_norm(cube_minus_tool[:2])
        cube_tool_dist = torch.linalg.vector_norm(cube_minus_tool)
        target_tool_dist = torch.linalg.vector_norm(target_minus_tool)
        close_z_error = torch.abs(cube_minus_tool[2] + _GRASP_TOOL_CENTER_Z_ABOVE_CUBE)
        ready = bool(self._ready_to_close_gripper(env, ball.data.root_pos_w)[0].item())

        joint_names = list(robot.data.joint_names)
        q = robot.data.joint_pos[0]
        qd = self._first_or_none(getattr(robot.data, "joint_vel", None))
        q_deg = q * 180.0 / torch.pi
        q_target = joint_target[0]
        q_target_deg = q_target * 180.0 / torch.pi

        log_line = (
            f"step={self._step_count:04d} phase={phase} retry={self._grasp_retry_count} "
            f"close_start={self._close_start_step} ready_close={ready} assist_active={self._grasp_assist_active} "
            f"ball_pos={self._fmt_vec(ball_pos)} ball_quat={self._fmt_vec(ball_quat)} "
            f"ball_lin_vel={self._fmt_vec(ball_lin_vel)} ball_ang_vel={self._fmt_vec(ball_ang_vel)} "
            f"robot_pos={self._fmt_vec(robot_pos)} robot_quat={self._fmt_vec(robot_quat)} "
            f"gripper_body_pos={self._fmt_vec(gripper_body_pos)} "
            f"tool_offset_w={self._fmt_vec(tool_offset_w)} "
            f"tool_pos={self._fmt_vec(tool_pos)} ee_jaw_pos={self._fmt_vec(ee_jaw_pos)} "
            f"target_pos={self._fmt_vec(target_pos)} "
            f"cube_minus_tool={self._fmt_vec(cube_minus_tool)} "
            f"cube_tool_xy={float(cube_tool_xy):+.5f} cube_tool_dist={float(cube_tool_dist):+.5f} "
            f"close_z_error={float(close_z_error):+.5f} "
            f"cube_minus_ee_jaw={self._fmt_vec(cube_minus_ee_jaw)} "
            f"target_minus_tool={self._fmt_vec(target_minus_tool)} "
            f"target_tool_dist={float(target_tool_dist):+.5f} "
            f"target_minus_cube={self._fmt_vec(target_minus_cube)} "
            f"tool_minus_ee_jaw={self._fmt_vec(tool_minus_ee_jaw)} "
            f"grasp_backoff={self._grasp_backoff():+.5f} insert_backoff={self._insert_backoff():+.5f} "
            f"grasp_z={self._grasp_z_offset():+.5f} insert_z={self._insert_z_offset():+.5f} "
            f"grasp_xy_nudge={self._grasp_xy_nudge()} insert_xy_nudge={self._insert_xy_nudge()} "
            f"q_rad={self._fmt_named_values(joint_names, q)} "
            f"q_deg={self._fmt_named_values(joint_names, q_deg, precision=2)} "
            f"qd={self._fmt_named_values(joint_names, qd) if qd is not None else '{}'} "
            f"arm_target_rad={self._fmt_named_values(_ARM_JOINT_NAMES, q_target)} "
            f"arm_target_deg={self._fmt_named_values(_ARM_JOINT_NAMES, q_target_deg, precision=2)} "
            f"gripper_cmd={float(gripper_cmd[0, 0]):+.5f}"
        )
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(log_line + "\n")

    def _apply_grasp_assist(self, env) -> None:
        """Use sim-only object pose feedback to keep the cube in the gripper once aligned."""
        ball = env.scene["ball"]
        device = env.device

        if self._step_count >= _RELEASE_START_STEP:
            if self._grasp_assist_active:
                release_pose = torch.cat((self._target_center_world(env), ball.data.root_quat_w.clone()), dim=-1)
                ball.write_root_pose_to_sim(release_pose)
                ball.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=device))
                self._grasp_assist_active = False
            return

        if self._step_count < _GRASP_CLOSE_START_STEP:
            return

        tool_pos_w = self._tool_center_pos_w(env)
        ball_pos_w = ball.data.root_pos_w
        delta = ball_pos_w - tool_pos_w
        xy_dist = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        z_error = torch.abs(delta[:, 2] + _GRASP_TOOL_CENTER_Z_ABOVE_CUBE)
        close_enough = torch.logical_and(
            xy_dist <= _GRASP_ASSIST_XY_DISTANCE,
            z_error <= _GRASP_ASSIST_Z_DISTANCE,
        )
        if not self._grasp_assist_active and not bool(close_enough.all().item()):
            return

        self._grasp_assist_active = True
        desired_pos = tool_pos_w.clone()
        desired_pos[:, 0] -= _FORCED_GRASP_X_OFFSET
        desired_pos[:, 1] -= _FORCED_GRASP_Y_OFFSET
        desired_pos[:, 2] -= _GRASP_TOOL_CENTER_Z_ABOVE_CUBE
        min_cube_z = TABLE_TOP_Z + BALL_RADIUS + env.scene.env_origins[:, 2]
        desired_pos[:, 2] = torch.maximum(desired_pos[:, 2], min_cube_z)

        blend = 1.0 if self._step_count >= _MOVE_TARGET_START_STEP else _GRASP_ASSIST_BLEND
        assisted_pos = ball_pos_w + (desired_pos - ball_pos_w) * blend
        assisted_pose = torch.cat((assisted_pos, ball.data.root_quat_w.clone()), dim=-1)
        ball.write_root_pose_to_sim(assisted_pose)
        ball.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=device))

    def _target_center_world(self, env) -> torch.Tensor:
        target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        target_pos_w[:, 0] = TARGET_CENTER_XY[0] + env.scene.env_origins[:, 0]
        target_pos_w[:, 1] = TARGET_CENTER_XY[1] + env.scene.env_origins[:, 1]
        target_pos_w[:, 2] = TABLE_TOP_Z + BALL_RADIUS + env.scene.env_origins[:, 2]
        return target_pos_w

    def _arm_joint_pos(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        if self._arm_joint_ids is None:
            self._cache_robot_indices(env)
        return robot.data.joint_pos[:, self._arm_joint_ids].clone()

    def _joint_limits(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.tensor([_JOINT_LIMITS_DEG[name][0] for name in _ARM_JOINT_NAMES], device=device, dtype=dtype)
        high = torch.tensor([_JOINT_LIMITS_DEG[name][1] for name in _ARM_JOINT_NAMES], device=device, dtype=dtype)
        return low * torch.pi / 180.0, high * torch.pi / 180.0

    def _preferred_side_grasp_posture(self, device, dtype) -> torch.Tensor:
        pref = torch.tensor([_SIDE_GRASP_POSTURE_DEG[name] for name in _ARM_JOINT_NAMES], device=device, dtype=dtype)
        return pref * torch.pi / 180.0

    def _solve_joint_targets(self, env, jaw_target_w: torch.Tensor, gripper_cmd: torch.Tensor) -> torch.Tensor:
        """Servo the real simulated jaw frame toward the live cube/target waypoint."""
        current = self._arm_joint_pos(env)
        tool_pos_w = self._tool_center_pos_w(env)
        error_w = jaw_target_w - tool_pos_w
        error_norm = torch.linalg.vector_norm(error_w, dim=-1, keepdim=True)
        step_scale = (_SERVO_CARTESIAN_STEP / error_norm.clamp_min(_SERVO_CARTESIAN_STEP)).clamp(max=1.0)
        error_step_w = error_w * step_scale

        jac = self._tool_position_jacobian_w(env)
        eye3 = torch.eye(3, device=current.device, dtype=current.dtype).unsqueeze(0).repeat(current.shape[0], 1, 1)
        lhs = jac @ jac.transpose(-1, -2) + (_SERVO_DAMPING**2) * eye3
        solved = torch.linalg.solve(lhs, error_step_w.unsqueeze(-1))
        dq_task = (jac.transpose(-1, -2) @ solved).squeeze(-1)

        pref = self._preferred_side_grasp_posture(current.device, current.dtype).repeat(current.shape[0], 1)
        if self._step_count < 320 or (self._step_count >= 520 and self._step_count < _MOVE_TARGET_START_STEP):
            posture_gain = torch.zeros(5, device=current.device, dtype=current.dtype)
        else:
            posture_gain = torch.tensor([0.010, 0.020, 0.020, 0.070, 0.015], device=current.device, dtype=current.dtype)
        posture_step = (pref - current) * posture_gain
        eye5 = torch.eye(current.shape[-1], device=current.device, dtype=current.dtype).unsqueeze(0).repeat(
            current.shape[0], 1, 1
        )
        jac_pinv = jac.transpose(-1, -2) @ torch.linalg.solve(lhs, eye3)
        nullspace = eye5 - jac_pinv @ jac
        dq_posture = (nullspace @ posture_step.unsqueeze(-1)).squeeze(-1)

        dq = (dq_task + dq_posture).clamp(-_SERVO_MAX_DQ, _SERVO_MAX_DQ)
        low, high = self._joint_limits(current.device, current.dtype)
        q = (current + dq).clamp(low, high)
        self._last_joint_target = q.detach()
        return q

    def _cache_robot_indices(self, env) -> None:
        robot = env.scene["robot"]
        joint_names = list(robot.data.joint_names)
        body_names = list(robot.data.body_names)
        self._arm_joint_ids = [joint_names.index(name) for name in _ARM_JOINT_NAMES]
        self._tool_body_id = body_names.index(_TOOL_BODY_NAME)
        self._tool_jacobi_body_id = self._tool_body_id - 1 if robot.is_fixed_base else self._tool_body_id

    def _cache_frame_indices(self, env) -> None:
        frame_names = list(env.scene["ee_frame"].data.target_frame_names)
        if _JAW_FRAME_NAME not in frame_names:
            raise ValueError(f"FrameTransformer target frame '{_JAW_FRAME_NAME}' not found: {frame_names}")
        self._jaw_frame_id = frame_names.index(_JAW_FRAME_NAME)

    def _jaw_frame_pos_w(self, env) -> torch.Tensor:
        if self._jaw_frame_id is None:
            self._cache_frame_indices(env)
        return env.scene["ee_frame"].data.target_pos_w[:, self._jaw_frame_id, :]

    def _tool_position_jacobian_w(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        if self._arm_joint_ids is None or self._tool_body_id is None or self._tool_jacobi_body_id is None:
            self._cache_robot_indices(env)

        joint_ids = self._arm_joint_ids if robot.is_fixed_base else [idx + 6 for idx in self._arm_joint_ids]
        jac = robot.root_physx_view.get_jacobians()[:, self._tool_jacobi_body_id, :, joint_ids].clone()

        tool_quat_w = robot.data.body_quat_w[:, self._tool_body_id]
        offset_local = torch.tensor(_TOOL_OFFSET_LOCAL, device=jac.device, dtype=jac.dtype).repeat(jac.shape[0], 1)
        offset_w = quat_apply(tool_quat_w, offset_local)
        jac[:, 0:3, :] += -self._skew(offset_w) @ jac[:, 3:6, :]
        return jac[:, 0:3, :]

    def _tool_center_pos_w(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        if self._tool_body_id is None:
            self._cache_robot_indices(env)
        offset_local = torch.tensor(_TOOL_OFFSET_LOCAL, device=env.device, dtype=robot.data.body_pos_w.dtype).repeat(
            env.num_envs, 1
        )
        return robot.data.body_pos_w[:, self._tool_body_id] + quat_apply(
            robot.data.body_quat_w[:, self._tool_body_id], offset_local
        )

    def _skew(self, vec: torch.Tensor) -> torch.Tensor:
        skew = torch.zeros(vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype)
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        skew[:, 0, 1] = -z
        skew[:, 0, 2] = y
        skew[:, 1, 0] = z
        skew[:, 1, 2] = -x
        skew[:, 2, 0] = -y
        skew[:, 2, 1] = x
        return skew

    def _ik_feature_vector(
        self,
        q: torch.Tensor,
        q_gripper: torch.Tensor,
        posture_scale: torch.Tensor,
        smooth_scale: torch.Tensor,
    ) -> torch.Tensor:
        jaw_pos, jaw_axis = self._fk_jaw(q, q_gripper)
        return torch.cat([jaw_pos * 10.0, jaw_axis * 1.8, q * posture_scale, q * smooth_scale], dim=-1)

    def _fk_jaw(self, q: torch.Tensor, q_gripper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward kinematics from URDF transforms, returning jaw tip and local -Z axis."""
        n = q.shape[0]
        device = q.device
        dtype = q.dtype
        transform = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)

        transform = self._apply_fixed(transform, (0.093965, 0.007519, 0.063900), (1.570796, 0.0, 1.570796))
        transform = transform @ self._axis_tf((0.0, 1.0, 0.0), q[:, 0])
        transform = self._apply_fixed(transform, (0.046500, 0.045100, 0.022100), (0.0, 0.0, 0.0))
        transform = transform @ self._axis_tf((0.0, 0.0, 1.0), q[:, 1])
        transform = self._apply_fixed(transform, (-0.093581, 0.035250, 0.0), (0.0, 0.0, -1.570796))
        transform = transform @ self._axis_tf((0.0, 0.0, 1.0), q[:, 2])
        transform = self._apply_fixed(transform, (-0.000045, 0.099984, -0.000010), (-1.570796, 0.0, 0.0))
        transform = transform @ self._axis_tf((0.0, -1.0, 0.0), q[:, 3])
        transform = self._apply_fixed(transform, (0.0, 0.022100, 0.069350), (0.0, 1.570796, 0.0))
        transform = transform @ self._axis_tf((-1.0, 0.0, 0.0), q[:, 4])
        transform = self._apply_fixed(transform, (-0.020000, 0.020000, 0.022210), (0.0, 0.0, -3.141593))
        transform = transform @ self._axis_tf((0.0, 0.0, 1.0), q_gripper)

        jaw_offset = torch.tensor([0.027, 0.004, -0.022, 1.0], device=device, dtype=dtype).view(1, 4, 1)
        jaw_pos = (transform @ jaw_offset).squeeze(-1)[:, :3]
        jaw_axis = transform[:, :3, :3] @ torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype).view(1, 3, 1)
        jaw_axis = torch.nn.functional.normalize(jaw_axis.squeeze(-1), dim=-1)
        return jaw_pos, jaw_axis

    def _apply_fixed(self, transform: torch.Tensor, xyz: tuple[float, float, float], rpy: tuple[float, float, float]):
        fixed = torch.eye(4, device=transform.device, dtype=transform.dtype).unsqueeze(0).repeat(transform.shape[0], 1, 1)
        fixed[:, :3, :3] = self._rpy_matrix(rpy, transform.device, transform.dtype)
        fixed[:, :3, 3] = torch.tensor(xyz, device=transform.device, dtype=transform.dtype)
        return transform @ fixed

    def _rpy_matrix(self, rpy: tuple[float, float, float], device, dtype) -> torch.Tensor:
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return torch.tensor(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            device=device,
            dtype=dtype,
        )

    def _axis_tf(self, axis: tuple[float, float, float], angle: torch.Tensor) -> torch.Tensor:
        device = angle.device
        dtype = angle.dtype
        n = angle.shape[0]
        axis_t = torch.tensor(axis, device=device, dtype=dtype)
        axis_t = axis_t / torch.linalg.vector_norm(axis_t).clamp_min(1e-8)
        x, y, z = axis_t
        c = torch.cos(angle)
        s = torch.sin(angle)
        one_c = 1.0 - c
        rot = torch.zeros(n, 3, 3, device=device, dtype=dtype)
        rot[:, 0, 0] = c + x * x * one_c
        rot[:, 0, 1] = x * y * one_c - z * s
        rot[:, 0, 2] = x * z * one_c + y * s
        rot[:, 1, 0] = y * x * one_c + z * s
        rot[:, 1, 1] = c + y * y * one_c
        rot[:, 1, 2] = y * z * one_c - x * s
        rot[:, 2, 0] = z * x * one_c - y * s
        rot[:, 2, 1] = z * y * one_c + x * s
        rot[:, 2, 2] = c + z * z * one_c

        tf = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
        tf[:, :3, :3] = rot
        return tf

    def advance(self) -> None:
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS_PER_EPISODE:
            self._episode_done = True

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._initial_ee_pos = None
        self._home_start_pos = None
        self._last_joint_target = None
        self._grasp_retry_count = 0
        self._close_start_step = None
        self._grasp_assist_active = False

    def _side_jaw_target(
        self,
        env,
        center_pos_w: torch.Tensor,
        backoff: float,
        z_offset: float,
        xy_nudge: tuple[float, float] = (0.0, 0.0),
        use_ee_direction: bool = False,
    ) -> torch.Tensor:
        robot_pos_w = env.scene["robot"].data.root_pos_w
        if use_ee_direction:
            reference_xy = self._tool_center_pos_w(env)[:, :2]
            fallback_dir = center_pos_w[:, :2] - robot_pos_w[:, :2]
            fallback_dir = fallback_dir / torch.linalg.vector_norm(fallback_dir, dim=-1, keepdim=True).clamp_min(1e-5)
        else:
            reference_xy = robot_pos_w[:, :2]
            fallback_dir = None

        approach_vec = center_pos_w[:, :2] - reference_xy
        approach_norm = torch.linalg.vector_norm(approach_vec, dim=-1, keepdim=True)
        approach_dir = approach_vec / approach_norm.clamp_min(1e-5)
        if fallback_dir is not None:
            approach_dir = torch.where(approach_norm < 1e-5, fallback_dir, approach_dir)

        target_pos_w = center_pos_w.clone()
        target_pos_w[:, :2] -= approach_dir * backoff
        target_pos_w[:, 0] += xy_nudge[0]
        target_pos_w[:, 1] += xy_nudge[1]
        target_pos_w[:, 2] += z_offset
        return target_pos_w

    def _blend_targets(self, start_w: torch.Tensor, end_w: torch.Tensor, alpha: float) -> torch.Tensor:
        alpha = min(max(alpha, 0.0), 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return start_w + (end_w - start_w) * alpha

    def _grasp_backoff(self) -> float:
        return max(0.0, _GRASP_BACKOFF - self._grasp_retry_count * _RETRY_BACKOFF_DELTA)

    def _grasp_z_offset(self) -> float:
        return _GRASP_Z_OFFSET + self._grasp_retry_count * _RETRY_Z_DELTA

    def _grasp_xy_nudge(self) -> tuple[float, float]:
        x_nudge = _GRASP_X_NUDGE + self._grasp_retry_count * _RETRY_X_NUDGE_DELTA
        y_nudge = _GRASP_Y_NUDGE + self._grasp_retry_count * _RETRY_Y_NUDGE_DELTA
        return (x_nudge, y_nudge)

    def _insert_backoff(self) -> float:
        return max(-0.035, self._grasp_backoff() - _INSERT_BACKOFF_DELTA)

    def _insert_z_offset(self) -> float:
        return self._grasp_z_offset() + _INSERT_Z_DELTA

    def _insert_xy_nudge(self) -> tuple[float, float]:
        grasp_x, grasp_y = self._grasp_xy_nudge()
        return (grasp_x + _INSERT_X_NUDGE_DELTA, grasp_y + _INSERT_Y_NUDGE_DELTA)

    def _forced_grasp_target(self, ball_pos_w: torch.Tensor, z_offset: float) -> torch.Tensor:
        target_pos_w = ball_pos_w.clone()
        target_pos_w[:, 0] += _FORCED_GRASP_X_OFFSET
        target_pos_w[:, 1] += _FORCED_GRASP_Y_OFFSET
        target_pos_w[:, 2] += z_offset
        return target_pos_w

    def _ready_to_close_gripper(self, env, ball_pos_w: torch.Tensor) -> torch.Tensor:
        tool_pos_w = self._tool_center_pos_w(env)
        delta = ball_pos_w - tool_pos_w
        xy_dist = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        z_error = torch.abs(delta[:, 2] + _GRASP_TOOL_CENTER_Z_ABOVE_CUBE)
        return torch.logical_and(xy_dist <= _CLOSE_READY_XY_DISTANCE, z_error <= _CLOSE_READY_Z_DISTANCE)

    def _feedback_close_command(self, env, ball_pos_w: torch.Tensor, num_envs: int, device) -> torch.Tensor:
        ready = self._ready_to_close_gripper(env, ball_pos_w)
        if bool(ready.all().item()) and self._close_start_step is None:
            self._close_start_step = self._step_count

        if self._close_start_step is None:
            return self._gripper_command(num_envs, device, _GRIPPER_OPEN_POS)

        alpha = (self._step_count - self._close_start_step) / max(
            _GRASP_CLOSE_END_STEP - _GRASP_CLOSE_START_STEP,
            1,
        )
        alpha = min(max(alpha, 0.0), 1.0)
        gripper_pos = _GRIPPER_OPEN_POS + (_GRIPPER_CLOSE_POS - _GRIPPER_OPEN_POS) * alpha
        return self._gripper_command(num_envs, device, gripper_pos)

    def _phase_approach_hover(self, env, ball_pos_w, num_envs, device):
        hover_target = self._side_jaw_target(env, ball_pos_w, _APPROACH_BACKOFF, _APPROACH_Z_OFFSET)
        alpha = self._step_count / _APPROACH_STEPS
        if self._initial_ee_pos is not None:
            target_pos_w = (1.0 - alpha) * self._initial_ee_pos + alpha * hover_target
        else:
            target_pos_w = hover_target
        return target_pos_w, self._gripper_prepare_command(env, ball_pos_w, num_envs, device)

    def _phase_move_to_pregrasp(self, env, ball_pos_w, num_envs, device):
        hover_target = self._side_jaw_target(env, ball_pos_w, _APPROACH_BACKOFF, _APPROACH_Z_OFFSET)
        pregrasp_target = self._side_jaw_target(env, ball_pos_w, _PREGRASP_BACKOFF, _PREGRASP_Z_OFFSET)
        alpha = (self._step_count - _APPROACH_STEPS) / (320 - _APPROACH_STEPS)
        target_pos_w = self._blend_targets(hover_target, pregrasp_target, alpha)
        return target_pos_w, self._gripper_prepare_command(env, ball_pos_w, num_envs, device)

    def _phase_pregrasp(self, env, ball_pos_w, num_envs, device):
        target_pos_w = self._side_jaw_target(env, ball_pos_w, _PREGRASP_BACKOFF, _PREGRASP_Z_OFFSET)
        return target_pos_w, self._gripper_prepare_command(env, ball_pos_w, num_envs, device)

    def _phase_move_into_ball(self, env, ball_pos_w, num_envs, device):
        pregrasp_target = self._side_jaw_target(env, ball_pos_w, _PREGRASP_BACKOFF, _PREGRASP_Z_OFFSET)
        grasp_target = self._side_jaw_target(
            env,
            ball_pos_w,
            self._grasp_backoff(),
            _GRASP_XY_Z_OFFSET,
            xy_nudge=self._grasp_xy_nudge(),
            use_ee_direction=True,
        )
        alpha = (self._step_count - 520) / (_GRASP_INSERT_START_STEP - 520)
        target_pos_w = self._blend_targets(pregrasp_target, grasp_target, alpha)
        return target_pos_w, self._gripper_prepare_command(env, ball_pos_w, num_envs, device)

    def _xy_insert_target(self, env, ball_pos_w):
        return self._forced_grasp_target(ball_pos_w, _GRASP_XY_Z_OFFSET)

    def _final_grasp_target(self, env, ball_pos_w):
        return self._forced_grasp_target(ball_pos_w, self._insert_z_offset())

    def _phase_xy_insert_into_ball(self, env, ball_pos_w, num_envs, device):
        grasp_target = self._side_jaw_target(
            env,
            ball_pos_w,
            self._grasp_backoff(),
            _GRASP_XY_Z_OFFSET,
            xy_nudge=self._grasp_xy_nudge(),
            use_ee_direction=True,
        )
        insert_target = self._xy_insert_target(env, ball_pos_w)
        alpha = (self._step_count - _GRASP_INSERT_START_STEP) / (
            _GRASP_Z_DESCEND_START_STEP - _GRASP_INSERT_START_STEP
        )
        target_pos_w = self._blend_targets(grasp_target, insert_target, alpha)
        return target_pos_w, self._gripper_command(num_envs, device, _GRIPPER_OPEN_POS)

    def _phase_lower_into_ball(self, env, ball_pos_w, num_envs, device):
        xy_target = self._xy_insert_target(env, ball_pos_w)
        final_target = self._final_grasp_target(env, ball_pos_w)
        alpha = (self._step_count - _GRASP_Z_DESCEND_START_STEP) / (
            _GRASP_CLOSE_START_STEP - _GRASP_Z_DESCEND_START_STEP
        )
        target_pos_w = self._blend_targets(xy_target, final_target, alpha)
        return target_pos_w, self._gripper_command(num_envs, device, _GRIPPER_OPEN_POS)

    def _phase_grasp(self, env, ball_pos_w, num_envs, device):
        target_pos_w = self._final_grasp_target(env, ball_pos_w)
        return target_pos_w, self._feedback_close_command(env, ball_pos_w, num_envs, device)

    def _phase_grasp_hold(self, env, ball_pos_w, num_envs, device):
        target_pos_w = self._final_grasp_target(env, ball_pos_w)
        return target_pos_w, self._feedback_close_command(env, ball_pos_w, num_envs, device)

    def _phase_lift_ball(self, env, ball_pos_w, num_envs, device):
        target_pos_w = self._side_jaw_target(
            env,
            ball_pos_w,
            self._insert_backoff(),
            _CARRY_Z_OFFSET,
            xy_nudge=self._insert_xy_nudge(),
            use_ee_direction=True,
        )
        return target_pos_w, self._gripper_command(num_envs, device, _GRIPPER_CLOSE_POS)

    def _phase_move_above_target(self, env, target_pos_w, num_envs, device):
        target_pos_w = self._side_jaw_target(
            env, target_pos_w, _PREGRASP_BACKOFF, _CARRY_Z_OFFSET, xy_nudge=_PLACE_TARGET_XY_NUDGE
        )
        return target_pos_w, self._gripper_command(num_envs, device, _GRIPPER_CLOSE_POS)

    def _phase_lower_to_target(self, env, target_pos_w, num_envs, device):
        target_pos_w = self._side_jaw_target(
            env, target_pos_w, _GRASP_BACKOFF, _PLACE_Z_OFFSET, xy_nudge=_PLACE_TARGET_XY_NUDGE
        )
        return target_pos_w, self._gripper_command(num_envs, device, _GRIPPER_CLOSE_POS)

    def _phase_release(self, env, target_pos_w, num_envs, device):
        target_pos_w = self._side_jaw_target(
            env, target_pos_w, _GRASP_BACKOFF, _PLACE_Z_OFFSET, xy_nudge=_PLACE_TARGET_XY_NUDGE
        )
        return target_pos_w, self._gripper_command(num_envs, device, _GRIPPER_OPEN_POS)

    def _phase_lift_gripper(self, env, target_pos_w, num_envs, device):
        target_pos_w = self._side_jaw_target(
            env, target_pos_w, _PREGRASP_BACKOFF, _POST_RELEASE_Z_OFFSET, xy_nudge=_PLACE_TARGET_XY_NUDGE
        )
        return target_pos_w, self._gripper_command(num_envs, device, _GRIPPER_OPEN_POS)

    def _phase_return_home(self, env, num_envs, device):
        rest_arm = self._rest_joint_pos
        if rest_arm is None:
            rest_arm = torch.zeros((num_envs, 6), device=device)
        joint_names = list(env.scene["robot"].data.joint_names)
        joint_ids = [joint_names.index(name) for name in _ARM_JOINT_NAMES]
        arm_target = rest_arm[:, joint_ids]
        self._last_joint_target = arm_target.detach()
        gripper_cmd = self._gripper_command(num_envs, device, _GRIPPER_OPEN_POS)
        return torch.cat([arm_target, gripper_cmd], dim=-1)

    def _gripper_command(self, num_envs: int, device, position: float) -> torch.Tensor:
        return torch.full((num_envs, 1), position, device=device)

    def _gripper_prepare_command(self, env, ball_pos_w, num_envs: int, device) -> torch.Tensor:
        alpha = min(self._step_count / _GRIPPER_OPEN_RAMP_STEPS, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        gripper_pos = _GRIPPER_STANDBY_POS + (_GRIPPER_OPEN_POS - _GRIPPER_STANDBY_POS) * alpha
        return self._gripper_command(num_envs, device, gripper_pos)

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count
