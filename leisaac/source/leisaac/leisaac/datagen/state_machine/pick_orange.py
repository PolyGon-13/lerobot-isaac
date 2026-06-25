"""State machine for the pick-orange task."""

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul
from leisaac.tasks.pick_orange.mdp import orange_grasped, task_done

from .base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_GRIPPER_OFFSET = 0.1  # vertical clearance for the gripper tip
_APPROACH_STEPS: int = 120  # steps to smoothly interpolate from init EE pos to hover (first orange only)

# --- grasp retry (so the recorded demos contain "miss -> re-approach -> grasp" recovery) ---
_MAX_GRASP_ATTEMPTS: int = 3  # first try + up to 2 retries before giving up on the episode
_GRASP_RETRY_STEP: int = 180  # rewind target: re-enter at hover-above-orange to re-approach
# Search-style retry: the first miss is often a structural IK dy error (SO-101 5-DOF can't hit
# the orange's y exactly), and the orange doesn't move, so re-grasping identically fails again.
# Instead, shift each retry sideways (y) by this much — retry#1 one way, retry#2 the other — to
# probe around the miss. This "feel left/right and re-grasp" also bakes a realistic recovery
# behavior into the demo.
_GRASP_RETRY_XY_SWEEP: float = 0.025  # m
# Closed-loop approach correction (sim data-collection only): we know the exact orange and EE-tip
# positions, so during the hover->grasp approach we feed the xy error back into the IK target to
# cancel the 5-DOF IK's structural dy/dx miss. The recorded demo (images/state/action) is unaffected.
_GRASP_CL_GAIN: float = 0.6  # feedback gain on (orange_xy - ee_tip_xy)
_GRASP_CL_BEGIN: int = 180   # start correcting at the hover phase
_GRASP_CL_END: int = 420     # stop at the end of the grasp (close) phase
# The FIRST grasp attempt fails in one of two ways (chosen at random) so the demos cover both
# recovery types; retries always aim true at the orange's *current* position:
#   - "whiff": close too high, orange stays put -> easy same-spot retry
#   - "nudge": small xy miss that bumps the orange so it rolls -> harder retry that must track
#     the orange to its new resting position (the realistic case).
_GRASP_MISS_HEIGHT: float = 0.03  # whiff: close this much above the orange (no contact)
_GRASP_MISS_XY: float = 0.02  # nudge: xy miss (m) that taps the orange and rolls it a little
# Drop the gripper tip past the hover clearance before closing. With full-pose IK 0.06 was needed
# (and made the wrist body hit the table); with position-only IK the wrist orients freely, so a
# shallower 0.03 should seat the fingers without ramming the table. Re-tune from GUI if needed.
_GRASP_EXTRA_DEPTH: float = 0.03

# Pitching the grasp orientation off vertical was tried but it threw the grasp position off by
# ~19cm (orientation and the xy/z offsets are coupled through the IK), so it's kept at 0 = the
# original top-down pose, which puts the tip closest to the orange.
_GRASP_TILT: float = 0.0

_REST_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 90.0,
    "wrist_flex": 50.0,
    "wrist_roll": 0.0,
    "gripper": -10.0,
}


def _apply_triangle_offset(pos_tensor: torch.Tensor, orange_now: int, radius: float = 0.10) -> torch.Tensor:
    """Apply an equilateral-triangle offset on the x-y plane."""
    idx = (orange_now - 1) % 3
    angle = idx * (2 * math.pi / 3)
    pos_tensor[:, 0] += radius * math.cos(angle)
    pos_tensor[:, 1] += radius * math.sin(angle)
    return pos_tensor


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class PickOrangeStateMachine(StateMachineBase):
    """State machine for the pick-orange manipulation task.

    The robot cycles through *num_oranges* oranges.  For each orange it
    executes a fixed sequence of steps that moves the gripper above the
    orange, grasps it, lifts it, transports it to the plate and places it.

    Args:
        num_oranges: Total number of oranges to pick and place. Defaults to 3.
    """

    MAX_STEPS_PER_ORANGE: int = 980
    # Grasp must be held at lift (480) AND mid-transport to the plate (540): if the orange slips
    # out while being carried, re-grasp instead of finishing empty-handed.
    _GRASP_CHECK_STEPS: tuple[int, ...] = (480, 540)

    def __init__(self, num_oranges: int = 3) -> None:
        self._num_oranges = num_oranges
        self._step_count: int = 0
        self._orange_now: int = 1
        self._episode_done: bool = False
        self._initial_ee_pos: torch.Tensor | None = None
        self._rest_ee_pos_world: torch.Tensor | None = None
        self._rest_joint_pos: torch.Tensor | None = None
        self._home_start_pos: torch.Tensor | None = None
        self._grasp_attempt: int = 0
        self._grasp_z_miss: float = 0.0
        self._grasp_xy_offset: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        """FK calibration: drive arm to rest pose and record the EE world position.

        Teleports joints to the SO-101 rest pose, steps the simulation once
        to propagate kinematics, then reads ``body_pos_w`` to get the EE world
        position that corresponds to the rest pose.  This position is used as
        the home target during the return-home phase so that ``task_done()``
        can verify the arm is within the expected joint-angle tolerances.
        """
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

    def check_success(self, env) -> bool:
        """Return True if all picked oranges are on the plate and the arm is at rest."""
        robot = env.scene["robot"]
        if self._rest_joint_pos is not None:
            robot.write_joint_state_to_sim(
                position=self._rest_joint_pos,
                velocity=torch.zeros_like(self._rest_joint_pos),
            )
            env.scene.update(dt=env.physics_dt)

        oranges_cfg = [SceneEntityCfg(f"Orange00{i+1}") for i in range(self._num_oranges)]
        success_tensor = task_done(
            env,
            oranges_cfg=oranges_cfg,
            plate_cfg=SceneEntityCfg("Plate"),
        )
        return bool(success_tensor.all().item())

    def pre_step(self, env) -> None:
        """Blend joint state toward rest pose during the final orange's home phase.

        Only active when ``orange_now == num_oranges`` and ``step_count >= 680``.
        Writes blended joint positions directly to the sim so the arm smoothly
        returns home while IK still receives the rest-pose EE target.
        """
        if self._orange_now == self._num_oranges and self._step_count >= 680 and self._rest_joint_pos is not None:
            robot = env.scene["robot"]
            if self._step_count == 680:
                self._home_start_pos = robot.data.joint_pos.clone()
            if self._home_start_pos is not None:
                alpha = min((self._step_count - 680) / 299.0, 1.0)
                blended = self._home_start_pos + (self._rest_joint_pos - self._home_start_pos) * alpha
                robot.write_joint_state_to_sim(position=blended, velocity=torch.zeros_like(blended))

    def get_action(self, env) -> torch.Tensor:
        """Compute the action tensor for the current step (8D IK pose target)."""
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs
        step = self._step_count

        orange_pos_w = env.scene[f"Orange00{self._orange_now}"].data.root_pos_w.clone()
        plate_pos_w = env.scene["Plate"].data.root_pos_w.clone()
        robot_base_pos_w = robot.data.root_pos_w.clone()
        robot_base_quat_w = robot.data.root_quat_w.clone()

        # position-only IK: action carries the target EE position only; the wrist orients freely.

        if self._orange_now == 1 and step == 0:
            self._initial_ee_pos = robot.data.body_pos_w[:, -1, :].clone()

        # On the first step of an orange, arm a miss for the FIRST grasp attempt — randomly a
        # high "whiff" (orange stays) or a small xy "nudge" (orange rolls), so the demos contain
        # both same-spot and track-to-new-position recoveries. Retries (grasp-check block below)
        # clear both so they aim true at the orange's current position.
        if step == 0:
            self._grasp_attempt = 0
            if torch.rand(1).item() < 0.5:
                self._grasp_z_miss = _GRASP_MISS_HEIGHT
                self._grasp_xy_offset = None
            else:
                self._grasp_z_miss = 0.0
                self._grasp_xy_offset = (torch.rand(num_envs, 2, device=device) - 0.5) * (2.0 * _GRASP_MISS_XY)

        # After the lift phase the current orange must be grasped (single-env only). On a miss,
        # rewind to the hover phase and re-approach (up to _MAX_GRASP_ATTEMPTS) instead of
        # aborting — this bakes a "miss -> re-approach -> grasp" recovery into the successful
        # episode so the policy can learn to retry. On success we just continue (step stays at
        # _GRASP_CHECK_STEP), so the plate/home timing downstream is unchanged.
        # DBG: distance at the moment the gripper finishes closing (grasp phase ends ~420), to
        # tell whether the gripper reaches the orange at all vs. drops it during the lift.
        if env.num_envs == 1 and step == 415:
            ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
            _df = (orange_pos_w - ee_pos)[0]
            _d = float(torch.linalg.vector_norm(_df))
            print(f"[SM] grasp-close@{step} attempt={self._grasp_attempt} "
                  f"dx={float(_df[0]):+.3f} dy={float(_df[1]):+.3f} dz={float(_df[2]):+.3f} dist={_d:.3f}")

        if env.num_envs == 1 and step in self._GRASP_CHECK_STEPS:
            # DBG: why does the grasp (and the retry) succeed/fail? Log EE<->orange distance and
            # the gripper joint at the moment of the check.
            ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
            _df = (orange_pos_w - ee_pos)[0]
            _dist = float(torch.linalg.vector_norm(_df))
            _gj = float(robot.data.joint_pos[0, -1])
            _grasped = self._is_current_orange_grasped(env)
            print(f"[SM] check@{step} attempt={self._grasp_attempt} "
                  f"dx={float(_df[0]):+.3f} dy={float(_df[1]):+.3f} dz={float(_df[2]):+.3f} "
                  f"dist={_dist:.3f} gripper={_gj:.3f} grasped={_grasped}")
            if not _grasped:
                if self._grasp_attempt + 1 < _MAX_GRASP_ATTEMPTS:
                    self._grasp_attempt += 1
                    self._grasp_z_miss = 0.0  # retry: full depth
                    # Search: shift the grasp sideways (y), alternating sides each retry, to probe
                    # around the structural dy miss the previous attempt made.
                    _sweep = _GRASP_RETRY_XY_SWEEP if (self._grasp_attempt % 2 == 1) else -_GRASP_RETRY_XY_SWEEP
                    self._grasp_xy_offset = torch.tensor([[0.0, _sweep]], device=device)
                    self._step_count = _GRASP_RETRY_STEP
                    step = self._step_count
                    print(f"[SM]   -> miss, retry #{self._grasp_attempt} sweep_y={_sweep:+.3f} "
                          f"(orange now at x={float(orange_pos_w[0, 0]):.3f} y={float(orange_pos_w[0, 1]):.3f})")
                else:
                    self._episode_done = True
                    print("[SM]   -> miss, give up")

        if self._orange_now == 1 and step < _APPROACH_STEPS:
            target_pos_w, gripper_cmd = self._phase_approach_hover(orange_pos_w, num_envs, device)
        elif step < 180:
            target_pos_w, gripper_cmd = self._phase_move_above_orange(orange_pos_w, num_envs, device)
        elif step < 300:
            target_pos_w, gripper_cmd = self._phase_hover_above_orange(orange_pos_w, num_envs, device)
        elif step < 360:
            target_pos_w, gripper_cmd = self._phase_lower_to_orange(orange_pos_w, num_envs, device)
        elif step < 420:
            target_pos_w, gripper_cmd = self._phase_grasp(orange_pos_w, num_envs, device)
        elif step < 500:
            target_pos_w, gripper_cmd = self._phase_lift_orange(orange_pos_w, num_envs, device)
        elif step < 550:
            target_pos_w, gripper_cmd = self._phase_move_above_plate(plate_pos_w, num_envs, device)
        elif step < 600:
            target_pos_w, gripper_cmd = self._phase_lower_to_plate(plate_pos_w, num_envs, device)
        elif step < 640:
            target_pos_w, gripper_cmd = self._phase_release(plate_pos_w, num_envs, device)
        elif step < 680:
            target_pos_w, gripper_cmd = self._phase_lift_gripper(plate_pos_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_return_home(num_envs, device)

        # Closed-loop xy correction during the approach: feed the measured EE-tip vs orange error
        # back into the target so the IK's structural dy/dx miss converges out (sim-only aid).
        if _GRASP_CL_BEGIN <= step < _GRASP_CL_END:
            ee_tip = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
            target_pos_w[:, :2] += _GRASP_CL_GAIN * (orange_pos_w[:, :2] - ee_tip[:, :2])

        diff_w = target_pos_w - robot_base_pos_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)
        # position-only IK: 3D EE position target + binary gripper (no orientation).
        return torch.cat([target_pos_local, gripper_cmd], dim=-1)

    def _is_current_orange_grasped(self, env) -> bool:
        """Return True if the orange currently being manipulated is held by the gripper."""
        grasped = orange_grasped(env, object_cfg=SceneEntityCfg(f"Orange00{self._orange_now}"))
        return bool(grasped.all().item())

    def advance(self) -> None:
        """Advance step counter, handle orange transitions, and fast-forward home phase.

        For non-final oranges, the return-home phase (steps 680–979) is skipped
        without simulation — the arm goes straight to the next orange.
        """
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS_PER_ORANGE:
            if self._orange_now >= self._num_oranges:
                self._episode_done = True
            else:
                self._orange_now += 1
                self._step_count = 0
        elif self._orange_now < self._num_oranges and self._step_count >= 680:
            # Fast-forward: skip the home phase for intermediate oranges.
            prev_orange = self._orange_now
            while self._orange_now == prev_orange and not self._episode_done:
                self._step_count += 1
                if self._step_count >= self.MAX_STEPS_PER_ORANGE:
                    self._orange_now += 1
                    self._step_count = 0

    def reset(self) -> None:
        """Reset the state machine to its initial state for a new episode."""
        self._step_count = 0
        self._orange_now = 1
        self._episode_done = False
        self._initial_ee_pos = None
        self._home_start_pos = None
        self._grasp_attempt = 0
        self._grasp_z_miss = 0.0
        self._grasp_xy_offset = None

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def _phase_approach_hover(self, orange_pos_w, num_envs, device):
        hover_target = orange_pos_w.clone()
        hover_target[:, 0] -= 0.03
        hover_target[:, 1] -= 0.01
        hover_target[:, 2] += 0.1 + _GRIPPER_OFFSET
        alpha = self._step_count / _APPROACH_STEPS
        if self._initial_ee_pos is not None:
            target_pos_w = (1.0 - alpha) * self._initial_ee_pos + alpha * hover_target
        else:
            target_pos_w = hover_target
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_move_above_orange(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.15 + _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_hover_above_orange(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.1 + _GRIPPER_OFFSET
        self._apply_grasp_xy_offset(target_pos_w)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_lower_to_orange(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += _GRIPPER_OFFSET - _GRASP_EXTRA_DEPTH + self._grasp_z_miss
        self._apply_grasp_xy_offset(target_pos_w)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_grasp(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += _GRIPPER_OFFSET - _GRASP_EXTRA_DEPTH + self._grasp_z_miss
        self._apply_grasp_xy_offset(target_pos_w)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _apply_grasp_xy_offset(self, target_pos_w: torch.Tensor) -> None:
        """Add the current grasp xy offset (nonzero only on the first, intentionally-imperfect
        attempt) to the target so some first attempts miss and trigger a retry."""
        if self._grasp_xy_offset is not None:
            target_pos_w[:, :2] += self._grasp_xy_offset

    def _phase_lift_orange(self, orange_pos_w, num_envs, device):
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.25
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_move_above_plate(self, plate_pos_w, num_envs, device):
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.25
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_lower_to_plate(self, plate_pos_w, num_envs, device):
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
        _apply_triangle_offset(target_pos_w, self._orange_now)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_release(self, plate_pos_w, num_envs, device):
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
        _apply_triangle_offset(target_pos_w, self._orange_now)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_lift_gripper(self, plate_pos_w, num_envs, device):
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.2
        _apply_triangle_offset(target_pos_w, self._orange_now)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_return_home(self, num_envs, device):
        if self._rest_ee_pos_world is not None:
            target_pos_w = self._rest_ee_pos_world.clone()
        elif self._initial_ee_pos is not None:
            target_pos_w = self._initial_ee_pos.clone()
        else:
            target_pos_w = torch.zeros(num_envs, 3, device=device)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def orange_now(self) -> int:
        return self._orange_now

    @property
    def step_count(self) -> int:
        return self._step_count
