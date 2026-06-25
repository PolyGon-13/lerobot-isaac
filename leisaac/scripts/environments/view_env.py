"""View a LeIsaac environment without advancing physics by default."""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View a LeIsaac environment.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="so101_state_machine",
    help="Device/action config to initialize before creating the environment.",
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--hide_robot", action="store_true", help="Hide the SO101 robot in scene previews.")
parser.add_argument("--step_physics", action="store_true", help="Advance the environment with a steady hold action.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import leisaac.tasks  # noqa: F401 — registers LeIsaac-* envs
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.utils.math import quat_apply, quat_inv

import omni.usd
from pxr import UsdGeom


def _make_hold_action(env, device: torch.device) -> torch.Tensor:
    """Create a steady action for viewing IK-based single-arm scenes."""
    action_dim = env.action_space.shape[-1]
    action = torch.zeros((env.num_envs, action_dim), device=device)

    if action_dim == 6:
        try:
            robot = env.scene["robot"]
        except KeyError:
            return action
        joint_names = list(robot.data.joint_names)
        arm_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
        if all(name in joint_names for name in arm_names):
            joint_ids = [joint_names.index(name) for name in arm_names]
            action[:, :5] = robot.data.joint_pos[:, joint_ids]
            action[:, 5] = 1.0
            return action

    if action_dim >= 4:
        try:
            robot = env.scene["robot"]
        except KeyError:
            return action
        else:
            ee_pos_w = robot.data.body_pos_w[:, -1, :]
            diff_w = ee_pos_w - robot.data.root_pos_w
            action[:, :3] = quat_apply(quat_inv(robot.data.root_quat_w), diff_w)
            action[:, 3] = 1.0

    return action


def _disable_recording_and_timeouts(env_cfg) -> None:
    """Viewer mode should not accumulate camera frames or auto-export episodes."""
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None


def _set_robot_visibility(visible: bool) -> None:
    """Show or hide task robot prims while keeping cameras and actions valid."""
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() == "Robot":
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                continue
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()


def _hide_placeholder_robots() -> None:
    """Hide robot prims for scene previews."""
    _set_robot_visibility(False)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if hasattr(env_cfg, "use_teleop_device"):
        env_cfg.use_teleop_device(args_cli.teleop_device)
    _disable_recording_and_timeouts(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = torch.device(args_cli.device)
    env.reset()

    if args_cli.hide_robot:
        _hide_placeholder_robots()

    action = _make_hold_action(env, device) if args_cli.step_physics else None
    while simulation_app.is_running():
        if args_cli.step_physics:
            env.step(action)
        else:
            env.sim.render()
            time.sleep(1.0 / 60.0)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
