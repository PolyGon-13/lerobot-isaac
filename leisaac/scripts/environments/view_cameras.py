"""Render the wrist + front(top) camera views of a LeIsaac task and save them as PNGs.

Boots the env with cameras enabled, renders a few frames so the camera buffers fill,
then dumps each camera RGB plus a side-by-side composite to --out.

Run (leisaac conda env):
    conda activate leisaac && source /opt/isaac-sim-5.1.0/setup_conda_env.sh
    python /home/polygon/lerobot_isaac/leisaac/scripts/environments/view_cameras.py \
        --task LeIsaac-SO101-PickCubeAndPlace-v0 --headless --enable_cameras
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render and save a task's camera views.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-PickCubeAndPlace-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--teleop_device", type=str, default="lerobot_ver2_state_machine", help="Action config to init.")
parser.add_argument("--cameras", type=str, default="front,wrist", help="Comma-separated scene camera names.")
parser.add_argument("--settle", type=int, default=12, help="Frames to render before grabbing camera images.")
parser.add_argument("--step_physics", action="store_true", help="Advance physics while settling. By default, only render frames to match view_env.")
parser.add_argument("--out", type=str, default="/home/polygon/lerobot_isaac/lerobot/lerobot_ver2/cam_views",
                    help="Output directory for PNGs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras require the rendering pipeline.
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import leisaac.tasks  # noqa: F401, E402 — registers LeIsaac-* envs
from isaaclab.utils.math import quat_apply, quat_inv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _save_png(arr: np.ndarray, path: str) -> None:
    """Save an HxWx3 uint8 array as PNG (PIL if available, else a P6 PPM fallback)."""
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception:
        ppm = os.path.splitext(path)[0] + ".ppm"
        with open(ppm, "wb") as f:
            f.write(f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode())
            f.write(arr.tobytes())
        print(f"  (PIL unavailable; wrote {ppm})")


def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """(H,W,C) tensor -> (H,W,3) uint8 numpy."""
    a = t.detach().cpu().float().numpy()
    if a.dtype != np.uint8 and a.max() <= 1.0 + 1e-3:
        a = a * 255.0
    a = np.clip(a, 0, 255).astype(np.uint8)
    return a[..., :3]


def _hold_action(env, device):
    """Steady action that keeps the IK target at the current end-effector pose."""
    action_dim = env.action_space.shape[-1]
    action = torch.zeros((env.num_envs, action_dim), device=device)
    if action_dim == 6:
        try:
            robot = env.scene["robot"]
            joint_names = list(robot.data.joint_names)
            arm_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
            if all(name in joint_names for name in arm_names):
                joint_ids = [joint_names.index(name) for name in arm_names]
                action[:, :5] = robot.data.joint_pos[:, joint_ids]
                action[:, 5] = 1.0
                return action
        except KeyError:
            pass
    if action_dim >= 4:
        try:
            robot = env.scene["robot"]
            ee_pos_w = robot.data.body_pos_w[:, -1, :]
            diff_w = ee_pos_w - robot.data.root_pos_w
            action[:, :3] = quat_apply(quat_inv(robot.data.root_quat_w), diff_w)
            action[:, 3] = 1.0
        except KeyError:
            pass
    return action


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if hasattr(env_cfg, "use_teleop_device"):
        env_cfg.use_teleop_device(args_cli.teleop_device)
    # Don't auto-terminate while we capture.
    if hasattr(env_cfg, "terminations"):
        for attr in ("time_out", "success"):
            if hasattr(env_cfg.terminations, attr):
                setattr(env_cfg.terminations, attr, None)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = torch.device(args_cli.device)
    env.reset()

    action = _hold_action(env, device) if args_cli.step_physics else None
    for _ in range(max(1, args_cli.settle)):
        if args_cli.step_physics:
            env.step(action)
        else:
            env.sim.render()

    cams = [c.strip() for c in args_cli.cameras.split(",") if c.strip()]
    images = []
    for name in cams:
        sensor = env.scene[name]
        rgb = sensor.data.output["rgb"][0]  # (H,W,C)
        img = _to_uint8_rgb(rgb)
        path = os.path.join(args_cli.out, f"{name}.png")
        _save_png(img, path)
        images.append((name, img))
        print(f"[saved] {name}: {img.shape} -> {path}")

    # side-by-side composite (pad to same height)
    if images:
        h = max(im.shape[0] for _, im in images)
        tiles = []
        for _, im in images:
            if im.shape[0] != h:
                pad = np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)
                im = np.vstack([im, pad])
            tiles.append(im)
            tiles.append(np.full((h, 8, 3), 255, np.uint8))  # white separator
        composite = np.hstack(tiles[:-1])
        cpath = os.path.join(args_cli.out, "composite.png")
        _save_png(composite, cpath)
        print(f"[saved] composite -> {cpath}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
