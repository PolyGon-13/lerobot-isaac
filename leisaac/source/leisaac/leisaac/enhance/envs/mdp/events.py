from typing import Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from pxr import Gf


def randomize_camera_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation uniformly within the given ranges.

    * It samples the camera position and rotation from the given ranges and adds them to the
      default camera position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    asset: Camera = env.scene[asset_cfg.name]

    ori_pos_w = asset.data.pos_w[env_ids]
    if convention == "ros":
        ori_quat_w = asset.data.quat_w_ros[env_ids]
    elif convention == "opengl":
        ori_quat_w = asset.data.quat_w_opengl[env_ids]
    elif convention == "world":
        ori_quat_w = asset.data.quat_w_world[env_ids]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # camera usually spawn with robot, so no need to add env_origins
    positions = ori_pos_w[:, 0:3] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)


def randomize_particle_object_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
):
    """Reset the particle object to a random position and rotation uniformly within the given ranges.

    * It samples the particle object position and rotation from the given ranges and adds them to the
      default particle object position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    particle_object = env.scene.particle_objects[asset_cfg.name]
    ori_world_pos, ori_world_quat = particle_object.get_world_poses()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    positions = ori_world_pos + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_world_quat, orientations_delta)

    particle_object.set_world_poses(positions, orientations)


def randomize_visual_material_color_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    color_range: dict[str, tuple[float, float]],
    shader_path: str = "geometry/material/Shader",
):
    """Randomize a simple PreviewSurface material color on reset.

    This is intentionally small and USD-level: it targets the PreviewSurface shader created by
    IsaacLab shape spawners (``{asset}/geometry/material/Shader``) and changes only
    ``inputs:diffuseColor``. It is meant for visual-domain randomization of simple scene props.
    """
    env_ids = _normalize_env_ids(env, env_ids)
    colors = _sample_rgb_uniform(env, env_ids, color_range).detach().cpu().tolist()
    paths = _matching_paths_by_env(env, asset_cfg.name, shader_path, env_ids)
    stage = env.scene.stage

    for env_id, color in zip(env_ids.detach().cpu().tolist(), colors):
        for prim_path in paths.get(env_id, []):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            attr = prim.GetAttribute("inputs:diffuseColor")
            if attr.IsValid():
                attr.Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))


def randomize_light_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    intensity_range: tuple[float, float] = (700.0, 1200.0),
    color_range: dict[str, tuple[float, float]] | None = None,
):
    """Randomize a light's intensity and, optionally, color on reset."""
    env_ids = _normalize_env_ids(env, env_ids)
    low_high = torch.tensor(intensity_range, device=env.device, dtype=torch.float32)
    intensities = math_utils.sample_uniform(
        low_high[0], low_high[1], (len(env_ids),), device=env.device
    ).detach().cpu().tolist()
    colors = None
    if color_range is not None:
        colors = _sample_rgb_uniform(env, env_ids, color_range).detach().cpu().tolist()

    paths = _matching_paths_by_env(env, asset_cfg.name, "", env_ids)
    stage = env.scene.stage

    for index, env_id in enumerate(env_ids.detach().cpu().tolist()):
        for prim_path in paths.get(env_id, []):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            intensity_attr = prim.GetAttribute("inputs:intensity")
            if intensity_attr.IsValid():
                intensity_attr.Set(float(intensities[index]))
            if colors is not None:
                color_attr = prim.GetAttribute("inputs:color")
                if color_attr.IsValid():
                    color = colors[index]
                    color_attr.Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))


def _normalize_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _sample_rgb_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    color_range: dict[str, tuple[float, float]],
) -> torch.Tensor:
    lows = torch.tensor([color_range[key][0] for key in ("r", "g", "b")], device=env.device)
    highs = torch.tensor([color_range[key][1] for key in ("r", "g", "b")], device=env.device)
    return math_utils.sample_uniform(lows, highs, (len(env_ids), 3), device=env.device).clamp(0.0, 1.0)


def _matching_paths_by_env(
    env: ManagerBasedRLEnv,
    asset_name: str,
    relative_path: str,
    env_ids: torch.Tensor,
) -> dict[int, list[str]]:
    scene_asset_cfg = getattr(env.scene.cfg, asset_name)
    relative_path = relative_path.strip("/")
    path_expr = scene_asset_cfg.prim_path
    if relative_path:
        path_expr = f"{path_expr}/{relative_path}"

    all_paths = sim_utils.find_matching_prim_paths(path_expr)
    env_id_list = env_ids.detach().cpu().tolist()
    paths_by_env = {int(env_id): [] for env_id in env_id_list}

    for prim_path in all_paths:
        matched_specific_env = False
        for env_id in env_id_list:
            if f"/env_{int(env_id)}/" in prim_path:
                paths_by_env[int(env_id)].append(prim_path)
                matched_specific_env = True
                break
        if not matched_specific_env and len(env_id_list) == 1:
            paths_by_env[int(env_id_list[0])].append(prim_path)

    return paths_by_env


def disable_rigid_body_gravity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """Disable gravity for specific bodies in an articulation.

    This function disables gravity for bodies specified in the asset_cfg.body_names.
    It uses modify_rigid_body_properties to set disable_gravity=True for the specified bodies.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply the change to.
        asset_cfg: Configuration specifying the asset and body names to disable gravity for.
                   Use body_names to specify which bodies to disable gravity (e.g., ".*arm.*" or ["shoulder", "elbow"]).
    """
    # Get the asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve body indices from body_names (already resolved by SceneEntityCfg)
    if asset_cfg.body_ids == slice(None):
        body_ids = list(range(asset.num_bodies))
    else:
        body_ids = asset_cfg.body_ids if isinstance(asset_cfg.body_ids, list) else [asset_cfg.body_ids]

    # Get link paths from the first environment (they follow the same pattern for all environments)
    link_paths = asset.root_physx_view.link_paths[0]

    # Disable gravity for each specified body
    for body_id in body_ids:
        if body_id >= len(link_paths):
            continue

        # Get the link path from first environment
        first_env_link_path = link_paths[body_id]

        # Convert to regex expression by replacing env_0 with env_.*
        link_path_expr = first_env_link_path.replace("/env_0/", "/env_.*/")

        # Resolve all matching prim paths and apply
        prim_paths = sim_utils.find_matching_prim_paths(link_path_expr)
        for prim_path in prim_paths:
            sim_utils.modify_rigid_body_properties(
                prim_path,
                sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            )
