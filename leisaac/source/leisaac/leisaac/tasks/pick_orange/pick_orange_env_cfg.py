import math

import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.assets.scenes.kitchen import (
    KITCHEN_ORANGE_USD_PATH_BY_COUNT,
    KITCHEN_WITH_ORANGE_CFG,
)
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp


@configclass
class PickOrangeSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the pick orange task."""

    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class ObservationsCfg(SingleArmObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        pick_orange001 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange001")})
        put_orange001_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate,
            params={"object_cfg": SceneEntityCfg("Orange001"), "plate_cfg": SceneEntityCfg("Plate")},
        )
        pick_orange002 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange002")})
        put_orange002_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate,
            params={"object_cfg": SceneEntityCfg("Orange002"), "plate_cfg": SceneEntityCfg("Plate")},
        )
        pick_orange003 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange003")})
        put_orange003_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate,
            params={"object_cfg": SceneEntityCfg("Orange003"), "plate_cfg": SceneEntityCfg("Plate")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):

    success = DoneTerm(
        func=mdp.task_done,
        params={
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )


@configclass
class PickOrangeEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the pick orange environment."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    task_description: str = "Pick three oranges and put them into the plate, then reset the arm to rest state."

    num_oranges: int = 3
    """Number of oranges in the scene (1, 2 or 3). Drives the scene USD variant, tracked
    sub-assets, subtask observations, success termination and per-episode randomization so
    that every part of the task references exactly the oranges that exist in the scene."""

    orange_position_range: float = 0.03
    """Half-width (meters) of the uniform per-episode x/y randomization applied to each orange."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Start every episode from the SO-101 rest pose (matches the state machine's
        # _REST_POSE_DEG and the return-home target) instead of the all-zeros USD default,
        # so the arm starts and ends in the same natural pose. Joint values are in radians.
        self.scene.robot.init_state.joint_pos = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -100.0 * math.pi / 180.0,
            "elbow_flex": 90.0 * math.pi / 180.0,
            "wrist_flex": 50.0 * math.pi / 180.0,
            "wrist_roll": 0.0,
            "gripper": -10.0 * math.pi / 180.0,
        }

        assert self.num_oranges in KITCHEN_ORANGE_USD_PATH_BY_COUNT, (
            f"num_oranges must be one of {sorted(KITCHEN_ORANGE_USD_PATH_BY_COUNT)}, got {self.num_oranges}"
        )
        scene_usd_path = KITCHEN_ORANGE_USD_PATH_BY_COUNT[self.num_oranges]

        # Point the kitchen asset at the scene variant that contains only num_oranges oranges.
        self.scene.scene.spawn.usd_path = scene_usd_path

        orange_names = [f"Orange00{i+1}" for i in range(self.num_oranges)]
        parse_usd_and_create_subassets(scene_usd_path, self, specific_name_list=[*orange_names, "Plate"])

        # Disable subtask observation terms for oranges that no longer exist in the scene.
        for i in range(self.num_oranges + 1, 4):
            setattr(self.observations.subtask_terms, f"pick_orange00{i}", None)
            setattr(self.observations.subtask_terms, f"put_orange00{i}_to_plate", None)

        # Success only checks the oranges that actually exist.
        self.terminations.success.params["oranges_cfg"] = [SceneEntityCfg(name) for name in orange_names]

        rng = self.orange_position_range
        randomize_options = [
            randomize_object_uniform(
                f"Orange00{i+1}", pose_range={"x": (-rng, rng), "y": (-rng, rng), "z": (0.0, 0.0)}
            )
            for i in range(self.num_oranges)
        ]
        randomize_options.append(
            randomize_object_uniform("Plate", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)})
        )
        randomize_options.append(
            randomize_camera_uniform(
                "front",
                pose_range={
                    "x": (-0.025, 0.025),
                    "y": (-0.025, 0.025),
                    "z": (-0.025, 0.025),
                    "roll": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                    "pitch": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                    "yaw": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                },
                convention="ros",
            )
        )
        domain_randomization(
            self,
            random_options=randomize_options,
        )


@configclass
class PickOneOrangeEnvCfg(PickOrangeEnvCfg):
    """Single-orange variant of the pick orange task (scene contains only 1 orange)."""

    task_description: str = "Pick the orange and place it into the plate, then reset the arm to rest state."

    num_oranges: int = 1

    # With a single orange, all positional variety must come from randomization. 8cm was too wide
    # (unreachable spots), 5cm then 4.5cm; back to 3cm while tuning the grasp/retry behavior.
    orange_position_range: float = 0.03


@configclass
class PickTwoOrangeEnvCfg(PickOrangeEnvCfg):
    """Two-orange variant of the pick orange task (scene contains only 2 oranges)."""

    task_description: str = "Pick two oranges and place them into the plate, then reset the arm to rest state."

    num_oranges: int = 2
