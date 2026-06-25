import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.assets.robots.lerobot import LEROBOT_VER2_CFG
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
    randomize_light_uniform,
    randomize_visual_material_color_uniform,
)

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp

TABLE_TOP_Z = 0.0
TABLE_SIZE = (0.90, 0.70, 0.04)
TABLE_CENTER_POS = (0.35, 0.0, TABLE_TOP_Z - TABLE_SIZE[2] / 2.0)

TABLE_COLLISION_SIZE = (TABLE_SIZE[0], 0.56, TABLE_SIZE[2])
TABLE_COLLISION_CENTER_POS = (
    TABLE_CENTER_POS[0],
    TABLE_CENTER_POS[1] + 0.07,
    TABLE_CENTER_POS[2],
)
ROBOT_BASE_INSET_FROM_FRONT_EDGE = 0.01
ROBOT_BASE_Z_OFFSET = -0.01

# 3.5cm red cube replaces the red cube. BALL_RADIUS is kept (the state machine imports it as
# the object half-height for grasp/place heights) and now equals the cube half-extent.
CUBE_SIZE = 0.035
BALL_RADIUS = CUBE_SIZE / 2.0
BALL_MASS = 0.01
BALL_START_POS = (0.22, -0.10, TABLE_TOP_Z + BALL_RADIUS)

# 회색 벽/바닥을 흰 테이블(스테이션) 크기에 딱 맞춤: 폭/깊이 = 테이블, 가장자리에 바로 붙임.
WALL_MARGIN = 0.01  # 테이블 가장자리에서 벽까지의 작은 간격
BACKGROUND_PANEL_SIZE = (TABLE_SIZE[0], 0.02, 1.25)
BACKGROUND_PANEL_POS = (
    TABLE_CENTER_POS[0],
    TABLE_CENTER_POS[1] + TABLE_SIZE[1] / 2.0 + WALL_MARGIN,
    TABLE_TOP_Z + BACKGROUND_PANEL_SIZE[2] / 2.0,
)
FLOOR_PANEL_SIZE = (TABLE_SIZE[0], TABLE_SIZE[1], 0.02)
FLOOR_PANEL_POS = (
    TABLE_CENTER_POS[0],
    TABLE_CENTER_POS[1],
    TABLE_CENTER_POS[2] - TABLE_SIZE[2] / 2.0 - 0.011,
)
SIDE_WALL_SIZE = (0.02, TABLE_SIZE[1], BACKGROUND_PANEL_SIZE[2])
LEFT_WALL_POS = (
    TABLE_CENTER_POS[0] - TABLE_SIZE[0] / 2.0 - WALL_MARGIN - SIDE_WALL_SIZE[0] / 2.0,
    TABLE_CENTER_POS[1],
    BACKGROUND_PANEL_POS[2],
)
RIGHT_WALL_POS = (
    TABLE_CENTER_POS[0] + TABLE_SIZE[0] / 2.0 + WALL_MARGIN + SIDE_WALL_SIZE[0] / 2.0,
    TABLE_CENTER_POS[1],
    BACKGROUND_PANEL_POS[2],
)

# 검은 영역(놓을 곳): 이전 target 위치로 복구, 크기는 현재 빨간 큐브 task의 8cm x 8cm 유지.
TARGET_SQUARE_SIZE = 0.08
TARGET_HALF_SIZE = TARGET_SQUARE_SIZE / 2.0
TARGET_CENTER_XY = (0.48, -0.04)

# 로봇 base: 검은 target square는 고정한 채 로봇만 살짝 조정한다.
# X는 빨간 큐브 쪽으로 2cm, Y는 테이블 안쪽으로 2cm 이동한다.
ROBOT_BASE_XY = (
    BALL_START_POS[0] + 0.04,
    TABLE_CENTER_POS[1] - TABLE_SIZE[1] / 2.0 + ROBOT_BASE_INSET_FROM_FRONT_EDGE + 0.02,
)  # (0.26, -0.33)
TARGET_MARKER_THICKNESS = 0.002
TARGET_MARKER_POS = (TARGET_CENTER_XY[0], TARGET_CENTER_XY[1], TABLE_TOP_Z + TARGET_MARKER_THICKNESS / 2.0)
DEFAULT_TABLE_COLOR = (0.95, 0.95, 0.95)
DEFAULT_BALL_COLOR = (0.80, 0.05, 0.05)  # red cube
DEFAULT_TARGET_COLOR = (0.01, 0.01, 0.01)
DEFAULT_BACKGROUND_COLOR = (0.30, 0.31, 0.32)
DEFAULT_FLOOR_COLOR = (0.36, 0.35, 0.33)
DEFAULT_SIDE_WALL_COLOR = (0.32, 0.33, 0.33)

TABLE_COLOR_RANGE = {
    "r": (0.93, 0.97),
    "g": (0.93, 0.97),
    "b": (0.93, 0.97),
}
BALL_COLOR_RANGE = {
    "r": (0.72, 0.86),
    "g": (0.03, 0.09),
    "b": (0.03, 0.09),
}
BACKGROUND_COLOR_RANGE = {
    "r": (0.28, 0.32),
    "g": (0.29, 0.33),
    "b": (0.30, 0.34),
}
FLOOR_COLOR_RANGE = {
    "r": (0.34, 0.38),
    "g": (0.33, 0.37),
    "b": (0.31, 0.35),
}
SIDE_WALL_COLOR_RANGE = {
    "r": (0.30, 0.34),
    "g": (0.31, 0.35),
    "b": (0.31, 0.35),
}
LIGHT_COLOR_RANGE = {
    "r": (0.69, 0.75),
    "g": (0.69, 0.75),
    "b": (0.67, 0.73),
}
LIGHT_INTENSITY_RANGE = (650.0, 1250.0)

# U20CAM-1080P-S1: horizontal FOV is about 106 deg. Isaac's pinhole camera uses
# horizontal_aperture / focal_length, so this focal length gives roughly that FOV.
U20CAM_HORIZONTAL_APERTURE = 38.11
U20CAM_FOCAL_LENGTH = 14.36

TASK_DESCRIPTION = (
    "Pick up the red cube and place it inside the black square, then reset the arm to rest state."
)


@configclass
class PickCubeAndPlaceSceneCfg(SingleArmTaskSceneCfg):
    """Minimal white-table scene with a red cube and black tape square."""

    scene: AssetBaseCfg | None = None

    # Swap the SO101 for the custom lerobot_ver2 arm (keeps its own joint names:
    # base/shoulder/elbow/wrist_pitch/wrist_roll/gripper, links base_link..J5/gripper).
    robot: ArticulationCfg = LEROBOT_VER2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ee_frame: root at base_link; "jaw" is the visible gripper grasp center. This frame is
    # attached to the gripper link because the currently converted USD exposes collision/visual
    # grasp geometry there. Keep arm IK position-only so this movable child frame does not over-
    # constrain the 5-DOF arm.
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper"),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper", name="jaw", offset=OffsetCfg(pos=(0.027, 0.004, -0.022))
            ),
        ],
    )

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEFAULT_TABLE_COLOR, roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_CENTER_POS),
    )

    table_collision: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableCollision",
        spawn=sim_utils.CuboidCfg(
            visible=False,
            size=TABLE_COLLISION_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_COLLISION_CENTER_POS),
    )

    # Named `ball` (state machine / observations reference SceneEntityCfg("ball")) but spawned as
    # a 3.5cm red cube.
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.05,
                angular_damping=0.05,
                max_linear_velocity=2.0,
                max_angular_velocity=720.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # High friction + zero restitution so the cube doesn't slip out of or bounce off
                # the fingers during grasp/lift.
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEFAULT_BALL_COLOR, roughness=0.65),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_START_POS),
    )

    target_square: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetSquare",
        spawn=sim_utils.CuboidCfg(
            size=(TARGET_SQUARE_SIZE, TARGET_SQUARE_SIZE, TARGET_MARKER_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEFAULT_TARGET_COLOR, roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TARGET_MARKER_POS),
    )

    background_panel: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BackgroundPanel",
        spawn=sim_utils.CuboidCfg(
            size=BACKGROUND_PANEL_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEFAULT_BACKGROUND_COLOR, roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKGROUND_PANEL_POS),
    )

    floor_panel: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FloorPanel",
        spawn=sim_utils.CuboidCfg(
            size=FLOOR_PANEL_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEFAULT_FLOOR_COLOR, roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FLOOR_PANEL_POS),
    )

    left_wall: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LeftWall",
        spawn=sim_utils.CuboidCfg(
            size=SIDE_WALL_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEFAULT_SIDE_WALL_COLOR, roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=LEFT_WALL_POS),
    )

    right_wall: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RightWall",
        spawn=sim_utils.CuboidCfg(
            size=SIDE_WALL_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEFAULT_SIDE_WALL_COLOR, roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=RIGHT_WALL_POS),
    )

    # 손목 카메라: gripper 링크에 장착(팔과 함께 움직임 = 진짜 손목캠). gripper의 jaw는 -z로
    # 뻗으므로 approach축 = gripper -z. 광축(ROS +Z)이 gripper -z를 보도록 rot=(0,1,0,0)
    # (x축 180°). 위치는 wrist 부근에서 jaw 앞쪽을 담도록 살짝 뒤/위(+z)로 오프셋한 시작값 —
    # 실제 카메라 마운트 위치는 GUI에서 wrist_camera prim을 골라 gizmo로 옮기며 미세조정.
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        # GUI에서 잡은 wrist_camera 마운트 포즈(gripper 로컬).
        # Translate(0.01197, 0.0335, -0.0996), Orient(-166.762, -17.272, -93.311)° (Omniverse Transform XYZ) → wxyz, convention="opengl".
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.01197, 0.0335, -0.0996),
            rot=(0.18669202394720136, -0.6614910873488091, -0.7260821597610891, 0.01950196988152081),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=U20CAM_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=U20CAM_HORIZONTAL_APERTURE,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=1280,
        height=720,
        update_period=1 / 30.0,
    )

    # 탑뷰 카메라(관측 키는 "front"). **월드(씬)에 고정** — 실제로도 테이블 위 고정 스탠드
    # 카메라라 로봇 회전에 따라 같이 돌면 안 됨(base_link에 붙이면 로봇 rot 바뀔 때 시점이 틀어짐).
    # GUI에서 잡은 탑뷰 기본 포즈(월드).
    # Translate(0.45847, -0.40516, 0.513), Orient(47.051, 21.488, 0.935)° (Omniverse Transform XYZ).
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_camera",
        # Property의 Translate/Orient(USD=OpenGL)를 Omniverse Transform XYZ 기준 quaternion으로 변환.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.45847, -0.40516, 0.513),
            rot=(0.9001723653023298, 0.3935415088652266, 0.1677208464493186, 0.08175886014624717),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=U20CAM_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=U20CAM_HORIZONTAL_APERTURE,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=1280,
        height=720,
        update_period=1 / 30.0,
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.72, 0.72, 0.70), intensity=900.0),
    )


@configclass
class ObservationsCfg(SingleArmObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask signals for data-generation diagnostics."""

        pick_ball = ObsTerm(func=mdp.ball_grasped, params={"object_cfg": SceneEntityCfg("ball")})
        place_ball_in_square = ObsTerm(
            func=mdp.ball_released_in_target_square,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "target_center": TARGET_CENTER_XY,
                "target_half_size": TARGET_HALF_SIZE,
            },
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
            "ball_cfg": SceneEntityCfg("ball"),
            "target_center": TARGET_CENTER_XY,
            "target_half_size": TARGET_HALF_SIZE,
            "require_rest": True,
        },
    )


@configclass
class PickCubeAndPlaceEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the red cube pick-and-place environment."""

    scene: PickCubeAndPlaceSceneCfg = PickCubeAndPlaceSceneCfg(env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    robot_name: str = "lerobot_ver2"
    task_description: str = TASK_DESCRIPTION
    ball_position_range: float = 0.0
    table_color: tuple[float, float, float] = DEFAULT_TABLE_COLOR
    ball_color: tuple[float, float, float] = DEFAULT_BALL_COLOR
    target_color: tuple[float, float, float] = DEFAULT_TARGET_COLOR
    background_color: tuple[float, float, float] = DEFAULT_BACKGROUND_COLOR
    floor_color: tuple[float, float, float] = DEFAULT_FLOOR_COLOR
    side_wall_color: tuple[float, float, float] = DEFAULT_SIDE_WALL_COLOR
    target_center_xy: tuple[float, float] = TARGET_CENTER_XY

    def __post_init__(self) -> None:
        super().__post_init__()

        self.default_feature_joint_names = [
            "base.pos",
            "shoulder.pos",
            "elbow.pos",
            "wrist_pitch.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        self.viewer.eye = (1.0, -0.9, 0.8)
        self.viewer.lookat = (0.30, 0.0, 0.05)

        self.scene.table.spawn.visual_material.diffuse_color = self.table_color
        self.scene.ball.spawn.visual_material.diffuse_color = self.ball_color
        self.scene.target_square.spawn.visual_material.diffuse_color = self.target_color
        self.scene.background_panel.spawn.visual_material.diffuse_color = self.background_color
        self.scene.floor_panel.spawn.visual_material.diffuse_color = self.floor_color
        self.scene.left_wall.spawn.visual_material.diffuse_color = self.side_wall_color
        self.scene.right_wall.spawn.visual_material.diffuse_color = self.side_wall_color

        # Robot root pose: keep identity to avoid a spawn-time yaw snap.
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        # 로봇 base는 검은 영역을 움직이지 않고 X만 빨간 큐브 쪽으로 이동한 ROBOT_BASE_XY로 고정.
        self.scene.robot.init_state.pos = (
            ROBOT_BASE_XY[0],
            ROBOT_BASE_XY[1],
            TABLE_TOP_Z + ROBOT_BASE_Z_OFFSET,
        )
        # URDF home 자세 = 전 관절 0°. base joint도 0°에서 시작해 초기 회전 보정을 만들지 않는다.
        self.scene.robot.init_state.joint_pos = {
            "base": 0.0,
            "shoulder": 0.0,
            "elbow": 0.0,
            "wrist_pitch": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

        self.terminations.success.params["target_center"] = self.target_center_xy
        self.observations.subtask_terms.place_ball_in_square.params["target_center"] = self.target_center_xy

        rng = self.ball_position_range
        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform("ball", pose_range={"x": (-rng, rng), "y": (-rng, rng), "z": (0.0, 0.0)}),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                    convention="ros",
                ),
                randomize_camera_uniform(
                    "wrist",
                    pose_range={
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                    convention="ros",
                ),
                randomize_light_uniform("light", LIGHT_INTENSITY_RANGE, LIGHT_COLOR_RANGE),
                randomize_visual_material_color_uniform("table", TABLE_COLOR_RANGE),
                randomize_visual_material_color_uniform("ball", BALL_COLOR_RANGE),
                randomize_visual_material_color_uniform("background_panel", BACKGROUND_COLOR_RANGE),
                randomize_visual_material_color_uniform("floor_panel", FLOOR_COLOR_RANGE),
                randomize_visual_material_color_uniform("left_wall", SIDE_WALL_COLOR_RANGE),
                randomize_visual_material_color_uniform("right_wall", SIDE_WALL_COLOR_RANGE),
            ],
        )

    def build_lerobot_frame(self, episode_data, dataset_cfg) -> dict:
        """Build a LeRobot frame using the custom arm's native joint order.

        The template implementation converts through SO101 motor limits. This
        task uses a different arm and is driven by IK, so for LeRobot export we
        store native joint positions and let the recorder shift actions to the
        next-step joint state when ``derive_action_from_state`` is enabled.
        """
        obs_data = episode_data._data["obs"]
        joint_state_6d = obs_data["joint_pos"][-1].cpu().numpy()

        if getattr(dataset_cfg, "derive_action_from_state", False):
            processed_action = joint_state_6d.copy()
        else:
            action = episode_data._data["actions"][-1]
            if action.shape[-1] == joint_state_6d.shape[-1]:
                processed_action = action.cpu().numpy()
            else:
                processed_action = joint_state_6d.copy()

        frame = {
            "action": processed_action,
            "observation.state": joint_state_6d,
            "task": self.task_description,
        }
        for frame_key in dataset_cfg.features.keys():
            if not frame_key.startswith("observation.images"):
                continue
            camera_key = frame_key.split(".")[-1]
            frame[frame_key] = obs_data[camera_key][-1].cpu().numpy()

        return frame
