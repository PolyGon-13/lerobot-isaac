"""Convert the (gripper-corrected) Issac_sim_6DOF_ARM URDF -> USD for Isaac/leisaac.

Reads the flat URDF in this folder (urdf/Issac_sim_6DOF_ARM.urdf), rewrites the
`package://lerobot_ver2/...` mesh paths to absolute, and converts with the
IsaacLab UrdfConverter. Output USD is written to this model folder as `lerobot_arm.usd`.

Run (leisaac conda env):
    conda activate leisaac && source /opt/isaac-sim-5.1.0/setup_conda_env.sh \
        && python /home/polygon/lerobot_isaac/lerobot/lerobot_ver2/convert_to_usd.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import xml.etree.ElementTree as ET  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Gf, Usd  # noqa: E402

PKG_NAME = "lerobot_ver2"
PKG_ROOT = f"/home/polygon/lerobot_isaac/lerobot/{PKG_NAME}"
SRC = f"{PKG_ROOT}/urdf/Issac_sim_6DOF_ARM.urdf"
TMP = f"{PKG_ROOT}/urdf/_isaac_convert.urdf"
OUT_DIR = PKG_ROOT
OUT_NAME = "lerobot_arm.usd"

# --- 1. preprocess URDF -------------------------------------------------
tree = ET.parse(SRC)
root = tree.getroot()

# package://lerobot_ver2/...  ->  absolute path (no ROS pkg resolution here)
n_fixed = 0
for mesh in root.iter("mesh"):
    fn = mesh.get("filename")
    if fn and fn.startswith(f"package://{PKG_NAME}/"):
        mesh.set("filename", fn.replace(f"package://{PKG_NAME}/", PKG_ROOT + "/"))
        n_fixed += 1
print(f"[preprocess] rewrote {n_fixed} mesh paths to absolute")

tree.write(TMP)
print(f"[preprocess] wrote temp URDF: {TMP}")

# --- 2. convert ---------------------------------------------------------
cfg = UrdfConverterCfg(
    asset_path=TMP,
    usd_dir=OUT_DIR,
    usd_file_name=OUT_NAME,
    fix_base=True,
    merge_fixed_joints=True,
    force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=17.8, damping=0.6),
    ),
)
converter = UrdfConverter(cfg)
print("[convert] USD written to:", converter.usd_path)

# --- 3. postprocess -----------------------------------------------------
# Keep the authored USD pose aligned with the reset pose. The task-level root
# rotation handles the scene orientation; base_link itself should not flash at
# a different pre-reset orientation.
stage = Usd.Stage.Open(converter.usd_path)
base_link = stage.GetPrimAtPath("/Issac_sim_6DOF_ARM/base_link")
base_link.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))

base_joint = stage.GetPrimAtPath("/Issac_sim_6DOF_ARM/joints/base")
base_joint.GetAttribute("state:angular:physics:position").Set(0.0)
base_joint.GetAttribute("drive:angular:physics:targetPosition").Set(0.0)
base_joint.GetAttribute("physics:lowerLimit").Set(-180.0)
base_joint.GetAttribute("physics:upperLimit").Set(270.0)
stage.GetRootLayer().Save()
print("[postprocess] base_link yaw=0deg, base joint default=0deg, base limit=[-180,270]deg")

simulation_app.close()
