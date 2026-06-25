from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the Kitchen Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

KITCHEN_WITH_ORANGE_USD_PATH = str(SCENES_ROOT / "kitchen_with_orange" / "scene.usd")

# Composition-override layers that subLayer scene.usd and deactivate the extra oranges,
# so the scene visually/physically contains only N oranges. Used by the PickOneOrange /
# PickTwoOrange task variants for scene-consistent data collection and evaluation.
KITCHEN_WITH_ONE_ORANGE_USD_PATH = str(SCENES_ROOT / "kitchen_with_orange" / "scene_one_orange.usda")
KITCHEN_WITH_TWO_ORANGE_USD_PATH = str(SCENES_ROOT / "kitchen_with_orange" / "scene_two_orange.usda")

# Maps num_oranges -> scene USD path (3 = original full scene).
KITCHEN_ORANGE_USD_PATH_BY_COUNT = {
    1: KITCHEN_WITH_ONE_ORANGE_USD_PATH,
    2: KITCHEN_WITH_TWO_ORANGE_USD_PATH,
    3: KITCHEN_WITH_ORANGE_USD_PATH,
}

KITCHEN_WITH_ORANGE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KITCHEN_WITH_ORANGE_USD_PATH,
    )
)

KITCHEN_WITH_HAMBURGER_USD_PATH = str(SCENES_ROOT / "kitchen_with_hamburger" / "scene.usd")

KITCHEN_WITH_HAMBURGER_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KITCHEN_WITH_HAMBURGER_USD_PATH,
    )
)
