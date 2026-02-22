import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


@pytest.mark.parametrize(
    ("east_panels", "south_panels"),
    [
        (0, 10),
        (10, 0),
    ],
)
def test_validate_config_accepts_single_array_systems(east_panels: int, south_panels: int) -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["array_east_panels"] = east_panels
    cfg["pv"]["array_south_panels"] = south_panels

    core.validate_config(cfg)


def test_validate_config_rejects_no_pv_arrays() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["array_east_panels"] = 0
    cfg["pv"]["array_south_panels"] = 0

    with pytest.raises(ValueError, match="at least one PV array must have > 0 panels"):
        core.validate_config(cfg)
