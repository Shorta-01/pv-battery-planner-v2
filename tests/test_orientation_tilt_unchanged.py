import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import planner_core as core


def test_orientation_and_tilt_fields_preserved_in_effective_config():
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"].update(
        {
            "tilt_east_deg": 31.0,
            "tilt_south_deg": 34.0,
            "azimuth_east_deg": 92.0,
            "azimuth_south_deg": 182.0,
        }
    )
    out = core.build_effective_config(cfg)
    assert out["pv"]["tilt_east_deg"] == 31.0
    assert out["pv"]["tilt_south_deg"] == 34.0
    assert out["pv"]["azimuth_east_deg"] == 92.0
    assert out["pv"]["azimuth_south_deg"] == 182.0
