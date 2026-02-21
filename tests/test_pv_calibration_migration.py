import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import planner_core


def test_migrate_legacy_global_relative_to_absolute():
    legacy_cfg = {
        "pv": {
            "pv_calibration_factor": 0.95,
            "pv_calibration_factor_east": 1.02,
            "pv_calibration_factor_south": 0.98,
        }
    }

    eff = planner_core.build_effective_config(legacy_cfg)

    assert "pv_calibration_factor" not in eff["pv"]
    assert eff["pv"]["pv_calibration_factor_east"] == pytest.approx(0.95 * 1.02, abs=1e-9)
    assert eff["pv"]["pv_calibration_factor_south"] == pytest.approx(0.95 * 0.98, abs=1e-9)


def test_no_global_key_keeps_absolute_values():
    cfg = {
        "pv": {
            "pv_calibration_factor_east": 1.01,
            "pv_calibration_factor_south": 0.99,
        }
    }

    eff = planner_core.build_effective_config(cfg)

    assert eff["pv"]["pv_calibration_factor_east"] == pytest.approx(1.01, abs=1e-9)
    assert eff["pv"]["pv_calibration_factor_south"] == pytest.approx(0.99, abs=1e-9)
