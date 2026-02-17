import copy
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def _single_hour_weather(tz: str = "Europe/Brussels") -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(dt.datetime(2026, 6, 21, 12, 0), tz=tz)])
    return pd.DataFrame(
        {
            "ghi_wm2": [1000.0],
            "dni_wm2": [800.0],
            "dhi_wm2": [200.0],
            "cloud_cover_pct": [0.0],
            "temp_air_c": [25.0],
            "wind_speed_ms": [1.0],
        },
        index=idx,
    )


def test_resolve_pv_loss_multipliers_split_and_combined() -> None:
    dc_pr, ac_eff = core.resolve_pv_loss_multipliers(0.8, 0.95, "combined")
    assert dc_pr == pytest.approx(0.8)
    assert ac_eff == pytest.approx(1.0)

    dc_pr, ac_eff = core.resolve_pv_loss_multipliers(0.8, 0.95, "split")
    assert dc_pr == pytest.approx(0.8)
    assert ac_eff == pytest.approx(0.95)


def test_combined_mode_forces_inverter_eff_to_one() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["loss_model"] = "combined"
    cfg["pv"]["pv_loss_model"] = "combined"
    cfg["pv"]["inverter_eff"] = 0.95

    with pytest.warns(RuntimeWarning, match="forcing pv.inverter_eff=1.0"):
        merged = core.build_effective_config(cfg)

    assert merged["pv"]["inverter_eff"] == pytest.approx(1.0)


@pytest.mark.skipif(not core.PVLIB_AVAILABLE, reason="pvlib not installed")
def test_loss_models_prevent_pr_inverter_double_counting() -> None:
    weather = _single_hour_weather()
    loc = core.Location(name="lembeek", latitude=50.71864, longitude=4.21247)

    base = copy.deepcopy(core.DEFAULT_CONFIG)
    base["pv"]["array_east_panels"] = 1
    base["pv"]["array_south_panels"] = 1
    base["pv"]["panel_wp"] = 500
    base["pv"]["tilt_south_deg"] = 0.0
    base["pv"]["azimuth_south_deg"] = 180.0
    base["pv"]["tilt_east_deg"] = 0.0
    base["pv"]["azimuth_east_deg"] = 180.0
    base["pv"]["pv_calibration_factor"] = 1.0
    base["pv"]["pv_calibration_factor_south"] = 1.0
    base["pv"]["pv_calibration_factor_east"] = 1.0
    base["pv"]["inverter_ac_model"] = "linear"
    base["pv"]["inverter_ac_kw_limit"] = 100.0
    base["pv"]["performance_ratio"] = 0.8

    cfg_combined = copy.deepcopy(base)
    cfg_combined["pv"]["loss_model"] = "combined"
    cfg_combined["pv"]["pv_loss_model"] = "combined"
    cfg_combined["pv"]["inverter_eff"] = 1.0

    cfg_split = copy.deepcopy(base)
    cfg_split["pv"]["loss_model"] = "split"
    cfg_split["pv"]["pv_loss_model"] = "split"
    cfg_split["pv"]["inverter_eff"] = 0.95

    with core.applied_config(cfg_combined):
        out_combined = core.build_pv_forecast(weather, loc, tz="Europe/Brussels")
    with core.applied_config(cfg_split):
        out_split = core.build_pv_forecast(weather, loc, tz="Europe/Brussels")

    ac_combined = float(out_combined["pv_total_kwh"].iloc[0])
    ac_split = float(out_split["pv_total_kwh"].iloc[0])

    assert ac_split == pytest.approx(ac_combined * 0.95, rel=1e-2)
