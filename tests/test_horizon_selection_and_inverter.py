import copy
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
import weather_ensemble as we


def test_auto_select_models_horizon_aware() -> None:
    tomorrow = we.auto_select_models_for_location(50.8, 4.3, requested_days=1)
    week = we.auto_select_models_for_location(50.8, 4.3, requested_days=7)

    assert len(tomorrow) <= 4
    assert len(week) >= 3
    assert len(week) <= 5
    assert "dwd_icon_d2" in tomorrow
    assert "dwd_icon_d2" not in week

    for model_id in tomorrow:
        assert we.get_model_caps(model_id)["max_days"] >= 1
    for model_id in week:
        caps = we.get_model_caps(model_id)
        assert caps["max_days"] >= 7
        assert caps["tier"] in {"medium", "global", "short"}


@pytest.mark.skipif(not core.PVLIB_AVAILABLE, reason="pvlib not installed")
def test_inverter_level_conversion_and_allocation_consistency() -> None:
    tz = "Europe/Brussels"
    idx = pd.date_range(pd.Timestamp(dt.datetime(2026, 6, 21, 6, 0), tz=tz), periods=8, freq="h")
    weather = pd.DataFrame(
        {
            "ghi_wm2": [200, 400, 650, 820, 900, 780, 500, 250],
            "dni_wm2": [120, 260, 420, 560, 650, 520, 300, 140],
            "dhi_wm2": [80, 140, 230, 260, 250, 260, 200, 120],
            "cloud_cover_pct": [30, 20, 15, 10, 8, 12, 20, 30],
            "temp_air_c": [12, 14, 17, 20, 23, 24, 21, 18],
            "wind_speed_ms": [1.2, 1.4, 1.5, 1.7, 2.0, 1.8, 1.6, 1.3],
        },
        index=idx,
    )

    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["array_east_panels"] = 16
    cfg["pv"]["array_south_panels"] = 16
    cfg["pv"]["panel_wp"] = 500
    cfg["pv"]["inverter_ac_kw_limit"] = 4.0
    cfg["pv"]["inverter_ac_model"] = "pvwatts"

    with core.applied_config(cfg):
        east, south, total_unclipped, total = core.estimate_pv_with_pvlib(
            weather, core.Location(name="x", latitude=50.8, longitude=4.3), tz=tz
        )

    assert (total_unclipped >= total).all()
    split_sum = (east + south)
    assert (split_sum - total).abs().max() < 1e-6
