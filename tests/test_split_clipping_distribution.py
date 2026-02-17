import copy
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


@pytest.mark.skipif(not core.PVLIB_AVAILABLE, reason="pvlib not installed")
def test_clipping_distribution_keeps_east_south_sum_equal_total() -> None:
    tz = "Europe/Brussels"
    idx = pd.DatetimeIndex([pd.Timestamp(dt.datetime(2026, 6, 21, 12, 0), tz=tz)])
    weather = pd.DataFrame(
        {
            "ghi_wm2": [1000.0],
            "dni_wm2": [900.0],
            "dhi_wm2": [200.0],
            "cloud_cover_pct": [0.0],
            "temp_air_c": [20.0],
            "wind_speed_ms": [1.0],
        },
        index=idx,
    )

    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["array_east_panels"] = 20
    cfg["pv"]["array_south_panels"] = 20
    cfg["pv"]["panel_wp"] = 500
    cfg["pv"]["inverter_ac_kw_limit"] = 2.0
    cfg["pv"]["inverter_ac_model"] = "linear"
    cfg["pv"]["performance_ratio"] = 1.0
    cfg["pv"]["inverter_eff"] = 1.0
    cfg["pv"]["loss_model"] = "split"
    cfg["pv"]["pv_loss_model"] = "split"

    with core.applied_config(cfg):
        out = core.build_pv_forecast(weather, core.Location(name="x", latitude=50.8, longitude=4.3), tz=tz)

    total = float(out["pv_total_kwh"].iloc[0])
    split = float(out["pv_east_kwh"].iloc[0] + out["pv_south_kwh"].iloc[0])
    unclipped = float(out["pv_total_unclipped_kwh"].iloc[0])
    clipped = float(out["pv_clipped_kwh"].iloc[0])

    assert total == pytest.approx(split)
    assert unclipped >= total
    assert clipped == pytest.approx(unclipped - total)
