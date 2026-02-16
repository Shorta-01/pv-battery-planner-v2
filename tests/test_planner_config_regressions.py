import copy
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_apply_config_backfills_per_array_calibration_from_global() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["pv_calibration_factor"] = 1.07
    cfg["pv"].pop("pv_calibration_factor_east", None)
    cfg["pv"].pop("pv_calibration_factor_south", None)

    core.apply_config(cfg)

    assert core.PV_CALIBRATION_FACTOR_EAST == 1.07
    assert core.PV_CALIBRATION_FACTOR_SOUTH == 1.07


def test_parse_offpeak_windows_supports_multi_window_roundtrip() -> None:
    windows = [
        [["22:00", "07:00"], ["13:00", "14:00"]],
        [["22:00", "07:00"]],
        [["22:00", "07:00"]],
        [["22:00", "07:00"]],
        [["22:00", "07:00"]],
        [["00:00", "24:00"]],
        [["00:00", "24:00"]],
    ]
    parsed = core.parse_offpeak_windows_by_dow(windows)
    assert parsed[0][0] == ("22:00", "07:00")
    assert parsed[0][1] == ("13:00", "14:00")


def test_normalize_hourly_index_uses_location_timezone_consistently() -> None:
    tz = "Europe/Brussels"
    day = dt.date(2026, 10, 25)
    idx = pd.date_range(pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz), periods=25, freq="h")
    df = pd.DataFrame(
        {
            "ghi_wm2": [0.0] * len(idx),
            "dni_wm2": [0.0] * len(idx),
            "dhi_wm2": [0.0] * len(idx),
            "cloud_cover_pct": [0.0] * len(idx),
            "temp_air_c": [10.0] * len(idx),
            "wind_speed_ms": [1.0] * len(idx),
        },
        index=idx,
    )
    out = core.normalize_hourly_forecast_index(df, day, tz)
    assert len(out) == 25
