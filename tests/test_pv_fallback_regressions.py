import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_estimate_pv_with_pvlib_cloud_fallback_keeps_overcast_transmittance_floor() -> None:
    if not core.PVLIB_AVAILABLE:
        pytest.skip("pvlib not installed")

    tz = "Europe/Brussels"
    idx = pd.DatetimeIndex([pd.Timestamp(dt.datetime(2026, 6, 21, 12, 0), tz=tz)])
    weather = pd.DataFrame(
        {
            "ghi_wm2": [float("nan")],
            "dni_wm2": [float("nan")],
            "dhi_wm2": [float("nan")],
            "cloud_cover_pct": [100.0],
            "temp_air_c": [20.0],
            "wind_speed_ms": [1.0],
        },
        index=idx,
    )

    east_kw, south_kw, *_rest = core.estimate_pv_with_pvlib(
        weather,
        core.Location(name="lembeek", latitude=50.71864, longitude=4.21247),
        tz=tz,
    )

    assert float(east_kw.iloc[0]) > 0.0
    assert float(south_kw.iloc[0]) > 0.0
