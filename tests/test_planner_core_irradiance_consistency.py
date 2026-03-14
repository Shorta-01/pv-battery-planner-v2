import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


@pytest.fixture
def hourly_weather() -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=6, freq="h", tz="Europe/Brussels")
    return pd.DataFrame(
        {
            "ghi_wm2": [0.0, 200.0, 500.0, 600.0, 300.0, 0.0],
            "dni_wm2": [0.0, 300.0, 600.0, 700.0, 350.0, 0.0],
            "dhi_wm2": [0.0, 80.0, 120.0, 140.0, 90.0, 0.0],
            "cloud_cover_pct": [70.0, 50.0, 20.0, 10.0, 30.0, 80.0],
            "temp_air_c": [10.0, 12.0, 16.0, 18.0, 15.0, 11.0],
            "wind_speed_ms": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
        },
        index=idx,
    )


def test_valid_upstream_dni_dhi_does_not_require_disc(monkeypatch: pytest.MonkeyPatch, hourly_weather: pd.DataFrame) -> None:
    def _disc_raises(*_args, **_kwargs):
        raise AssertionError("disc should not be used")

    monkeypatch.setattr(core.pvlib.irradiance, "disc", _disc_raises, raising=True)

    loc = core.Location(name="x", latitude=50.85, longitude=4.35)
    east, south, *_ = core.estimate_pv_with_pvlib(hourly_weather, loc, tz="Europe/Brussels")
    assert east.notna().any()
    assert south.notna().any()


def test_missing_upstream_irradiance_still_repairs_safely(hourly_weather: pd.DataFrame) -> None:
    loc = core.Location(name="x", latitude=50.85, longitude=4.35)
    df = hourly_weather.drop(columns=["dni_wm2", "dhi_wm2"]).copy()
    east, south, *_ = core.estimate_pv_with_pvlib(df, loc, tz="Europe/Brussels")
    assert east.fillna(0.0).ge(0.0).all()
    assert south.fillna(0.0).ge(0.0).all()
