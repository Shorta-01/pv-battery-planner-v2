import datetime as dt
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
import ui_shared
import weather_ensemble as we


def test_compute_clear_sky_reference_kwh_does_not_crash() -> None:
    idx = pd.date_range("2026-06-01 00:00", periods=24, freq="h", tz="Europe/Brussels")
    weather_df = pd.DataFrame(
        {
            "temp_air_c": [18.0] * len(idx),
            "wind_speed_ms": [2.0] * len(idx),
            "cloud_cover_pct": [45.0] * len(idx),
        },
        index=idx,
    )
    loc = core.Location(name="BE", latitude=50.85, longitude=4.35, elevation_m=60)

    out = ui_shared.compute_clear_sky_reference_kwh(weather_df, loc)

    assert isinstance(out, float)
    assert math.isfinite(out)
    assert out >= 0.0


def test_compute_allow_synth_mask_is_conservative() -> None:
    idx = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="Europe/Brussels")
    model_a = pd.DataFrame({"ghi_wm2": [float("nan"), 100.0, float("nan")]}, index=idx)
    model_b = pd.DataFrame({"ghi_wm2": [float("nan"), float("nan"), 50.0]}, index=idx)
    weather_ok = {
        "model_a": SimpleNamespace(df=model_a),
        "model_b": SimpleNamespace(df=model_b),
    }

    out = we._compute_allow_synth_mask(weather_ok, idx)

    expected = pd.Series([True, False, False], index=idx, dtype=bool, name="ghi_wm2")
    pd.testing.assert_series_equal(out, expected)


def test_fetch_open_meteo_keeps_missing_ghi_as_nan(monkeypatch) -> None:
    idx = pd.date_range("2026-01-10 00:00", periods=3, freq="h", tz="Europe/Brussels")
    payload = {
        "hourly": {
            "time": [ts.isoformat() for ts in idx],
            "temperature_2m": [10.0, 11.0, 12.0],
            "wind_speed_10m": [1.0, 1.0, 1.0],
            "cloud_cover": [10.0, 20.0, 30.0],
            "shortwave_radiation": [None, 0.0, 100.0],
            "direct_normal_irradiance": [None, 0.0, 70.0],
            "diffuse_radiation": [None, 0.0, 30.0],
        },
        "daily": {
            "sunrise": [idx[0].isoformat()],
            "sunset": [idx[-1].isoformat()],
        },
    }

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", lambda *args, **kwargs: payload)

    forecast, _, _, _ = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="BE", latitude=50.85, longitude=4.35),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )

    ghi = forecast.df["ghi_wm2"].iloc[:3]
    assert float(ghi.iloc[0]) == 0.0
    assert float(ghi.iloc[1]) == 100.0
    assert pd.isna(ghi.iloc[2])
