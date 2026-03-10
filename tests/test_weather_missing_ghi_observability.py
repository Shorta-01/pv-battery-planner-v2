import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
import weather_ensemble as we


def _base_payload(idx: pd.DatetimeIndex) -> dict:
    return {
        "hourly": {
            "time": [ts.isoformat() for ts in idx],
            "temperature_2m": [10.0] * len(idx),
            "wind_speed_10m": [1.0] * len(idx),
            "cloud_cover": [20.0] * len(idx),
        },
        "daily": {
            "sunrise": [idx[0].isoformat()],
            "sunset": [idx[-1].isoformat()],
        },
    }


def test_missing_shortwave_with_dni_dhi_reports_ghi_missing(monkeypatch) -> None:
    idx = pd.date_range("2026-01-10 00:00", periods=24, freq="h", tz="Europe/Brussels")
    payload = _base_payload(idx)
    payload["hourly"].update({
        "direct_normal_irradiance": [200.0] * len(idx),
        "diffuse_radiation": [60.0] * len(idx),
    })

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", lambda *args, **kwargs: payload)

    _, missing_vars, _, fetch_meta = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="BE", latitude=50.85, longitude=4.35),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )

    assert "shortwave_radiation" in missing_vars
    assert fetch_meta["ghi_missing"] is True
    assert fetch_meta["ghi_missing_hours"] == 24


def test_missing_ghi_without_fallback_reports_missing_vars(monkeypatch) -> None:
    idx = pd.date_range("2026-01-11 00:00", periods=24, freq="h", tz="Europe/Brussels")
    payload = _base_payload(idx)

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", lambda *args, **kwargs: payload)

    _, missing_vars, _, fetch_meta = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="BE", latitude=50.85, longitude=4.35),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 11),
    )

    assert "shortwave_radiation" in missing_vars
    assert "direct_normal_irradiance" in missing_vars
    assert "diffuse_radiation" in missing_vars
    assert fetch_meta["ghi_missing"] is True
    assert fetch_meta["ghi_missing_hours"] == 24


def test_normal_ghi_present_keeps_ghi_missing_false(monkeypatch) -> None:
    idx = pd.date_range("2026-01-12 00:00", periods=24, freq="h", tz="Europe/Brussels")
    payload = _base_payload(idx)
    payload["hourly"].update({
        "shortwave_radiation": [0.0] * len(idx),
        "direct_normal_irradiance": [0.0] * len(idx),
        "diffuse_radiation": [0.0] * len(idx),
    })

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", lambda *args, **kwargs: payload)

    _, missing_vars, _, fetch_meta = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="BE", latitude=50.85, longitude=4.35),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 12),
    )

    assert "shortwave_radiation" not in missing_vars
    assert fetch_meta["ghi_missing"] is False
    assert fetch_meta["ghi_missing_hours"] == 0

