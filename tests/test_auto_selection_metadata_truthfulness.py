import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
import weather_ensemble as we


@pytest.fixture
def hourly_index() -> pd.DatetimeIndex:
    return pd.date_range("2026-01-10", periods=24, freq="h", tz="Europe/Brussels")


def _fake_pv(df, _loc, tz=None):
    s = pd.Series([1.0] * len(df.index), index=df.index)
    return pd.DataFrame(
        {
            "pv_total_kwh": s,
            "pv_total_unclipped_kwh": s,
            "pv_east_kwh": s / 2,
            "pv_south_kwh": s / 2,
            "pv_clipped_kwh": [0.0] * len(df.index),
        },
        index=df.index,
    )


def test_auto_mode_does_not_report_manual(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    loc = core.Location(name="x", latitude=50.85, longitude=4.35)
    weather_df = pd.DataFrame(
        {
            "temp_air_c": [10.0] * len(hourly_index),
            "ghi_wm2": [0.0 if (i < 7 or i > 17) else 120.0 for i in range(len(hourly_index))],
            "dni_wm2": [0.0 if (i < 7 or i > 17) else 80.0 for i in range(len(hourly_index))],
            "dhi_wm2": [0.0 if (i < 7 or i > 17) else 35.0 for i in range(len(hourly_index))],
            "cloud_cover_pct": [25.0] * len(hourly_index),
            "wind_speed_ms": [1.0] * len(hourly_index),
            "weather_code": [1] * len(hourly_index),
            "precip_probability_pct": [0.0] * len(hourly_index),
            "precip_mm": [0.0] * len(hourly_index),
            "rain_mm": [0.0] * len(hourly_index),
        },
        index=hourly_index,
    )

    def fake_weather(model_id, *_args, **_kwargs):
        return core.ForecastResult(df=weather_df.copy(), sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()), [], False, {"source": "live", "live_failed_used_cached": False}

    monkeypatch.setattr(we, "fetch_open_meteo_weather", fake_weather)
    monkeypatch.setattr(core, "build_pv_forecast", _fake_pv)

    out = we.build_ensemble_forecast(
        loc=loc,
        target_date=dt.date(2026, 1, 10),
        tz="Europe/Brussels",
        weather_models=["knmi_harmonie_arome", "dwd_icon_d2", "ecmwf_ifs"],
        ensemble_method="weighted",
        pv_uncertainty=False,
        selection_mode="auto",
    )

    assert out.model_selection_reason != "manual"


def test_regional_plus_global_only_when_true() -> None:
    assert we._selection_reason_from_models(["knmi_harmonie_arome", "ecmwf_ifs"]) == "regional_plus_global"
    assert we._selection_reason_from_models(["knmi_harmonie_arome", "dwd_icon_d2"]) == "regional_only"
