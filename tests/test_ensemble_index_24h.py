import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
import weather_ensemble as we


@pytest.mark.parametrize(
    ("target_date", "expected_len"),
    [
        (dt.date(2026, 3, 29), 23),
        (dt.date(2026, 10, 25), 25),
        (dt.date(2026, 6, 21), 24),
    ],
)
def test_ensemble_output_matches_local_day_index(monkeypatch, target_date, expected_len):
    loc = core.Location(name="x", latitude=50.8, longitude=4.3)
    tz = "Europe/Brussels"

    idx_short = pd.date_range(
        pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=tz),
        periods=20,
        freq="h",
    )

    def fake_weather(model_id, *_args, **_kwargs):
        df = pd.DataFrame(
            {
                "temp_air_c": [10.0] * len(idx_short),
                "ghi_wm2": [0.0] * len(idx_short),
                "dni_wm2": [0.0] * len(idx_short),
                "dhi_wm2": [0.0] * len(idx_short),
                "cloud_cover_pct": [0.0] * len(idx_short),
                "wind_speed_ms": [1.0] * len(idx_short),
                "temperature_2m": [10.0] * len(idx_short),
                "wind_speed_10m": [1.0] * len(idx_short),
                "cloud_cover": [0.0] * len(idx_short),
                "shortwave_radiation": [0.0] * len(idx_short),
                "direct_normal_irradiance": [0.0] * len(idx_short),
                "diffuse_radiation": [0.0] * len(idx_short),
            },
            index=idx_short,
        )
        fetch_meta = {
            "requested_days": 1,
            "horizon_days": 1,
            "model_max_days": 1,
            "derived_irradiance_hours": 0,
        }
        return core.ForecastResult(df=df, sunrise=idx_short[7].to_pydatetime(), sunset=idx_short[17].to_pydatetime()), [], False, fetch_meta

    def fake_build_pv(df, _loc, tz=None):
        s = pd.Series([1.0] * len(df.index), index=df.index)
        return pd.DataFrame(
            {
                "pv_total_kwh": s,
                "pv_total_unclipped_kwh": s,
                "pv_east_kwh": s / 2,
                "pv_south_kwh": s / 2,
                "pv_clipped_kwh": s * 0,
            },
            index=df.index,
        )

    monkeypatch.setattr(we, "fetch_open_meteo_weather", fake_weather)
    monkeypatch.setattr(core, "build_pv_forecast", fake_build_pv)

    out = we.build_ensemble_forecast(
        loc=loc,
        target_date=target_date,
        tz=tz,
        weather_models=["ecmwf_ifs", "dwd_icon_d2"],
        ensemble_method="mean",
        pv_uncertainty=False,
    )

    expected_index = we.local_day_hourly_index(target_date, tz)

    assert len(expected_index) == expected_len
    assert list(out.pv_ensemble_p50.index) == list(expected_index)
    assert list(out.weather_ensemble_table.df.index) == list(expected_index)
