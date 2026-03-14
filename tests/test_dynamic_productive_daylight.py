import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import weather_ensemble as we
import planner_core as core


def _forecast(idx):
    return core.ForecastResult(df=pd.DataFrame(index=idx), sunrise=idx[7].to_pydatetime(), sunset=idx[17].to_pydatetime())


def test_productive_daylight_not_fixed_8_17() -> None:
    idx = pd.date_range("2026-12-21 00:00", periods=24, freq="h", tz="Europe/Brussels")
    mask = we._productive_daylight_mask(idx, _forecast(idx))
    assert bool(mask.loc[idx[12]])
    assert not bool(mask.loc[idx[17]])


def test_nighttime_rain_does_not_force_fronty_wet() -> None:
    idx = pd.date_range("2026-01-10 00:00", periods=24, freq="h", tz="Europe/Brussels")
    wet_night = pd.DataFrame({
        "cloud_cover_pct": [20.0] * 24,
        "weather_code": [61.0 if h < 6 or h > 20 else 1.0 for h in range(24)],
        "precip_probability_pct": [90.0 if h < 6 or h > 20 else 5.0 for h in range(24)],
        "precip_mm": [0.8 if h < 6 or h > 20 else 0.0 for h in range(24)],
        "rain_mm": [0.5 if h < 6 or h > 20 else 0.0 for h in range(24)],
    }, index=idx)
    weather = {"ecmwf_ifs": core.ForecastResult(df=wet_night, sunrise=idx[7].to_pydatetime(), sunset=idx[17].to_pydatetime())}
    weather["ecmwf_ifs"].latitude = 50.85
    weather["ecmwf_ifs"].longitude = 4.35
    assert we._classify_day_type_from_ensemble(weather, idx) != "fronty_wet"
