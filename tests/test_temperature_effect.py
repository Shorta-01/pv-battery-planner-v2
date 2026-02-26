import datetime as dt
import pandas as pd
import planner_core as core


def test_hotter_day_reduces_pv_same_irradiance():
    tz = "Europe/Brussels"
    idx = pd.date_range(pd.Timestamp("2026-06-20 06:00", tz=tz), periods=14, freq="h")
    base = pd.DataFrame(index=idx)
    base["ghi_wm2"] = 700.0
    base["dni_wm2"] = 500.0
    base["dhi_wm2"] = 200.0
    base["wind_speed_ms"] = 1.0
    base["cloud_cover_pct"] = 20.0

    cool = base.copy()
    cool["temp_air_c"] = 10.0
    hot = base.copy()
    hot["temp_air_c"] = 35.0

    loc = core.Location(name="x", latitude=50.85, longitude=4.35, elevation_m=20)
    cool_pv = core.build_pv_forecast(cool, loc, tz=tz)
    hot_pv = core.build_pv_forecast(hot, loc, tz=tz)
    assert float(hot_pv["pv_total_kwh"].sum()) < float(cool_pv["pv_total_kwh"].sum())
