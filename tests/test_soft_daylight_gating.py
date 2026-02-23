import pandas as pd

from planner_core import (
    _apply_soft_daylight_factor_and_twilight_clamp,
    _soft_daylight_factor_from_elevation,
)


def test_soft_daylight_factor_from_elevation():
    assert _soft_daylight_factor_from_elevation(-10.0) == 0.0
    assert _soft_daylight_factor_from_elevation(-3.0) == 0.0
    mid = _soft_daylight_factor_from_elevation(1.5)
    assert 0.0 < mid < 1.0
    assert _soft_daylight_factor_from_elevation(6.0) == 1.0


def test_apply_factor_and_twilight_clamp_without_pvlib():
    idx = pd.date_range("2026-01-01 00:00", periods=4, freq="h", tz="Europe/Brussels")
    df = pd.DataFrame(
        {
            "pv_total_kwh": [0.005, 0.015, 0.4, float("nan")],
            "pv_east_kwh": [0.005, 0.009, 0.2, float("nan")],
        },
        index=idx,
    )
    factor = pd.Series([0.0, 0.2, 0.5, 1.0], index=idx)

    out = _apply_soft_daylight_factor_and_twilight_clamp(df, factor)

    assert out.loc[idx[0], "pv_total_kwh"] == 0.0
    assert out.loc[idx[1], "pv_total_kwh"] == 0.0
    assert out.loc[idx[2], "pv_total_kwh"] == 0.2
    assert pd.isna(out.loc[idx[3], "pv_total_kwh"])

    assert out.loc[idx[0], "pv_east_kwh"] == 0.0
    assert out.loc[idx[1], "pv_east_kwh"] == 0.0
    assert out.loc[idx[2], "pv_east_kwh"] == 0.1
    assert pd.isna(out.loc[idx[3], "pv_east_kwh"])
