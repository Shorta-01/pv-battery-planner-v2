import pandas as pd

from planner_core import _apply_last_resort_ghi, _integrate_hourly_power_trapezoid


def test_integrate_hourly_power_trapezoid():
    idx = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="Europe/Brussels")
    power = pd.Series([0.0, 10.0, 10.0], index=idx)
    out = _integrate_hourly_power_trapezoid(power)
    assert list(out.values) == [5.0, 10.0, 10.0]


def test_last_resort_ghi_fill_respects_allow_mask():
    idx = pd.date_range("2026-01-01 12:00", periods=1, freq="h", tz="Europe/Brussels")
    provider_ghi = pd.Series([float("nan")], index=idx)
    ghi_candidate = pd.Series([800.0], index=idx)

    blocked = _apply_last_resort_ghi(provider_ghi, ghi_candidate, pd.Series([False], index=idx))
    allowed = _apply_last_resort_ghi(provider_ghi, ghi_candidate, pd.Series([True], index=idx))

    assert blocked.isna().all()
    assert float(allowed.iloc[0]) == 800.0


def test_15min_bucket_to_hour_start():
    idx = pd.to_datetime(
        [
            "2026-01-01 09:15",
            "2026-01-01 09:30",
            "2026-01-01 09:45",
            "2026-01-01 10:00",
        ],
        utc=True,
    ).tz_convert("Europe/Brussels")
    power_kw = pd.Series([4.0, 4.0, 4.0, 4.0], index=idx)
    e15 = power_kw * 0.25
    bucket = idx.ceil("h") - pd.Timedelta(hours=1)
    hourly = e15.groupby(bucket).sum(min_count=1)

    assert float(hourly.iloc[0]) == 4.0
