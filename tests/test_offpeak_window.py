import datetime as dt

import planner_core as core


def _tariff_with_same_windows(windows):
    return {"offpeak_windows_by_dow": [windows for _ in range(7)]}


def test_compute_charging_window_classic_overnight():
    cfg = _tariff_with_same_windows([["22:00", "07:00"]])
    target_date = dt.date(2026, 2, 24)

    start, end = core.compute_charging_window_for_target_date(target_date, cfg)

    assert start == core.pd.Timestamp("2026-02-23 22:00", tz=core.TIMEZONE)
    assert end == core.pd.Timestamp("2026-02-24 07:00", tz=core.TIMEZONE)


def test_compute_charging_window_shifted_overnight():
    cfg = _tariff_with_same_windows([["23:00", "06:00"]])
    target_date = dt.date(2026, 2, 24)

    start, end = core.compute_charging_window_for_target_date(target_date, cfg)

    assert start == core.pd.Timestamp("2026-02-23 23:00", tz=core.TIMEZONE)
    assert end == core.pd.Timestamp("2026-02-24 06:00", tz=core.TIMEZONE)


def test_compute_charging_window_all_day_offpeak():
    cfg = _tariff_with_same_windows([["00:00", "24:00"]])
    target_date = dt.date(2026, 2, 24)

    start, end = core.compute_charging_window_for_target_date(target_date, cfg)

    assert start == core.pd.Timestamp("2026-02-23 00:00", tz=core.TIMEZONE)
    assert end == core.pd.Timestamp("2026-02-24 00:00", tz=core.TIMEZONE)


def test_compute_charging_window_picks_midnight_crossing_window():
    cfg = _tariff_with_same_windows([["01:00", "05:00"], ["23:00", "06:00"]])
    target_date = dt.date(2026, 2, 24)

    start, end = core.compute_charging_window_for_target_date(target_date, cfg)

    assert start == core.pd.Timestamp("2026-02-23 23:00", tz=core.TIMEZONE)
    assert end == core.pd.Timestamp("2026-02-24 06:00", tz=core.TIMEZONE)
