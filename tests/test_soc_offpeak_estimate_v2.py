import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt

import pandas as pd
from zoneinfo import ZoneInfo

import planner_core as core


def test_tod_weighting_higher_during_peak_overlap() -> None:
    tz = ZoneInfo("Europe/Brussels")
    non_peak = core.estimate_window_consumption_kwh(
        start_local=dt.datetime(2026, 1, 10, 10, 0, tzinfo=tz),
        end_local=dt.datetime(2026, 1, 10, 13, 0, tzinfo=tz),
        effective_daily_kwh=24.0,
    )
    peak = core.estimate_window_consumption_kwh(
        start_local=dt.datetime(2026, 1, 10, 17, 0, tzinfo=tz),
        end_local=dt.datetime(2026, 1, 10, 20, 0, tzinfo=tz),
        effective_daily_kwh=24.0,
    )
    assert peak > non_peak


def test_pv_credit_b2_math() -> None:
    load = core.estimate_window_consumption_kwh(
        start_local=dt.datetime(2026, 1, 10, 12, 0, tzinfo=ZoneInfo("Europe/Brussels")),
        end_local=dt.datetime(2026, 1, 10, 16, 0, tzinfo=ZoneInfo("Europe/Brussels")),
        effective_daily_kwh=24.0,
    )
    pv_remaining = 3.0
    pv_credit = min(pv_remaining * 0.5, load)
    assert pv_credit <= load
    assert pv_credit == 1.5


def test_min_soc_floor_is_respected() -> None:
    tz = ZoneInfo("Europe/Brussels")
    soc_est, _, _, _, _ = core.estimate_soc_at_offpeak_start(
        soc_now_percent=20.0,
        now_local=dt.datetime(2026, 1, 10, 10, 0, tzinfo=tz),
        offpeak_start=pd.Timestamp(dt.datetime(2026, 1, 10, 22, 0, tzinfo=tz)),
        effective_daily_kwh=40.0,
        pv_credit_kwh=0.0,
        battery_kwh=10.0,
        min_soc_percent=15.0,
        used_history=True,
        pv_credit_available=True,
    )
    assert soc_est == 15.0


def test_confidence_downgrade_rules() -> None:
    tz = ZoneInfo("Europe/Brussels")
    _, _, conf_peak, _, _ = core.estimate_soc_at_offpeak_start(
        soc_now_percent=80.0,
        now_local=dt.datetime(2026, 1, 10, 16, 0, tzinfo=tz),
        offpeak_start=pd.Timestamp(dt.datetime(2026, 1, 10, 19, 0, tzinfo=tz)),
        effective_daily_kwh=10.0,
        pv_credit_kwh=0.0,
        battery_kwh=10.0,
        min_soc_percent=10.0,
        used_history=True,
        pv_credit_available=True,
    )
    assert conf_peak == "Low"

    _, _, conf_daytime_no_pv, _, _ = core.estimate_soc_at_offpeak_start(
        soc_now_percent=80.0,
        now_local=dt.datetime(2026, 1, 10, 9, 0, tzinfo=tz),
        offpeak_start=pd.Timestamp(dt.datetime(2026, 1, 10, 10, 0, tzinfo=tz)),
        effective_daily_kwh=10.0,
        pv_credit_kwh=0.0,
        battery_kwh=10.0,
        min_soc_percent=10.0,
        used_history=True,
        pv_credit_available=False,
    )
    assert conf_daytime_no_pv == "Low"

    _, _, conf_no_history, _, _ = core.estimate_soc_at_offpeak_start(
        soc_now_percent=80.0,
        now_local=dt.datetime(2026, 1, 10, 9, 0, tzinfo=tz),
        offpeak_start=pd.Timestamp(dt.datetime(2026, 1, 10, 10, 0, tzinfo=tz)),
        effective_daily_kwh=10.0,
        pv_credit_kwh=0.0,
        battery_kwh=10.0,
        min_soc_percent=10.0,
        used_history=False,
        pv_credit_available=True,
    )
    assert conf_no_history == "Medium"
