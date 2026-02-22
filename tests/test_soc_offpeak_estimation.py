import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt

import pandas as pd
from zoneinfo import ZoneInfo

import planner_core as core


def test_estimate_soc_at_offpeak_start_simple_2h_gap() -> None:
    tz = ZoneInfo("Europe/Brussels")
    now_local = dt.datetime(2026, 1, 10, 20, 0, tzinfo=tz)
    offpeak_start = pd.Timestamp(dt.datetime(2026, 1, 10, 22, 0, tzinfo=tz))

    soc_est, hours_until, confidence, method, _debug = core.estimate_soc_at_offpeak_start(
        soc_now_percent=50.0,
        now_local=now_local,
        offpeak_start=offpeak_start,
        effective_daily_kwh=24.0,
        pv_credit_kwh=0.0,
        battery_kwh=10.0,
        min_soc_percent=15.0,
        used_history=True,
        pv_credit_available=True,
    )

    assert hours_until == 2.0
    assert soc_est < 50.0
    assert confidence == "Low"
    assert method == "history+tod_weight+pv0"


def test_estimate_soc_at_offpeak_start_clamps_to_min_soc() -> None:
    tz = ZoneInfo("Europe/Brussels")
    now_local = dt.datetime(2026, 1, 10, 16, 0, tzinfo=tz)
    offpeak_start = pd.Timestamp(dt.datetime(2026, 1, 10, 22, 0, tzinfo=tz))

    soc_est, hours_until, confidence, method, _debug = core.estimate_soc_at_offpeak_start(
        soc_now_percent=20.0,
        now_local=now_local,
        offpeak_start=offpeak_start,
        effective_daily_kwh=24.0,
        pv_credit_kwh=0.0,
        battery_kwh=10.0,
        min_soc_percent=15.0,
        used_history=True,
        pv_credit_available=True,
    )

    assert hours_until == 6.0
    assert soc_est == 15.0
    assert confidence in {"Medium", "Low"}
    assert method == "history+tod_weight+pv0"


def test_estimate_soc_at_offpeak_start_far_away_low_confidence() -> None:
    tz = ZoneInfo("Europe/Brussels")
    now_local = dt.datetime(2026, 1, 10, 14, 0, tzinfo=tz)
    offpeak_start = pd.Timestamp(dt.datetime(2026, 1, 10, 22, 0, tzinfo=tz))

    soc_est, hours_until, confidence, method, _debug = core.estimate_soc_at_offpeak_start(
        soc_now_percent=65.0,
        now_local=now_local,
        offpeak_start=offpeak_start,
        effective_daily_kwh=24.0,
        pv_credit_kwh=0.0,
        battery_kwh=10.0,
        min_soc_percent=15.0,
        used_history=True,
        pv_credit_available=True,
    )

    assert hours_until == 8.0
    assert soc_est == 15.0
    assert confidence == "Low"
    assert method == "history+tod_weight+pv0"
