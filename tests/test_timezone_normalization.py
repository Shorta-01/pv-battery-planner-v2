import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt

import pandas as pd
from dateutil import tz
from zoneinfo import ZoneInfo

import planner_core as core


def test_estimate_window_consumption_kwh_tolerates_tzinfo_mismatch() -> None:
    start_local = dt.datetime(2026, 1, 10, 18, 30, tzinfo=ZoneInfo("Europe/Brussels"))
    end_local = dt.datetime(2026, 1, 10, 22, 15, tzinfo=tz.gettz("Europe/Brussels"))

    result = core.estimate_window_consumption_kwh(
        start_local=start_local,
        end_local=end_local,
        effective_daily_kwh=10.0,
    )

    assert isinstance(result, float)


def test_estimate_soc_at_offpeak_start_tolerates_tzinfo_mismatch() -> None:
    now_local = dt.datetime(2026, 1, 10, 16, 45, tzinfo=ZoneInfo("Europe/Brussels"))
    offpeak_start = pd.Timestamp(dt.datetime(2026, 1, 10, 22, 0, tzinfo=tz.gettz("Europe/Brussels")))

    result = core.estimate_soc_at_offpeak_start(
        soc_now_percent=50.0,
        now_local=now_local,
        offpeak_start=offpeak_start,
        effective_daily_kwh=10.0,
        pv_credit_kwh=0.0,
        battery_kwh=10.0,
        min_soc_percent=10.0,
        used_history=True,
        pv_credit_available=True,
    )

    assert isinstance(result, tuple)
    assert len(result) == 5
