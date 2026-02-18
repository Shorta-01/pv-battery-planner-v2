import copy
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def _hourly_index(day: dt.date, tz: str = "Europe/Brussels") -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz), periods=24, freq="h")


def test_soc_high_counts_early_morning_daylight_surplus() -> None:
    day = dt.date(2026, 6, 15)
    idx = _hourly_index(day)
    df = pd.DataFrame({"pv_total_kwh": 0.0}, index=idx)
    df.loc[idx[6], "pv_total_kwh"] = 4.0  # 06:00 surplus (inside daylight, outside weekday expensive hours)

    surplus_daylight, soc_high_daylight = core.compute_soc_high_headroom(
        df,
        total_consumption_kwh=0.0,
        for_date=day,
        sunrise=pd.Timestamp(dt.datetime.combine(day, dt.time(5, 30)), tz="Europe/Brussels").to_pydatetime(),
        sunset=pd.Timestamp(dt.datetime.combine(day, dt.time(22, 0)), tz="Europe/Brussels").to_pydatetime(),
    )
    surplus_old_like, soc_high_old_like = core.compute_soc_high_headroom(
        df,
        total_consumption_kwh=0.0,
        for_date=day,
        sunrise=pd.Timestamp(dt.datetime.combine(day, dt.time(7, 0)), tz="Europe/Brussels").to_pydatetime(),
        sunset=pd.Timestamp(dt.datetime.combine(day, dt.time(22, 0)), tz="Europe/Brussels").to_pydatetime(),
    )

    assert surplus_daylight == pytest.approx(4.0)
    assert surplus_old_like == pytest.approx(0.0)
    assert soc_high_daylight < soc_high_old_like


def test_soc_high_headroom_respects_battery_charge_power_cap() -> None:
    day = dt.date(2026, 6, 16)
    idx = _hourly_index(day)
    df = pd.DataFrame({"pv_total_kwh": 0.0}, index=idx)
    df.loc[idx[12], "pv_total_kwh"] = 8.0  # large one-hour surplus

    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["battery"]["battery_max_charge_kw"] = 2.0

    with core.applied_config(cfg):
        surplus, soc_high = core.compute_soc_high_headroom(
            df,
            total_consumption_kwh=0.0,
            for_date=day,
            sunrise=pd.Timestamp(dt.datetime.combine(day, dt.time(5, 0)), tz="Europe/Brussels").to_pydatetime(),
            sunset=pd.Timestamp(dt.datetime.combine(day, dt.time(22, 0)), tz="Europe/Brussels").to_pydatetime(),
        )

        energy_only_soc_high = 1.0 - ((surplus * core.BATTERY_PV_CHARGE_EFF) / core.BATTERY_KWH)
        capped_soc_high = 1.0 - ((2.0 * 1.0) / core.BATTERY_KWH)

        assert surplus == pytest.approx(8.0)
        assert soc_high == pytest.approx(capped_soc_high)
        assert soc_high > energy_only_soc_high


def test_choose_cutoff_soc_uses_soc_high_on_conflict() -> None:
    day = dt.date(2026, 6, 15)
    cutoff_soc, reason = core.choose_cutoff_soc(day, soc_low=0.80, soc_high=0.55)

    assert cutoff_soc == pytest.approx(0.55)
    assert "Using headroom target" in reason
