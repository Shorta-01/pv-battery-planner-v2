import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
from backend_api import pick_decision_quantile


def _expensive_window_df(day: dt.date) -> pd.DataFrame:
    idx = pd.date_range(
        pd.Timestamp(dt.datetime.combine(day, dt.time(7, 0)), tz="Europe/Brussels"),
        periods=4,
        freq="h",
    )
    return pd.DataFrame(
        {
            "pv_total_kwh": [2.0, 2.0, 2.0, 2.0],
            "pv_total_decision_kwh": [0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )


def test_compute_soc_low_timing_aware_respects_pv_col() -> None:
    day = dt.date(2026, 1, 12)  # Monday; includes expensive morning window
    df = _expensive_window_df(day)

    soc_with_p50 = core.compute_soc_low_timing_aware(
        df,
        total_consumption_kwh=12.0,
        for_date=day,
        pv_col="pv_total_kwh",
    )
    soc_with_decision = core.compute_soc_low_timing_aware(
        df,
        total_consumption_kwh=12.0,
        for_date=day,
        pv_col="pv_total_decision_kwh",
    )

    assert soc_with_decision > soc_with_p50


def test_compute_soc_high_headroom_uses_expected_p50() -> None:
    day = dt.date(2026, 1, 12)
    idx = pd.date_range(
        pd.Timestamp(dt.datetime.combine(day, dt.time(7, 0)), tz="Europe/Brussels"),
        periods=4,
        freq="h",
    )
    df = pd.DataFrame(
        {
            "pv_total_kwh": [4.0, 4.0, 4.0, 4.0],
            "pv_total_decision_kwh": [0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )

    surplus_1, soc_high_1 = core.compute_soc_high_headroom(
        df,
        total_consumption_kwh=4.0,
        for_date=day,
        sunrise=idx[0].to_pydatetime(),
        sunset=idx[-1].to_pydatetime(),
    )

    df["pv_total_decision_kwh"] = 10.0
    surplus_2, soc_high_2 = core.compute_soc_high_headroom(
        df,
        total_consumption_kwh=4.0,
        for_date=day,
        sunrise=idx[0].to_pydatetime(),
        sunset=idx[-1].to_pydatetime(),
    )

    assert surplus_1 > 0.0
    assert surplus_1 == surplus_2
    assert soc_high_1 == soc_high_2


def test_pick_decision_quantile_switches_on_confidence() -> None:
    assert pick_decision_quantile("Low") == ("p10", "low_confidence")
    assert pick_decision_quantile("Medium") == ("p25", "normal")
    assert pick_decision_quantile("High") == ("p25", "normal")
