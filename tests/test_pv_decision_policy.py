import datetime as dt

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import planner_core as core


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


def test_run_detailed_plan_prefers_decision_pv_column(monkeypatch: "pytest.MonkeyPatch") -> None:
    day = dt.date(2026, 1, 12)
    idx = pd.date_range(pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz="Europe/Brussels"), periods=24, freq="h")
    weather_df = pd.DataFrame({"weather_code": [1] * len(idx)}, index=idx)
    weather = core.ForecastResult(df=weather_df, sunrise=idx[7].to_pydatetime(), sunset=idx[17].to_pydatetime())

    pv_df = pd.DataFrame({"pv_total_kwh": [1.0] * len(idx), "pv_total_decision_kwh": [0.5] * len(idx)}, index=idx)
    cons = pd.Series([0.5] * len(idx), index=idx)

    captured: dict[str, str] = {}

    def fake_soc_low(df, total_consumption_kwh, for_date, buffer_soc=0.0, tariff_cfg=None, pv_col="pv_total_kwh"):
        captured["pv_col"] = pv_col
        return 0.2

    monkeypatch.setattr(core, "compute_soc_low_timing_aware", fake_soc_low)
    monkeypatch.setattr(core, "compute_soc_high_headroom", lambda *_args, **_kwargs: (0.0, 0.8))
    monkeypatch.setattr(core, "choose_cutoff_soc", lambda *_args, **_kwargs: (0.5, "ok"))
    monkeypatch.setattr(core, "plan_charge_power", lambda *_args, **_kwargs: (None, 0.0, "ok", 0.5))
    monkeypatch.setattr(core, "simulate_expensive_hours_detailed", lambda pv, *_args, **_kwargs: (pd.DataFrame(index=pv.index), 0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(
        core,
        "simulate_full_day_soc",
        lambda pv, *_args, **_kwargs: (pd.Series([0.5] * len(pv.index), index=pv.index), pd.DataFrame(index=pv.index)),
    )

    core.run_detailed_plan(
        target_date=day,
        weather=weather,
        pv_df=pv_df,
        consumption_kwh=cons,
        soc_at_22_percent=40.0,
        buffer_percent=0.0,
        max_ac_charge_power_kw=3.0,
    )

    assert captured["pv_col"] == "pv_total_decision_kwh"
