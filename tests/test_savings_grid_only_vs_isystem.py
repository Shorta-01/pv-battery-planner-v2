import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


TZ = core.TIMEZONE


def _mk_idx(start: str, hours: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start, tz=TZ), periods=hours, freq="h")


def _tariff_cfg() -> dict:
    return {
        "peak_grid_price_eur_per_kwh": 0.30,
        "offpeak_grid_price_eur_per_kwh": 0.20,
        "injection_grid_price_eur_per_kwh": 0.05,
        "offpeak_windows_by_dow": [
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["00:00", "24:00"]],
            [["00:00", "24:00"]],
        ],
    }


def test_grid_only_costs_are_not_below_pv_baseline(monkeypatch) -> None:
    tomorrow_date = dt.date(2026, 1, 6)
    cycle_idx = _mk_idx("2026-01-05 22:00", 24)
    tomorrow_idx = _mk_idx("2026-01-06 00:00", 24)

    pv_df = pd.DataFrame({"pv_total_kwh": [0.5] * 24}, index=cycle_idx)
    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": [0.0] * 24,
            "grid_export_kwh": [0.0] * 24,
        },
        index=tomorrow_idx,
    )

    monkeypatch.setattr(core, "load_kwh_at", lambda _ts, _tot, dt_h=1.0: 1.0 * float(dt_h))

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.5,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=24.0,
        tariff_cfg=_tariff_cfg(),
    )

    assert out["grid_only_cost_eur_tomorrow"] >= out["baseline_cost_eur_tomorrow"]
    assert out["grid_only_cost_eur_cycle"] >= out["baseline_cost_eur_cycle"]


def test_benefit_vs_grid_only_consistency() -> None:
    tomorrow_date = dt.date(2026, 1, 6)
    cycle_idx = _mk_idx("2026-01-05 22:00", 24)
    tomorrow_idx = _mk_idx("2026-01-06 00:00", 24)

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * 24}, index=cycle_idx)
    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": [0.2] * 24,
            "grid_export_kwh": [0.0] * 24,
        },
        index=tomorrow_idx,
    )

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.5,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=24.0,
        tariff_cfg=_tariff_cfg(),
    )

    assert out["benefit_vs_grid_only_eur_cycle"] == pytest.approx(
        out["grid_only_cost_eur_cycle"] - out["isystem_cost_eur_cycle"]
    )
    assert len(out["hourly_benefit_vs_grid_only_eur_tomorrow_cash"]) == 24
    assert len(out["hourly_benefit_vs_grid_only_eur_cycle_cash"]) == 24
