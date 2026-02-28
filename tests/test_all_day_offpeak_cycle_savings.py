import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_all_day_offpeak_cycle_savings_includes_tomorrow_pv() -> None:
    tariff_cfg = {
        "peak_grid_price_eur_per_kwh": 0.18,
        "offpeak_grid_price_eur_per_kwh": 0.14,
        "injection_grid_price_eur_per_kwh": 0.01,
        "offpeak_windows_by_dow": [
            [["00:00", "24:00"]],
            [["00:00", "24:00"]],
            [["00:00", "24:00"]],
            [["00:00", "24:00"]],
            [["00:00", "24:00"]],
            [["00:00", "24:00"]],
            [["00:00", "24:00"]],
        ],
    }

    tomorrow_date = dt.date(2026, 1, 10)
    today_date = tomorrow_date - dt.timedelta(days=1)
    total_consumption_kwh = 35.0

    tomorrow_start = pd.Timestamp(dt.datetime.combine(tomorrow_date, dt.time(0, 0)), tz=core.TIMEZONE)
    idx_tomorrow = pd.date_range(tomorrow_start, tomorrow_start + dt.timedelta(days=1), freq="h", inclusive="left")

    load_tom = core.build_hourly_load_series(idx_tomorrow, total_consumption_kwh)
    pv_hourly_kwh = 18.4 / 24.0
    pv_tom = pd.Series([pv_hourly_kwh] * len(idx_tomorrow), index=idx_tomorrow, dtype=float)

    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": (load_tom - pv_tom).clip(lower=0.0),
            "grid_export_kwh": (pv_tom - load_tom).clip(lower=0.0),
            "soc_end_pct": [10.0] * len(idx_tomorrow),
        },
        index=idx_tomorrow,
    )
    pv_df = pd.DataFrame({"pv_total_kwh": pv_tom}, index=idx_tomorrow)

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.10,
        charge_kw=0.0,
        cutoff_soc=0.90,
        today_date=today_date,
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=total_consumption_kwh,
        tariff_cfg=tariff_cfg,
    )

    assert out["grid_only_cost_eur_cycle"] == pytest.approx(35.0 * 0.14, abs=1e-6)
    assert out["isystem_cost_eur_cycle"] < out["grid_only_cost_eur_cycle"]
    assert out["benefit_vs_grid_only_eur_cycle"] > 0.5
