import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
from ui_utils import resolve_pv_outlook_savings


def test_savings_scope_falls_back_to_tomorrow_for_missing_daylight_cycle_coverage() -> None:
    tariff_cfg = {
        "peak_grid_price_eur_per_kwh": 0.30,
        "offpeak_grid_price_eur_per_kwh": 0.20,
        "injection_grid_price_eur_per_kwh": 0.05,
        "offpeak_windows_by_dow": [
            [["08:00", "12:00"]],  # Monday has no crossing-midnight window
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["00:00", "24:00"]],  # Sunday all-day
        ],
    }

    tomorrow_date = dt.date(2026, 3, 2)  # Monday
    today_date = tomorrow_date - dt.timedelta(days=1)

    tomorrow_start = pd.Timestamp(dt.datetime.combine(tomorrow_date, dt.time(0, 0)), tz=core.TIMEZONE)
    idx_tomorrow = pd.date_range(tomorrow_start, tomorrow_start + dt.timedelta(days=1), freq="h", inclusive="left")

    load_tom = core.build_hourly_load_series(idx_tomorrow, 24.0)
    pv_tom = pd.Series(0.0, index=idx_tomorrow, dtype=float)
    pv_tom.loc[(pv_tom.index.hour >= 10) & (pv_tom.index.hour <= 14)] = 2.0

    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": (load_tom - pv_tom).clip(lower=0.0),
            "grid_export_kwh": (pv_tom - load_tom).clip(lower=0.0),
            "soc_end_pct": [30.0] * len(idx_tomorrow),
        },
        index=idx_tomorrow,
    )
    pv_df = pd.DataFrame({"pv_total_kwh": pv_tom}, index=idx_tomorrow)

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.3,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=today_date,
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=24.0,
        tariff_cfg=tariff_cfg,
    )

    assert out["savings_preferred_scope"] == "tomorrow"

    resolved = resolve_pv_outlook_savings(out)
    assert resolved["display_scope"] == "tomorrow"
    assert "Cycle savings unavailable" in resolved["note"]
