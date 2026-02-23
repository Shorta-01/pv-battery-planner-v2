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


def test_no_offpeak_double_count_in_plan_import() -> None:
    tomorrow_date = dt.date(2026, 1, 6)
    cycle_idx = _mk_idx("2026-01-05 22:00", 24)
    tomorrow_idx = _mk_idx("2026-01-06 00:00", 24)

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(cycle_idx)}, index=cycle_idx)
    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": [0.0] * len(tomorrow_idx),
            "grid_export_kwh": [0.0] * len(tomorrow_idx),
        },
        index=tomorrow_idx,
    )

    target_ts = pd.Timestamp("2026-01-06 01:00", tz=TZ)
    flows_df.loc[target_ts, "grid_import_kwh"] = 3.0

    tariff_cfg = {
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

    result = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.5,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=24.0,
        tariff_cfg=tariff_cfg,
    )

    # Baseline is 1 kWh import/hour with no PV. Plan uses explicit 3 kWh at 01:00.
    # If off-peak were double counted, this would become 4 kWh for that hour.
    expected_plan_tomorrow = 3.0 * 0.20
    assert result["plan_cost_eur_tomorrow"] == pytest.approx(expected_plan_tomorrow)


def test_no_battery_baseline_pv_first_formula_over_cycle(monkeypatch) -> None:
    tomorrow_date = dt.date(2026, 1, 6)
    cycle_idx = _mk_idx("2026-01-05 22:00", 24)
    tomorrow_idx = _mk_idx("2026-01-06 00:00", 24)

    pv_vals = [
        0.0,
        3.0,
        0.0,
        4.0,
    ] + [0.0] * 20
    pv_df = pd.DataFrame({"pv_total_kwh": pv_vals}, index=cycle_idx)
    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": [0.0] * 24,
            "grid_export_kwh": [0.0] * 24,
        },
        index=tomorrow_idx,
    )

    monkeypatch.setattr(core, "load_kwh_at", lambda ts, total_kwh, dt_h=1.0: 2.0 * float(dt_h))

    tariff_cfg = {
        "peak_grid_price_eur_per_kwh": 1.0,
        "offpeak_grid_price_eur_per_kwh": 1.0,
        "injection_grid_price_eur_per_kwh": 0.5,
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

    result = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.5,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=24.0,
        tariff_cfg=tariff_cfg,
    )

    # Hourly baseline with load=2:
    # pv 0 => +2 cost, pv 3 => import0/export1 => -0.5, pv0 => +2, pv4 => export2 => -1, rest 20h => +2 each
    expected = 2.0 - 0.5 + 2.0 - 1.0 + 20 * 2.0
    assert result["baseline_cost_eur_cycle"] == pytest.approx(expected)


def test_cycle_horizon_and_hourly_metadata() -> None:
    tomorrow_date = dt.date(2026, 1, 6)
    cycle_idx = _mk_idx("2026-01-05 22:00", 24)
    tomorrow_idx = _mk_idx("2026-01-06 00:00", 24)

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * 24}, index=cycle_idx)
    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": [1.0] * 24,
            "grid_export_kwh": [0.0] * 24,
        },
        index=tomorrow_idx,
    )

    tariff_cfg = {
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

    result = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.5,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=24.0,
        tariff_cfg=tariff_cfg,
    )

    assert result["savings_horizon_kind"] == "offpeak_cycle"
    assert result["savings_horizon_start_iso"].startswith("2026-01-05T22:00:00")
    assert result["savings_horizon_end_iso"].startswith("2026-01-06T22:00:00")
    assert len(result["hourly_savings_eur_tomorrow"]) == 24
