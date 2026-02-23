import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


TZ = core.TIMEZONE


def _tariff_window_only(night_load_from_battery: bool = True) -> dict:
    return {
        "peak_grid_price_eur_per_kwh": 0.30,
        "offpeak_grid_price_eur_per_kwh": 0.20,
        "injection_grid_price_eur_per_kwh": 0.05,
        "optimization_mode": "window_only",
        "night_load_from_battery": night_load_from_battery,
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


def test_window_only_forces_grid_first_offpeak_even_if_legacy_flag_true() -> None:
    tariff_cfg = _tariff_window_only(night_load_from_battery=True)
    tomorrow_date = dt.date(2026, 1, 6)
    session_start, session_end = core.compute_charging_window_for_target_date(tomorrow_date, tariff_cfg)

    night_df = core.simulate_night_charging_series(
        soc_at_22=0.7,
        charge_kw=0.0,
        cutoff_soc=0.9,
        session_start=session_start,
        session_end=session_end,
        total_consumption_kwh=24.0,
        tariff_cfg=tariff_cfg,
    )

    assert float(night_df["batt_discharge_kwh"].sum()) == 0.0


def test_savings_cycle_returns_debug_fields_and_valid_window() -> None:
    tomorrow_date = dt.date(2026, 1, 6)
    cycle_start = pd.Timestamp("2026-01-05 22:00", tz=TZ)
    cycle_idx = pd.date_range(cycle_start, periods=24, freq="h")
    tomorrow_idx = pd.date_range(pd.Timestamp("2026-01-06 00:00", tz=TZ), periods=24, freq="h")

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(cycle_idx)}, index=cycle_idx)
    flows_df = pd.DataFrame({"grid_import_kwh": [1.0] * 24, "grid_export_kwh": [0.0] * 24}, index=tomorrow_idx)

    result = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.05,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=24.0,
        tariff_cfg=_tariff_window_only(night_load_from_battery=True),
    )

    assert "savings_cycle_start_soc_percent_used" in result
    assert "savings_night_load_from_battery_used" in result
    assert "savings_cycle_window_start_local" in result
    assert "savings_cycle_window_end_local" in result
    assert result["savings_cycle_start_soc_percent_used"] == 5.0
    assert result["savings_night_load_from_battery_used"] is False
    assert result["savings_cycle_window_start_local"].startswith("2026-01-05T22:00:00")
    assert result["savings_cycle_window_end_local"].startswith("2026-01-06T22:00:00")


def test_hourly_savings_remains_tomorrow_24_values() -> None:
    tomorrow_date = dt.date(2026, 1, 6)
    cycle_idx = pd.date_range(pd.Timestamp("2026-01-05 22:00", tz=TZ), periods=24, freq="h")
    tomorrow_idx = pd.date_range(pd.Timestamp("2026-01-06 00:00", tz=TZ), periods=24, freq="h")
    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * 24}, index=cycle_idx)
    flows_df = pd.DataFrame({"grid_import_kwh": [1.0] * 24, "grid_export_kwh": [0.0] * 24}, index=tomorrow_idx)

    result = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.35,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=24.0,
        tariff_cfg=_tariff_window_only(),
    )

    assert isinstance(result["hourly_savings_eur_tomorrow"], list)
    assert len(result["hourly_savings_eur_tomorrow"]) == 24
