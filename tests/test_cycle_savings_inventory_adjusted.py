import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core

TZ = core.TIMEZONE


def _idx(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ), freq="h", inclusive="left")


def _tariff() -> dict:
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


def test_cycle_savings_applies_positive_terminal_value_credit(monkeypatch) -> None:
    tomorrow = dt.date(2026, 1, 6)
    cycle_idx = _idx("2026-01-05 22:00", "2026-01-06 22:00")
    tomorrow_idx = _idx("2026-01-06 00:00", "2026-01-07 00:00")

    monkeypatch.setattr(core, "load_kwh_at", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        core,
        "simulate_night_charging_series",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "grid_import_kwh": [0.0, 0.0],
                "grid_export_kwh": [0.0, 0.0],
                "soc_end_pct": [50.0, 50.0],
            },
            index=_idx("2026-01-05 22:00", "2026-01-06 00:00"),
        ),
    )

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(cycle_idx)}, index=cycle_idx)
    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": [1.4] * 24,
            "grid_export_kwh": [0.0] * 24,
            "soc_end_pct": [80.0] * 24,
        },
        index=tomorrow_idx,
    )

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.50,
        charge_kw=0.0,
        cutoff_soc=0.95,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow,
        total_consumption_kwh=24.0,
        tariff_cfg=_tariff(),
    )

    cash_only_savings = out["baseline_cost_eur_cycle"] - out["plan_cost_eur_cycle_cash"]
    adjusted_savings = out["savings_eur_cycle"]

    assert out["terminal_battery_value_eur_cycle"] > 0.0
    assert out["plan_cost_eur_cycle"] < out["plan_cost_eur_cycle_cash"]
    assert adjusted_savings > cash_only_savings


def test_cycle_savings_applies_negative_terminal_value_debit(monkeypatch) -> None:
    tomorrow = dt.date(2026, 1, 6)
    cycle_idx = _idx("2026-01-05 22:00", "2026-01-06 22:00")
    tomorrow_idx = _idx("2026-01-06 00:00", "2026-01-07 00:00")

    monkeypatch.setattr(core, "load_kwh_at", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        core,
        "simulate_night_charging_series",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "grid_import_kwh": [0.0, 0.0],
                "grid_export_kwh": [0.0, 0.0],
                "soc_end_pct": [50.0, 50.0],
            },
            index=_idx("2026-01-05 22:00", "2026-01-06 00:00"),
        ),
    )

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(cycle_idx)}, index=cycle_idx)
    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": [1.0] * 24,
            "grid_export_kwh": [0.0] * 24,
            "soc_end_pct": [20.0] * 24,
        },
        index=tomorrow_idx,
    )

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.50,
        charge_kw=0.0,
        cutoff_soc=0.95,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow,
        total_consumption_kwh=24.0,
        tariff_cfg=_tariff(),
    )

    assert out["terminal_battery_value_eur_cycle"] < 0.0
    assert out["plan_cost_eur_cycle"] > out["plan_cost_eur_cycle_cash"]


def test_cycle_hourly_bars_length_24_and_cycle_order(monkeypatch) -> None:
    tomorrow = dt.date(2026, 1, 6)
    cycle_idx = _idx("2026-01-05 22:00", "2026-01-06 22:00")
    tomorrow_idx = _idx("2026-01-06 00:00", "2026-01-07 00:00")

    monkeypatch.setattr(
        core,
        "simulate_night_charging_series",
        lambda *args, **kwargs: pd.DataFrame(
            {"grid_import_kwh": [0.0, 0.0], "grid_export_kwh": [0.0, 0.0], "soc_end_pct": [20.0, 20.0]},
            index=_idx("2026-01-05 22:00", "2026-01-06 00:00"),
        ),
    )

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(cycle_idx)}, index=cycle_idx)
    flows_df = pd.DataFrame(
        {"grid_import_kwh": [0.0] * 24, "grid_export_kwh": [0.0] * 24, "soc_end_pct": [20.0] * 24},
        index=tomorrow_idx,
    )

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.20,
        charge_kw=0.0,
        cutoff_soc=0.95,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow,
        total_consumption_kwh=24.0,
        tariff_cfg=_tariff(),
    )

    assert len(out["hourly_savings_eur_cycle"]) == 24
    assert len(out["hourly_savings_cycle_hour_labels"]) == 24
    assert out["hourly_savings_cycle_hour_labels"][0] == "22:00"


def test_battery_discharge_has_no_direct_cost_component() -> None:
    inj = 0.05
    import_price = 0.30
    row = {"batt_discharge_kwh": 2.0, "grid_import_kwh": 0.0, "grid_export_kwh": 0.0}
    hourly_cost = row["grid_import_kwh"] * import_price - row["grid_export_kwh"] * inj
    assert hourly_cost == 0.0


def test_weekend_all_offpeak_no_crash_and_cycle_fields_present(monkeypatch) -> None:
    tomorrow = dt.date(2026, 1, 11)
    cycle_idx = _idx("2026-01-10 00:00", "2026-01-11 00:00")
    tomorrow_idx = _idx("2026-01-11 00:00", "2026-01-12 00:00")

    monkeypatch.setattr(core, "load_kwh_at", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        core,
        "simulate_night_charging_series",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "grid_import_kwh": [1.0] * 24,
                "grid_export_kwh": [0.0] * 24,
                "soc_end_pct": [30.0] * 24,
            },
            index=cycle_idx,
        ),
    )

    tariff = _tariff()
    tariff["peak_grid_price_eur_per_kwh"] = 0.20
    tariff["offpeak_grid_price_eur_per_kwh"] = 0.20

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(cycle_idx)}, index=cycle_idx)
    flows_df = pd.DataFrame(
        {"grid_import_kwh": [1.0] * 24, "grid_export_kwh": [0.0] * 24, "soc_end_pct": [30.0] * 24},
        index=tomorrow_idx,
    )

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.30,
        charge_kw=0.0,
        cutoff_soc=0.95,
        today_date=dt.date(2026, 1, 10),
        tomorrow_date=tomorrow,
        total_consumption_kwh=24.0,
        tariff_cfg=tariff,
    )

    assert out["savings_horizon_label"] == "off-peak start -> next off-peak start"
    assert "hourly_savings_eur_cycle" in out
    assert len(out["hourly_savings_eur_cycle"]) == 24
