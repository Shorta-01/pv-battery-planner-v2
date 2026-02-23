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


def test_double_count_regression_not_present(monkeypatch) -> None:
    tomorrow = dt.date(2026, 1, 6)
    cycle_idx = _idx("2026-01-05 22:00", "2026-01-06 22:00")
    tomorrow_idx = _idx("2026-01-06 00:00", "2026-01-07 00:00")

    monkeypatch.setattr(
        core,
        "simulate_night_charging_series",
        lambda *args, **kwargs: pd.DataFrame(
            {"grid_import_kwh": [0.0, 0.0], "grid_export_kwh": [0.0, 0.0]},
            index=_idx("2026-01-05 22:00", "2026-01-06 00:00"),
        ),
    )

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(cycle_idx)}, index=cycle_idx)
    flows_df = pd.DataFrame({"grid_import_kwh": [0.0] * 24, "grid_export_kwh": [0.0] * 24}, index=tomorrow_idx)
    flows_df.loc[pd.Timestamp("2026-01-06 01:00", tz=TZ), "grid_import_kwh"] = 3.0

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

    assert out["plan_cost_eur_tomorrow"] == pytest.approx(3.0 * 0.20)


def test_no_battery_baseline_pv_first_formula(monkeypatch) -> None:
    tomorrow = dt.date(2026, 1, 6)
    cycle_idx = _idx("2026-01-05 22:00", "2026-01-06 22:00")
    tomorrow_idx = _idx("2026-01-06 00:00", "2026-01-07 00:00")

    monkeypatch.setattr(core, "load_kwh_at", lambda ts, total_kwh, dt_h=1.0: 2.0 * float(dt_h))
    monkeypatch.setattr(
        core,
        "simulate_night_charging_series",
        lambda *args, **kwargs: pd.DataFrame(
            {"grid_import_kwh": [0.0, 0.0], "grid_export_kwh": [0.0, 0.0]},
            index=_idx("2026-01-05 22:00", "2026-01-06 00:00"),
        ),
    )

    pv_vals = [0.0, 3.0, 0.0, 4.0] + [0.0] * 20
    pv_df = pd.DataFrame({"pv_total_kwh": pv_vals}, index=cycle_idx)
    flows_df = pd.DataFrame({"grid_import_kwh": [0.0] * 24, "grid_export_kwh": [0.0] * 24}, index=tomorrow_idx)

    tariff = _tariff()
    tariff["peak_grid_price_eur_per_kwh"] = 1.0
    tariff["offpeak_grid_price_eur_per_kwh"] = 1.0
    tariff["injection_grid_price_eur_per_kwh"] = 0.5

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.20,
        charge_kw=0.0,
        cutoff_soc=0.95,
        today_date=dt.date(2026, 1, 5),
        tomorrow_date=tomorrow,
        total_consumption_kwh=24.0,
        tariff_cfg=tariff,
    )

    expected = 2.0 - 0.5 + 2.0 - 1.0 + 20 * 2.0
    assert out["baseline_cost_eur_cycle"] == pytest.approx(expected)


def test_cycle_horizon_metadata_and_hourly_length(monkeypatch) -> None:
    tomorrow = dt.date(2026, 1, 6)
    idx_cycle = _idx("2026-01-05 22:00", "2026-01-06 22:00")
    idx_tomorrow = _idx("2026-01-06 00:00", "2026-01-07 00:00")

    monkeypatch.setattr(
        core,
        "compute_charging_window_for_target_date",
        lambda d, cfg: (
            pd.Timestamp(dt.datetime.combine(d - dt.timedelta(days=1), dt.time(22, 0)), tz=TZ)
            if d == tomorrow
            else pd.Timestamp(dt.datetime.combine(d - dt.timedelta(days=1), dt.time(22, 0)), tz=TZ),
            pd.Timestamp(dt.datetime.combine(d, dt.time(7, 0)), tz=TZ),
        ),
    )
    monkeypatch.setattr(
        core,
        "simulate_night_charging_series",
        lambda *args, **kwargs: pd.DataFrame(
            {"grid_import_kwh": [0.0, 0.0], "grid_export_kwh": [0.0, 0.0]},
            index=_idx("2026-01-05 22:00", "2026-01-06 00:00"),
        ),
    )

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(idx_cycle)}, index=idx_cycle)
    flows_df = pd.DataFrame({"grid_import_kwh": [0.0] * 24, "grid_export_kwh": [0.0] * 24}, index=idx_tomorrow)

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

    assert out["savings_horizon_kind"] == "offpeak_cycle"
    assert out["savings_horizon_start_iso"]
    assert out["savings_horizon_end_iso"]
    assert len(out["hourly_savings_eur_tomorrow"]) == 24


def test_plan_export_not_overwritten_in_offpeak(monkeypatch) -> None:
    tomorrow = dt.date(2026, 1, 6)
    cycle_idx = _idx("2026-01-05 22:00", "2026-01-06 22:00")
    tomorrow_idx = _idx("2026-01-06 00:00", "2026-01-07 00:00")

    monkeypatch.setattr(core, "load_kwh_at", lambda ts, total_kwh, dt_h=1.0: 1.0 * float(dt_h))
    monkeypatch.setattr(
        core,
        "simulate_night_charging_series",
        lambda *args, **kwargs: pd.DataFrame(
            {"grid_import_kwh": [0.0, 0.0], "grid_export_kwh": [0.0, 0.0]},
            index=_idx("2026-01-05 22:00", "2026-01-06 00:00"),
        ),
    )

    pv_df = pd.DataFrame({"pv_total_kwh": [3.0, 3.0] + [0.0] * 22}, index=cycle_idx)
    flows_df = pd.DataFrame({"grid_import_kwh": [0.0] * 24, "grid_export_kwh": [0.0] * 24}, index=tomorrow_idx)
    flows_df.loc[pd.Timestamp("2026-01-06 00:00", tz=TZ), "grid_export_kwh"] = 5.0

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

    # planned export should be used directly at 00:00 (0.25€ credit), not baseline export of 2.0 kWh (0.10€ credit)
    assert out["plan_cost_eur_tomorrow"] == pytest.approx(-0.25)
