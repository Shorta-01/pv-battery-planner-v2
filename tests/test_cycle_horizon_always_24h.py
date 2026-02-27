import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


TZ = core.TIMEZONE


def _tariff_cfg() -> dict:
    return {
        "peak_grid_price_eur_per_kwh": 0.18,
        "offpeak_grid_price_eur_per_kwh": 0.14,
        "injection_grid_price_eur_per_kwh": 0.01,
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


def _series_from_load_kwh_at(index: pd.DatetimeIndex, total_kwh: float) -> pd.Series:
    return pd.Series([core.load_kwh_at(ts, total_kwh, 1.0) for ts in index], index=index, dtype=float)


def test_cycle_horizon_weekend_transition_is_fixed_24h(monkeypatch: pytest.MonkeyPatch) -> None:
    tariff_cfg = _tariff_cfg()
    tomorrow_date = dt.date(2026, 1, 10)  # Saturday
    today_date = tomorrow_date - dt.timedelta(days=1)
    total_kwh = 35.0

    tomorrow_start = pd.Timestamp(dt.datetime.combine(tomorrow_date, dt.time(0, 0)), tz=TZ)
    idx_tomorrow = pd.date_range(tomorrow_start, tomorrow_start + dt.timedelta(days=1), freq="h", inclusive="left")

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(idx_tomorrow)}, index=idx_tomorrow)
    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": core.build_hourly_load_series(idx_tomorrow, total_kwh),
            "grid_export_kwh": [0.0] * len(idx_tomorrow),
            "soc_end_pct": [30.0] * len(idx_tomorrow),
        },
        index=idx_tomorrow,
    )

    def _mock_night(*_args, **kwargs):
        session_start = kwargs["session_start"]
        cycle_idx = pd.date_range(session_start, session_start + dt.timedelta(hours=24), freq="h", inclusive="left", tz=TZ)
        imports = _series_from_load_kwh_at(cycle_idx, total_kwh)
        return pd.DataFrame({"grid_import_kwh": imports, "grid_export_kwh": 0.0, "soc_end_pct": 30.0}, index=cycle_idx)

    monkeypatch.setattr(core, "simulate_night_charging_series", _mock_night)

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.3,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=today_date,
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=total_kwh,
        tariff_cfg=tariff_cfg,
    )

    assert out["grid_only_cost_eur_cycle"] == pytest.approx(total_kwh * 0.14, abs=1e-6)
    assert out["isystem_cost_eur_cycle"] == pytest.approx(out["grid_only_cost_eur_cycle"], abs=1e-6)
    assert out["benefit_vs_grid_only_eur_cycle"] == pytest.approx(0.0, abs=1e-6)
    assert len(out["hourly_benefit_vs_grid_only_eur_cycle_cash"]) == 24


def test_cycle_horizon_weekday_is_fixed_24h(monkeypatch: pytest.MonkeyPatch) -> None:
    tariff_cfg = _tariff_cfg()
    tomorrow_date = dt.date(2026, 1, 6)  # Tuesday
    today_date = tomorrow_date - dt.timedelta(days=1)
    total_kwh = 35.0

    tomorrow_start = pd.Timestamp(dt.datetime.combine(tomorrow_date, dt.time(0, 0)), tz=TZ)
    idx_tomorrow = pd.date_range(tomorrow_start, tomorrow_start + dt.timedelta(days=1), freq="h", inclusive="left")

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(idx_tomorrow)}, index=idx_tomorrow)
    flows_df = pd.DataFrame(
        {
            "grid_import_kwh": _series_from_load_kwh_at(idx_tomorrow, total_kwh),
            "grid_export_kwh": [0.0] * len(idx_tomorrow),
            "soc_end_pct": [35.0] * len(idx_tomorrow),
        },
        index=idx_tomorrow,
    )

    def _mock_night(*_args, **kwargs):
        session_start = kwargs["session_start"]
        cycle_idx = pd.date_range(session_start, session_start + dt.timedelta(hours=24), freq="h", inclusive="left", tz=TZ)
        imports = _series_from_load_kwh_at(cycle_idx, total_kwh)
        return pd.DataFrame({"grid_import_kwh": imports, "grid_export_kwh": 0.0, "soc_end_pct": 35.0}, index=cycle_idx)

    monkeypatch.setattr(core, "simulate_night_charging_series", _mock_night)

    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.35,
        charge_kw=0.0,
        cutoff_soc=0.9,
        today_date=today_date,
        tomorrow_date=tomorrow_date,
        total_consumption_kwh=total_kwh,
        tariff_cfg=tariff_cfg,
    )

    assert (35.0 * 0.14) <= out["grid_only_cost_eur_cycle"] <= (35.0 * 0.18)
    assert out["benefit_vs_grid_only_eur_cycle"] == pytest.approx(0.0, abs=1e-6)
    assert len(out["hourly_benefit_vs_grid_only_eur_cycle_cash"]) == 24
