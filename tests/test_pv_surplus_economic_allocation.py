import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
from ui_utils import resolve_pv_outlook_savings


def _fake_night_df(target_date: dt.date, soc_end_pct: float = 50.0) -> pd.DataFrame:
    start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=core.TIMEZONE)
    end = start + dt.timedelta(hours=6)
    idx = pd.date_range(start, end, freq="h", inclusive="left", tz=core.TIMEZONE)
    return pd.DataFrame(
        {
            "grid_import_kwh": 0.0,
            "grid_export_kwh": 0.0,
            "batt_charge_kwh": 0.0,
            "batt_discharge_kwh": 0.0,
            "soc_end_pct": soc_end_pct,
        },
        index=idx,
    )


def _run_surplus_case(monkeypatch, *, injection: float, peak: float, offpeak: float) -> pd.DataFrame:
    target_date = dt.date(2026, 1, 6)
    start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=core.TIMEZONE)
    idx = pd.date_range(start, start + dt.timedelta(days=1), freq="h", inclusive="left", tz=core.TIMEZONE)
    pv = pd.DataFrame({"pv_total_kwh": 0.0, "pv_total_unclipped_kwh": 0.0}, index=idx)
    pv.loc[start + dt.timedelta(hours=12), "pv_total_kwh"] = 2.0
    pv.loc[start + dt.timedelta(hours=12), "pv_total_unclipped_kwh"] = 2.0

    monkeypatch.setattr(core, "simulate_night_charging_series", lambda *args, **kwargs: _fake_night_df(target_date))
    monkeypatch.setattr(
        core,
        "build_cycle_hourly_load_series",
        lambda *args, **kwargs: pd.Series(0.0, index=idx, dtype=float),
    )

    tariff_cfg = {
        "offpeak_windows_by_dow": [[("00:00", "06:00")]] * 7,
        "peak_grid_price_eur_per_kwh": peak,
        "offpeak_grid_price_eur_per_kwh": offpeak,
        "injection_grid_price_eur_per_kwh": injection,
        "optimization_mode": "window_only",
    }

    _, flows_df = core.simulate_full_day_soc(
        df=pv,
        total_consumption_kwh=0.0,
        soc_at_22=0.5,
        charge_kw=0.0,
        cutoff_soc=0.8,
        tomorrow_date=target_date,
        tariff_cfg=tariff_cfg,
    )
    return flows_df


def test_export_preferred_when_injection_value_exceeds_stored_value(monkeypatch) -> None:
    flows_df = _run_surplus_case(monkeypatch, injection=0.40, peak=0.20, offpeak=0.10)

    assert flows_df.attrs["pv_surplus_store_econ_enabled"] is True
    assert flows_df.attrs["pv_store_vs_export_decisions_count"] > 0
    assert flows_df.attrs["pv_surplus_export_preferred_kwh"] > flows_df.attrs["pv_surplus_store_preferred_kwh"]
    assert float(flows_df["grid_export_kwh"].sum()) > 0.0


def test_store_preferred_when_peak_displacement_value_is_higher(monkeypatch) -> None:
    flows_df = _run_surplus_case(monkeypatch, injection=0.05, peak=0.40, offpeak=0.10)

    assert flows_df.attrs["pv_surplus_store_econ_enabled"] is True
    assert flows_df.attrs["pv_store_vs_export_decisions_count"] > 0
    assert flows_df.attrs["pv_surplus_store_preferred_kwh"] > flows_df.attrs["pv_surplus_export_preferred_kwh"]
    assert float(flows_df["batt_charge_kwh"].sum()) > 0.0


def test_resolve_pv_outlook_savings_cycle_labels_remain_truthful() -> None:
    out = resolve_pv_outlook_savings(
        {
            "baseline_cost_eur_cycle": 10.0,
            "plan_cost_eur_cycle": 7.5,
            "savings_eur_cycle": 2.5,
            "hourly_savings_eur_tomorrow": [0.1] * 24,
        }
    )

    assert out["display_scope"] == "cycle"
    assert out["bars_scope"] == "tomorrow"
    assert "Cycle savings shown" in out["note"]
    assert out["detail_note"] == "⏱️ Bars: tomorrow (00–24)"
