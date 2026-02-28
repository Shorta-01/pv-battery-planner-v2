import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def _fake_night_df(target_date: dt.date, soc_end_pct: float = 20.0) -> pd.DataFrame:
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


def test_store_pv_before_export_when_all_day_offpeak(monkeypatch) -> None:
    target_date = dt.date(2026, 1, 10)
    start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=core.TIMEZONE)
    idx = pd.date_range(start, start + dt.timedelta(days=1), freq="h", inclusive="left", tz=core.TIMEZONE)

    pv_df = pd.DataFrame({"pv_total_kwh": 0.0, "pv_total_unclipped_kwh": 0.0}, index=idx)
    for hour in range(10, 16):
        ts = start + dt.timedelta(hours=hour)
        pv_df.loc[ts, "pv_total_kwh"] = 4.0
        pv_df.loc[ts, "pv_total_unclipped_kwh"] = 4.0

    monkeypatch.setattr(core, "simulate_night_charging_series", lambda *args, **kwargs: _fake_night_df(target_date))
    monkeypatch.setattr(
        core,
        "build_cycle_hourly_load_series",
        lambda *args, **kwargs: pd.Series(0.0, index=idx, dtype=float),
    )

    tariff_cfg = {
        "offpeak_windows_by_dow": [[["00:00", "24:00"]]] * 7,
        "peak_grid_price_eur_per_kwh": 0.18,
        "offpeak_grid_price_eur_per_kwh": 0.14,
        "injection_grid_price_eur_per_kwh": 0.01,
        "optimization_mode": "window_only",
    }

    _, flows_df = core.simulate_full_day_soc(
        df=pv_df,
        total_consumption_kwh=0.0,
        soc_at_22=0.2,
        charge_kw=0.0,
        cutoff_soc=0.95,
        tomorrow_date=target_date,
        tariff_cfg=tariff_cfg,
    )

    charging_rows = flows_df[flows_df["batt_charge_kwh"] > 0]

    assert float(flows_df["batt_charge_kwh"].sum()) > 0.0
    assert float(charging_rows["grid_export_kwh"].sum()) < 0.5
    assert float(flows_df.attrs["pv_surplus_store_preferred_kwh"]) > float(flows_df.attrs["pv_surplus_export_preferred_kwh"])
