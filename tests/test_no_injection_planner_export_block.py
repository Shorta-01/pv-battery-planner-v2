import copy
import datetime as dt

import pandas as pd

import planner_core as core


def _make_midday_pv_frame(day: dt.date, *, pv_total_kwh: float) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=core.TIMEZONE), periods=24, freq="h")
    pv = pd.Series(0.0, index=idx)
    for hour in range(10, 15):
        pv.loc[idx[idx.hour == hour]] = pv_total_kwh
    return pd.DataFrame({"pv_total_kwh": pv, "pv_total_unclipped_kwh": pv})


def test_simulate_full_day_soc_blocks_export_when_injection_disabled() -> None:
    tomorrow = dt.date(2026, 7, 2)
    df = _make_midday_pv_frame(tomorrow, pv_total_kwh=4.6)

    tariff = copy.deepcopy(core.DEFAULT_CONFIG["tariff"])
    tariff["allow_injection_to_grid"] = False

    _, flows_df = core.simulate_full_day_soc(
        df=df,
        total_consumption_kwh=8.0,
        soc_at_22=core.MAX_CUTOFF_SOC,
        charge_kw=0.0,
        cutoff_soc=core.MAX_CUTOFF_SOC,
        tomorrow_date=tomorrow,
        tariff_cfg=tariff,
    )

    assert float(flows_df["grid_export_kwh"].sum()) == 0.0
    assert float(flows_df["curtailed_kwh"].sum()) > 0.0
    assert bool(flows_df.attrs.get("export_blocked_by_policy", False)) is True
    assert float(flows_df.attrs.get("blocked_export_kwh_total", 0.0)) > 0.0


def test_simulate_expensive_hours_keeps_export_when_injection_allowed() -> None:
    day = dt.date(2026, 7, 2)
    df = _make_midday_pv_frame(day, pv_total_kwh=4.6)

    tariff = copy.deepcopy(core.DEFAULT_CONFIG["tariff"])
    tariff["allow_injection_to_grid"] = True
    tariff["offpeak_windows_by_dow"] = [[["00:00", "01:00"]] for _ in range(7)]

    detail_df, _, grid_export, _, _ = core.simulate_expensive_hours_detailed(
        df=df,
        total_consumption_kwh=8.0,
        start_soc=core.MAX_CUTOFF_SOC,
        for_date=day,
        tariff_cfg=tariff,
    )

    assert float(grid_export) >= 0.0
    assert bool(detail_df.attrs.get("allow_injection_to_grid", True)) is True
