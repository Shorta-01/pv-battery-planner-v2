import datetime as dt

import pandas as pd
import pytest

import planner_core as core
import scoring


def _cfg():
    windows = {str(i): [] for i in range(7)}
    windows["0"] = [["22:00", "07:00"]]
    windows["6"] = [["00:00", "24:00"]]
    return {
        "offpeak_windows_by_dow": windows,
        "offpeak_grid_price_eur_per_kwh": 0.20,
        "peak_grid_price_eur_per_kwh": 0.40,
        "injection_grid_price_eur_per_kwh": 0.05,
        "night_load_from_battery": True,
    }


def test_tariff_mask_clock_based_wrap_window():
    cfg = _cfg()
    monday = dt.date(2026, 1, 5)
    idx = pd.date_range(pd.Timestamp(monday, tz=core.TIMEZONE), periods=24, freq="h")
    mask = core.get_offpeak_mask(idx, cfg)
    for h in range(24):
        expected = h in {0, 1, 2, 3, 4, 5, 6, 22, 23}
        assert bool(mask.iloc[h]) is expected


def test_charge_session_starts_at_22_and_excludes_earlier_hours():
    cfg = _cfg()
    sunday = dt.date(2026, 1, 4)
    session = core.get_charge_session_index(sunday, cfg, start_hhmm="22:00")
    assert session.min() == pd.Timestamp("2026-01-04 22:00", tz=core.TIMEZONE)
    assert len(session) == 9


def test_night_simulation_includes_load_and_grid_import():
    cfg = _cfg()
    df = core.simulate_night_charging_series(
        soc_at_22=0.30,
        charge_kw=2.5,
        cutoff_soc=0.80,
        tomorrow_date=dt.date(2026, 1, 5),
        total_consumption_kwh=24.0,
        tariff_cfg=cfg,
    )
    assert "load_kwh" in df.columns
    assert float(df["load_kwh"].sum()) > 0.0
    assert float(df["grid_import_kwh"].sum()) > 0.0


def test_savings_payload_shape():
    cfg = _cfg()
    tomorrow = dt.date(2026, 1, 5)
    idx = pd.date_range(pd.Timestamp(tomorrow, tz=core.TIMEZONE), periods=24, freq="h")
    pv_df = pd.DataFrame({"pv_total_kwh": [0.5] * 24}, index=idx)
    flows_df = pd.DataFrame({"grid_import_kwh": [0.8] * 24, "grid_export_kwh": [0.1] * 24}, index=idx)
    out = core.compute_euro_savings_no_battery_vs_plan(
        pv_df=pv_df,
        flows_df=flows_df,
        soc_at_22=0.3,
        charge_kw=2.0,
        cutoff_soc=0.7,
        today_date=tomorrow - dt.timedelta(days=1),
        tomorrow_date=tomorrow,
        total_consumption_kwh=24.0,
        tariff_cfg=cfg,
    )
    for k in [
        "baseline_cost_eur_total",
        "plan_cost_eur_total",
        "savings_eur_total",
        "baseline_cost_eur_tomorrow",
        "plan_cost_eur_tomorrow",
        "savings_eur_tomorrow",
        "hourly_savings_eur_tomorrow",
    ]:
        assert k in out
    assert len(out["hourly_savings_eur_tomorrow"]) == 24


def test_pv_quality_ratio_range_when_supported():
    if not hasattr(core, "Location"):
        pytest.skip("Location type unavailable")
    tomorrow = dt.date(2026, 1, 5)
    idx = pd.date_range(pd.Timestamp(tomorrow, tz=core.TIMEZONE), periods=24, freq="h")
    pv_df = pd.DataFrame({"pv_total_kwh": [0.3] * 24}, index=idx)
    weather_df = pd.DataFrame(index=idx)
    kwargs = {
        "pv_df": pv_df,
        "weather_df": weather_df,
        "target_date": tomorrow,
        "tz": str(core.TIMEZONE),
        "fallback_score": 55,
    }
    import inspect

    if "loc" in inspect.signature(scoring.compute_pv_quality_score).parameters:
        kwargs["loc"] = core.Location(name="Test", latitude=52.37, longitude=4.90)
    out = scoring.compute_pv_quality_score(**kwargs)
    assert "ratio" in out
    ratio = out.get("ratio")
    assert ratio is None or (0.0 <= float(ratio) <= 1.0)
