import copy
import datetime as dt

import pandas as pd
import pytest

import planner_core as core


def _uniform_offpeak_windows(start: str, end: str) -> list[list[list[str]]]:
    return [[[start, end]] for _ in range(7)]


def test_max_grid_import_kw_defaults_to_zero_when_missing() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["tariff"].pop("max_grid_import_kw", None)

    effective = core.build_effective_config(cfg)

    assert float(effective["tariff"].get("max_grid_import_kw", -1.0)) == 0.0


def test_max_grid_import_kw_validation_rejects_negative_and_non_numeric() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["tariff"]["max_grid_import_kw"] = -0.01
    with pytest.raises(ValueError, match="max_grid_import_kw"):
        core.build_effective_config(cfg)

    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["tariff"]["max_grid_import_kw"] = "abc"
    with pytest.raises(ValueError, match="max_grid_import_kw"):
        core.build_effective_config(cfg)


def test_night_charging_no_cap_behaves_as_before_and_not_binding() -> None:
    tomorrow = dt.date(2026, 1, 13)
    tariff = copy.deepcopy(core.DEFAULT_CONFIG["tariff"])
    tariff["offpeak_windows_by_dow"] = _uniform_offpeak_windows("22:00", "07:00")
    tariff["night_load_from_battery"] = False
    tariff["max_grid_import_kw"] = 0.0

    session_start, session_end = core.compute_charging_window_for_target_date(tomorrow, tariff)
    idx = pd.date_range(session_start, session_end, freq="h", inclusive="left", tz=core.TIMEZONE)
    loads = pd.Series(0.0, index=idx)

    night_df = core.simulate_night_charging_series(
        soc_at_22=0.10,
        charge_kw=2.0,
        cutoff_soc=core.MAX_CUTOFF_SOC,
        session_start=session_start,
        session_end=session_end,
        total_consumption_kwh=0.0,
        tariff_cfg=tariff,
        precomputed_loads=loads,
    )

    assert float(night_df["batt_charge_kwh"].sum()) > 0.0
    assert int(night_df.attrs.get("grid_import_cap_binding_events", 0)) == 0


def test_night_charging_cap_limits_charging_to_grid_headroom() -> None:
    tomorrow = dt.date(2026, 1, 13)
    tariff = copy.deepcopy(core.DEFAULT_CONFIG["tariff"])
    tariff["offpeak_windows_by_dow"] = _uniform_offpeak_windows("22:00", "07:00")
    tariff["night_load_from_battery"] = False
    tariff["max_grid_import_kw"] = 1.5

    session_start, session_end = core.compute_charging_window_for_target_date(tomorrow, tariff)
    idx = pd.date_range(session_start, session_end, freq="h", inclusive="left", tz=core.TIMEZONE)
    loads = pd.Series(0.4, index=idx)

    night_df = core.simulate_night_charging_series(
        soc_at_22=0.10,
        charge_kw=3.0,
        cutoff_soc=core.MAX_CUTOFF_SOC,
        session_start=session_start,
        session_end=session_end,
        total_consumption_kwh=0.0,
        tariff_cfg=tariff,
        precomputed_loads=loads,
    )

    cap_import_kwh = 1.5
    assert (night_df["grid_import_kwh"] <= cap_import_kwh + 1e-9).all()
    expected_charging_import = night_df["batt_charge_kwh"] / core.BATTERY_AC_CHARGE_EFF
    headroom = (cap_import_kwh - night_df["load_kwh"]).clip(lower=0.0)
    assert (expected_charging_import <= headroom + 1e-9).all()
    assert int(night_df.attrs.get("grid_import_cap_binding_events", 0)) > 0


def test_night_charging_cap_detects_load_only_exceeds_cap() -> None:
    tomorrow = dt.date(2026, 1, 13)
    tariff = copy.deepcopy(core.DEFAULT_CONFIG["tariff"])
    tariff["offpeak_windows_by_dow"] = _uniform_offpeak_windows("22:00", "07:00")
    tariff["night_load_from_battery"] = False
    tariff["max_grid_import_kw"] = 1.0

    session_start, session_end = core.compute_charging_window_for_target_date(tomorrow, tariff)
    idx = pd.date_range(session_start, session_end, freq="h", inclusive="left", tz=core.TIMEZONE)
    loads = pd.Series(1.6, index=idx)

    night_df = core.simulate_night_charging_series(
        soc_at_22=0.10,
        charge_kw=2.0,
        cutoff_soc=core.MAX_CUTOFF_SOC,
        session_start=session_start,
        session_end=session_end,
        total_consumption_kwh=0.0,
        tariff_cfg=tariff,
        precomputed_loads=loads,
    )

    assert (night_df["grid_import_kwh"] > 1.0).any()
    assert int(night_df.attrs.get("grid_import_cap_load_exceeds_events", 0)) > 0
