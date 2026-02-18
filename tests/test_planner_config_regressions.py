import copy
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_apply_config_uses_global_and_per_array_calibration_product() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["pv_calibration_factor"] = 1.07
    cfg["pv"]["pv_calibration_factor_east"] = 0.95
    cfg["pv"]["pv_calibration_factor_south"] = 1.10

    core.apply_config(cfg)

    assert core.PV_CALIBRATION_FACTOR_EAST == 1.07 * 0.95
    assert core.PV_CALIBRATION_FACTOR_SOUTH == 1.07 * 1.10


def _stub_build_pv_forecast_from_calibration(df: pd.DataFrame, _loc: core.Location, tz: str | None = None) -> pd.DataFrame:
    out = df.copy()
    east_kwh = pd.Series([1.0 * core.PV_CALIBRATION_FACTOR_EAST] * len(out.index), index=out.index)
    south_kwh = pd.Series([2.0 * core.PV_CALIBRATION_FACTOR_SOUTH] * len(out.index), index=out.index)
    total_kwh = east_kwh + south_kwh
    out["pv_east_kwh"] = east_kwh
    out["pv_south_kwh"] = south_kwh
    out["pv_total_unclipped_kwh"] = total_kwh
    out["pv_total_kwh"] = total_kwh
    out["pv_total_unclipped_kw"] = total_kwh
    out["pv_total_kw"] = total_kwh
    out["pv_dc_available_kwh"] = total_kwh
    out["pv_ac_limited_kwh"] = total_kwh
    out["pv_dc_available_kw"] = total_kwh
    out["pv_ac_limited_kw"] = total_kwh
    out["pv_clipped_kwh"] = 0.0
    out.attrs["pv_method"] = "stub"
    return out


def _pv_stub_totals_from_config(cfg: dict) -> tuple[float, float, float]:
    core.apply_config(cfg)
    idx = pd.date_range(pd.Timestamp(dt.datetime(2026, 1, 10, 0, 0), tz="Europe/Brussels"), periods=24, freq="h")
    df = pd.DataFrame(index=idx)
    pv_df = _stub_build_pv_forecast_from_calibration(df, core.Location("stub", core.LATITUDE, core.LONGITUDE), tz="Europe/Brussels")
    return (
        float(pv_df["pv_east_kwh"].sum()),
        float(pv_df["pv_south_kwh"].sum()),
        float(pv_df["pv_total_kwh"].sum()),
    )


def test_global_only_calibration_affects_pv_columns() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["pv_calibration_factor"] = 1.10
    cfg["pv"]["pv_calibration_factor_east"] = 1.00
    cfg["pv"]["pv_calibration_factor_south"] = 1.00

    pv_east_kwh, pv_south_kwh, pv_total_kwh = _pv_stub_totals_from_config(cfg)

    assert pv_east_kwh == pytest.approx(24.0 * 1.10)
    assert pv_south_kwh == pytest.approx(48.0 * 1.10)
    assert pv_total_kwh == pytest.approx(72.0 * 1.10)


def test_per_array_only_calibration_affects_pv_columns() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["pv_calibration_factor"] = 1.00
    cfg["pv"]["pv_calibration_factor_east"] = 0.90
    cfg["pv"]["pv_calibration_factor_south"] = 1.20

    pv_east_kwh, pv_south_kwh, pv_total_kwh = _pv_stub_totals_from_config(cfg)

    assert pv_east_kwh == pytest.approx(24.0 * 0.90)
    assert pv_south_kwh == pytest.approx(48.0 * 1.20)
    assert pv_total_kwh == pytest.approx((24.0 * 0.90) + (48.0 * 1.20))


def test_combined_global_and_per_array_calibration_affects_pv_columns() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["pv_calibration_factor"] = 1.10
    cfg["pv"]["pv_calibration_factor_east"] = 0.90
    cfg["pv"]["pv_calibration_factor_south"] = 1.20

    pv_east_kwh, pv_south_kwh, pv_total_kwh = _pv_stub_totals_from_config(cfg)

    assert pv_east_kwh == pytest.approx(24.0 * 1.10 * 0.90)
    assert pv_south_kwh == pytest.approx(48.0 * 1.10 * 1.20)
    assert pv_total_kwh == pytest.approx((24.0 * 1.10 * 0.90) + (48.0 * 1.10 * 1.20))


def test_parse_offpeak_windows_supports_multi_window_roundtrip() -> None:
    windows = [
        [["22:00", "07:00"], ["13:00", "14:00"]],
        [["22:00", "07:00"]],
        [["22:00", "07:00"]],
        [["22:00", "07:00"]],
        [["22:00", "07:00"]],
        [["00:00", "24:00"]],
        [["00:00", "24:00"]],
    ]
    parsed = core.parse_offpeak_windows_by_dow(windows)
    assert parsed[0][0] == ("22:00", "07:00")
    assert parsed[0][1] == ("13:00", "14:00")




def _uniform_offpeak_windows(start: str, end: str) -> list[list[list[str]]]:
    return [[[start, end]] for _ in range(7)]


@pytest.mark.parametrize(
    ("window_start", "window_end", "expected_hours"),
    [
        ("22:00", "07:00", 9),
        ("23:00", "06:00", 7),
        ("21:00", "05:00", 8),
    ],
)
def test_offpeak_window_hours_are_consistent_across_summary_planning_and_simulation(
    window_start: str,
    window_end: str,
    expected_hours: int,
) -> None:
    original_cfg = copy.deepcopy(core.EFFECTIVE_CFG)
    try:
        cfg = copy.deepcopy(core.DEFAULT_CONFIG)
        cfg["tariff"]["offpeak_windows_by_dow"] = _uniform_offpeak_windows(window_start, window_end)
        core.apply_config(cfg)

        charge_date = dt.date(2026, 1, 12)
        summary_hours, _ = core.overnight_charge_hours_summary(charge_date)
        assert int(summary_hours) == expected_hours

        _, charge_kw, _, _ = core.plan_charge_power(
            soc_start=0.10,
            soc_cutoff=core.MAX_CUTOFF_SOC,
            charge_date=charge_date,
            user_cap_kw=10.0,
        )
        assert charge_kw > 0.0

        night_df = core.simulate_night_charging_series(
            soc_at_22=0.10,
            charge_kw=1.0,
            cutoff_soc=core.MAX_CUTOFF_SOC,
            tomorrow_date=charge_date + dt.timedelta(days=1),
        )
        charged_hours = int((night_df["grid_import_kwh"] > 1e-9).sum())
        assert charged_hours == expected_hours

        idx = pd.date_range(
            pd.Timestamp(dt.datetime.combine(charge_date, dt.time(0, 0)), tz=core.TIMEZONE),
            periods=48,
            freq="h",
        )
        mask_hours = int(core.get_offpeak_mask(idx, charge_date).sum())
        assert mask_hours == expected_hours
    finally:
        core.apply_config(original_cfg)


def test_normalize_hourly_index_uses_location_timezone_consistently() -> None:
    tz = "Europe/Brussels"
    day = dt.date(2026, 10, 25)
    idx = pd.date_range(pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz), periods=25, freq="h")
    df = pd.DataFrame(
        {
            "ghi_wm2": [0.0] * len(idx),
            "dni_wm2": [0.0] * len(idx),
            "dhi_wm2": [0.0] * len(idx),
            "cloud_cover_pct": [0.0] * len(idx),
            "temp_air_c": [10.0] * len(idx),
            "wind_speed_ms": [1.0] * len(idx),
        },
        index=idx,
    )
    out = core.normalize_hourly_forecast_index(df, day, tz)
    assert len(out) == 25


def _build_stub_weather(date: dt.date, tz: str) -> core.ForecastResult:
    idx = pd.date_range(pd.Timestamp(dt.datetime.combine(date, dt.time(0, 0)), tz=tz), periods=24, freq="h")
    weather_df = pd.DataFrame(
        {
            "temp_air_c": [10.0] * len(idx),
            "ghi_wm2": [0.0] * len(idx),
            "dni_wm2": [0.0] * len(idx),
            "dhi_wm2": [0.0] * len(idx),
            "cloud_cover_pct": [20.0] * len(idx),
            "wind_speed_ms": [1.0] * len(idx),
        },
        index=idx,
    )
    sunrise = pd.Timestamp(dt.datetime.combine(date, dt.time(7, 0)), tz=tz).to_pydatetime()
    sunset = pd.Timestamp(dt.datetime.combine(date, dt.time(18, 0)), tz=tz).to_pydatetime()
    return core.ForecastResult(df=weather_df, sunrise=sunrise, sunset=sunset)


def _stub_build_pv_forecast(df: pd.DataFrame, _loc: core.Location, tz: str | None = None) -> pd.DataFrame:
    out = df.copy()
    hourly_base = float(core.PANEL_WP) / 1000.0
    pv_series = pd.Series([hourly_base] * len(out.index), index=out.index)
    out["pv_east_kwh"] = pv_series * 0.4
    out["pv_south_kwh"] = pv_series * 0.6
    out["pv_total_unclipped_kwh"] = pv_series
    out["pv_total_kwh"] = pv_series
    out["pv_total_unclipped_kw"] = pv_series
    out["pv_total_kw"] = pv_series
    out["pv_dc_available_kwh"] = pv_series
    out["pv_ac_limited_kwh"] = pv_series
    out["pv_dc_available_kw"] = pv_series
    out["pv_ac_limited_kw"] = pv_series
    out["pv_clipped_kwh"] = 0.0
    out.attrs["pv_method"] = "stub"
    return out


def _pipeline_metrics(cfg: dict) -> dict[str, float]:
    result = core.run_forecast_pipeline(
        cfg=cfg,
        target_date=dt.date(2026, 1, 12),
        soc_at_22_percent=35.0,
        yesterday_kwh=12.0,
        buffer_percent=1.0,
        user_max_ac_kw=4.0,
    )
    return {
        "pv_total_kwh": float(result.hourly_df["pv_total_kwh"].sum()),
        "charge_kw": float(result.charge_kw),
        "cutoff_soc": float(result.cutoff_soc),
        "grid_import_kwh": float(result.full_day_flows_df["grid_import_kwh"].sum()),
    }


def test_run_forecast_pipeline_isolation_across_back_to_back_configs(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setattr(core, "fetch_weather_for_date", lambda loc, target_date, tz=None: _build_stub_weather(target_date, tz or "Europe/Brussels"))
    monkeypatch.setattr(core, "build_pv_forecast", _stub_build_pv_forecast)

    cfg_a = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg_a["pv"]["panel_wp"] = 500
    cfg_a["battery"]["battery_kwh"] = 10.0
    cfg_a["battery"]["battery_max_discharge_kw"] = 2.0

    cfg_b = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg_b["pv"]["panel_wp"] = 250
    cfg_b["battery"]["battery_kwh"] = 20.0
    cfg_b["battery"]["battery_max_discharge_kw"] = 4.0

    baseline_a = _pipeline_metrics(cfg_a)
    baseline_b = _pipeline_metrics(cfg_b)

    alt_a1 = _pipeline_metrics(cfg_a)
    alt_b1 = _pipeline_metrics(cfg_b)
    alt_a2 = _pipeline_metrics(cfg_a)
    alt_b2 = _pipeline_metrics(cfg_b)

    assert alt_a1 == baseline_a
    assert alt_b1 == baseline_b
    assert alt_a2 == baseline_a
    assert alt_b2 == baseline_b


def _make_flat_pv_day(day: dt.date, tz: str = "Europe/Brussels") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz), periods=24, freq="h")
    return pd.DataFrame({"pv_total_kwh": [0.0] * len(idx)}, index=idx)


def test_compute_soc_low_window_only_is_price_invariant() -> None:
    day = dt.date(2026, 1, 12)  # Monday
    pv = _make_flat_pv_day(day)

    low_price_tariff = copy.deepcopy(core.DEFAULT_CONFIG["tariff"])
    low_price_tariff["optimization_mode"] = "window_only"
    low_price_tariff["peak_grid_price_eur_per_kwh"] = 0.30

    high_price_tariff = copy.deepcopy(low_price_tariff)
    high_price_tariff["peak_grid_price_eur_per_kwh"] = 1.20

    soc_low_a = core.compute_soc_low_timing_aware(pv, total_consumption_kwh=12.0, for_date=day, tariff_cfg=low_price_tariff)
    soc_low_b = core.compute_soc_low_timing_aware(pv, total_consumption_kwh=12.0, for_date=day, tariff_cfg=high_price_tariff)

    assert soc_low_a == pytest.approx(soc_low_b)


def test_compute_soc_low_price_aware_changes_with_prices() -> None:
    day = dt.date(2026, 1, 12)  # Monday
    pv = _make_flat_pv_day(day)

    low_price_tariff = copy.deepcopy(core.DEFAULT_CONFIG["tariff"])
    low_price_tariff["optimization_mode"] = "price_aware"
    low_price_tariff["peak_grid_price_eur_per_kwh"] = 0.30

    high_price_tariff = copy.deepcopy(low_price_tariff)
    high_price_tariff["peak_grid_price_eur_per_kwh"] = 1.20

    soc_low_a = core.compute_soc_low_timing_aware(pv, total_consumption_kwh=12.0, for_date=day, tariff_cfg=low_price_tariff)
    soc_low_b = core.compute_soc_low_timing_aware(pv, total_consumption_kwh=12.0, for_date=day, tariff_cfg=high_price_tariff)

    assert soc_low_b > soc_low_a


def test_run_forecast_pipeline_price_response_modes(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setattr(core, "fetch_weather_for_date", lambda loc, target_date, tz=None: _build_stub_weather(target_date, tz or "Europe/Brussels"))
    monkeypatch.setattr(core, "build_pv_forecast", _stub_build_pv_forecast)

    base_cfg = copy.deepcopy(core.DEFAULT_CONFIG)

    window_low = copy.deepcopy(base_cfg)
    window_low["tariff"]["optimization_mode"] = "window_only"
    window_low["tariff"]["peak_grid_price_eur_per_kwh"] = 0.30

    window_high = copy.deepcopy(window_low)
    window_high["tariff"]["peak_grid_price_eur_per_kwh"] = 1.20

    aware_low = copy.deepcopy(base_cfg)
    aware_low["tariff"]["optimization_mode"] = "price_aware"
    aware_low["tariff"]["peak_grid_price_eur_per_kwh"] = 0.30

    aware_high = copy.deepcopy(aware_low)
    aware_high["tariff"]["peak_grid_price_eur_per_kwh"] = 1.20

    def _run(cfg: dict) -> core.PlannerOutput:
        return core.run_forecast_pipeline(
            cfg=cfg,
            target_date=dt.date(2026, 1, 12),
            soc_at_22_percent=35.0,
            yesterday_kwh=28.0,
            buffer_percent=0.0,
            user_max_ac_kw=4.0,
        )

    window_low_out = _run(window_low)
    window_high_out = _run(window_high)
    aware_low_out = _run(aware_low)
    aware_high_out = _run(aware_high)

    assert float(window_low_out.cutoff_soc) == pytest.approx(float(window_high_out.cutoff_soc))
    assert float(window_low_out.charge_kw) == pytest.approx(float(window_high_out.charge_kw))

    assert float(aware_high_out.cutoff_soc) > float(aware_low_out.cutoff_soc)
    assert float(aware_high_out.charge_kw) >= float(aware_low_out.charge_kw)


def test_apply_config_plumbs_new_pv_modeling_options() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["iam_model"] = "ashrae"
    cfg["pv"]["iam_ashrae_b"] = 0.08
    cfg["pv"]["albedo"] = 0.35
    cfg["pv"]["inverter_ac_model"] = "pvwatts"

    core.apply_config(cfg)

    assert core.PV_IAM_MODEL == "ashrae"
    assert core.PV_IAM_ASHRAE_B == pytest.approx(0.08)
    assert core.PV_ALBEDO == pytest.approx(0.35)
    assert core.INVERTER_AC_MODEL == "pvwatts"


def test_validate_config_rejects_invalid_advanced_pv_options() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["iam_model"] = "bad-model"
    with pytest.raises(ValueError, match="pv.iam_model"):
        core.validate_config(cfg)

    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["albedo"] = 1.2
    with pytest.raises(ValueError, match="pv.albedo"):
        core.validate_config(cfg)

    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["inverter_ac_model"] = "unknown"
    with pytest.raises(ValueError, match="pv.inverter_ac_model"):
        core.validate_config(cfg)
