import copy
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_apply_config_backfills_per_array_calibration_from_global() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["pv_calibration_factor"] = 1.07
    cfg["pv"].pop("pv_calibration_factor_east", None)
    cfg["pv"].pop("pv_calibration_factor_south", None)

    core.apply_config(cfg)

    assert core.PV_CALIBRATION_FACTOR_EAST == 1.07
    assert core.PV_CALIBRATION_FACTOR_SOUTH == 1.07


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
        target_date=dt.date(2026, 1, 10),
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
