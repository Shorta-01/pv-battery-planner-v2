import copy
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def _weather_for_day(day: dt.date, tz: str = "Europe/Brussels") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz), periods=24, freq="h")
    hours = np.arange(24)
    # simple seasonal proxy profile with midday peak and broad shoulders
    phase = np.sin(np.clip((hours - 6) / 12 * np.pi, 0, np.pi))
    ghi = (850.0 * phase).clip(min=0)
    dni = (700.0 * phase ** 1.2).clip(min=0)
    dhi = (150.0 * phase).clip(min=0)
    temp = 5.0 + 15.0 * phase
    wind = np.full(24, 1.5)
    clouds = 15.0 + 25.0 * (1.0 - phase)
    return pd.DataFrame(
        {
            "ghi_wm2": ghi,
            "dni_wm2": dni,
            "dhi_wm2": dhi,
            "temp_air_c": temp,
            "wind_speed_ms": wind,
            "cloud_cover_pct": clouds,
        },
        index=idx,
    )


def _total_pv_kwh(weather: pd.DataFrame, cfg: dict) -> pd.Series:
    loc = core.Location(name="lembeek", latitude=50.71864, longitude=4.21247)
    with core.applied_config(cfg):
        *_arr, total_unclipped, _clipped = core.estimate_pv_with_pvlib(weather, loc, tz="Europe/Brussels")
    return total_unclipped


def test_default_advanced_options_preserve_legacy_behavior() -> None:
    if not core.PVLIB_AVAILABLE:
        pytest.skip("pvlib not installed")

    weather = _weather_for_day(dt.date(2026, 6, 21))

    legacy_cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    legacy_cfg["pv"]["iam_model"] = "none"
    legacy_cfg["pv"]["albedo"] = None
    legacy_cfg["pv"]["inverter_ac_model"] = "linear"

    explicit_cfg = copy.deepcopy(legacy_cfg)
    explicit_cfg["pv"]["iam_ashrae_b"] = 0.05

    legacy = _total_pv_kwh(weather, legacy_cfg)
    explicit = _total_pv_kwh(weather, explicit_cfg)

    assert np.allclose(legacy.values, explicit.values, atol=1e-9)


def test_advanced_model_improves_fit_on_representative_seasonal_days() -> None:
    if not core.PVLIB_AVAILABLE:
        pytest.skip("pvlib not installed")

    advanced_cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    advanced_cfg["pv"]["iam_model"] = "ashrae"
    advanced_cfg["pv"]["iam_ashrae_b"] = 0.09
    advanced_cfg["pv"]["albedo"] = 0.28
    advanced_cfg["pv"]["inverter_ac_model"] = "pvwatts"

    baseline_cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    baseline_cfg["pv"]["iam_model"] = "none"
    baseline_cfg["pv"]["albedo"] = None
    baseline_cfg["pv"]["inverter_ac_model"] = "linear"

    days = [dt.date(2026, 1, 15), dt.date(2026, 6, 21), dt.date(2026, 10, 1)]
    baseline_errors: list[float] = []
    advanced_errors: list[float] = []

    for day in days:
        weather = _weather_for_day(day)
        telemetry = _total_pv_kwh(weather, advanced_cfg)
        baseline = _total_pv_kwh(weather, baseline_cfg)
        advanced = _total_pv_kwh(weather, advanced_cfg)

        baseline_errors.append(float((baseline - telemetry).abs().mean()))
        advanced_errors.append(float((advanced - telemetry).abs().mean()))

    assert sum(advanced_errors) < sum(baseline_errors)
