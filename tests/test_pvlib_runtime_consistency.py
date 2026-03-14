import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
import weather_ensemble as we


def _sample_inputs() -> tuple[pd.DataFrame, core.Location]:
    idx = pd.date_range("2026-01-10", periods=2, freq="h", tz="Europe/Brussels")
    df = pd.DataFrame(
        {
            "ghi_wm2": [0.0, 50.0],
            "dni_wm2": [0.0, 30.0],
            "dhi_wm2": [0.0, 20.0],
            "temp_air_c": [10.0, 11.0],
            "wind_speed_ms": [1.0, 1.5],
            "cloud_cover_pct": [80.0, 60.0],
        },
        index=idx,
    )
    loc = core.Location(name="test", latitude=50.85, longitude=4.35, elevation_m=50.0)
    return df, loc


@pytest.mark.parametrize(
    "entrypoint",
    [
        "build_pv_forecast",
        "estimate_pv_with_pvlib",
        "compute_solar_elevation_series",
        "validate_pvlib_runtime",
    ],
)
def test_pvlib_missing_fails_with_consistent_error(monkeypatch, entrypoint: str) -> None:
    df, loc = _sample_inputs()
    monkeypatch.setattr(core, "PVLIB_AVAILABLE", False)

    with pytest.raises(RuntimeError) as excinfo:
        if entrypoint == "build_pv_forecast":
            core.build_pv_forecast(df, loc, tz="Europe/Brussels")
        elif entrypoint == "estimate_pv_with_pvlib":
            core.estimate_pv_with_pvlib(df, loc, tz="Europe/Brussels")
        elif entrypoint == "compute_solar_elevation_series":
            core.compute_solar_elevation_series(df.index, loc)
        else:
            we.validate_pvlib_runtime(require_production_quality=True)

    assert str(excinfo.value) == core.PVLIB_REQUIRED_ERROR_MESSAGE
    assert "NameError" not in str(excinfo.value)


def test_validate_pvlib_runtime_non_production_returns_false(monkeypatch) -> None:
    monkeypatch.setattr(core, "PVLIB_AVAILABLE", False)
    assert we.validate_pvlib_runtime(require_production_quality=False) is False
