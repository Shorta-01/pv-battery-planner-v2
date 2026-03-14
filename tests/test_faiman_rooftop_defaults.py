import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import planner_core as core


def _weather() -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp("2026-07-01 10:00", tz="Europe/Brussels"), periods=4, freq="h")
    return pd.DataFrame(
        {
            "ghi_wm2": 700.0,
            "dni_wm2": 500.0,
            "dhi_wm2": 200.0,
            "temp_air_c": 22.0,
            "wind_speed_ms": 3.0,
            "cloud_cover_pct": 20.0,
        },
        index=idx,
    )


def test_project_rooftop_faiman_defaults_are_explicit_and_used(monkeypatch) -> None:
    called = {"u0": None, "u1": None}

    def _fake_faiman(*, poa_global, temp_air, wind_speed, u0, u1):
        called["u0"] = u0
        called["u1"] = u1
        return pd.Series(temp_air, index=poa_global.index, dtype=float)

    monkeypatch.setattr(core.pvlib.temperature, "faiman", _fake_faiman)

    loc = core.Location("BE", 50.85, 4.35, elevation_m=120.0)
    out = core.build_pv_forecast(_weather(), loc, tz="Europe/Brussels")

    assert called["u0"] == core.PV_TEMPERATURE_FAIMAN_U0
    assert called["u1"] == core.PV_TEMPERATURE_FAIMAN_U1
    assert float(out.attrs["faiman_u0"]) == core.PV_TEMPERATURE_FAIMAN_U0
    assert float(out.attrs["faiman_u1"]) == core.PV_TEMPERATURE_FAIMAN_U1


def test_module_height_conversion_is_deterministic() -> None:
    s = pd.Series([0.0, 1.0, 5.0, 10.0])
    first = core.wind_speed_module_height_from_10m(s)
    second = core.wind_speed_module_height_from_10m(s)
    pd.testing.assert_series_equal(first, second)
