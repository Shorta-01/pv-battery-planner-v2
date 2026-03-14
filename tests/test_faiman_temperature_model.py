import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import planner_core as core


def _weather(temp_c: float, wind_ms: float) -> pd.DataFrame:
    tz = "Europe/Brussels"
    idx = pd.date_range(pd.Timestamp("2026-06-20 06:00", tz=tz), periods=10, freq="h")
    return pd.DataFrame(
        {
            "ghi_wm2": 700.0,
            "dni_wm2": 500.0,
            "dhi_wm2": 200.0,
            "temp_air_c": temp_c,
            "wind_speed_ms": wind_ms,
            "cloud_cover_pct": 20.0,
        },
        index=idx,
    )


def test_faiman_is_used_and_parameters_are_explicit(monkeypatch) -> None:
    called = {"faiman": 0, "sapm": 0, "u0": None, "u1": None}

    def _fake_faiman(*, poa_global, temp_air, wind_speed, u0, u1):
        called["faiman"] += 1
        called["u0"] = u0
        called["u1"] = u1
        return pd.Series(temp_air, index=poa_global.index, dtype=float) + 25.0

    def _fake_sapm_cell(*args, **kwargs):
        called["sapm"] += 1
        raise AssertionError("SAPM path should not be used")

    monkeypatch.setattr(core.pvlib.temperature, "faiman", _fake_faiman)
    monkeypatch.setattr(core.pvlib.temperature, "sapm_cell", _fake_sapm_cell)

    loc = core.Location("BE", 50.85, 4.35, elevation_m=100.0)
    out = core.build_pv_forecast(_weather(20.0, 2.0), loc, tz="Europe/Brussels")
    assert float(out["pv_total_kwh"].sum()) >= 0.0
    assert called["faiman"] > 0
    assert called["sapm"] == 0
    assert called["u0"] == core.PV_TEMPERATURE_FAIMAN_U0
    assert called["u1"] == core.PV_TEMPERATURE_FAIMAN_U1


def test_hotter_air_reduces_output_and_more_wind_increases_output() -> None:
    loc = core.Location("BE", 50.85, 4.35, elevation_m=100.0)
    cool = core.build_pv_forecast(_weather(10.0, 1.0), loc, tz="Europe/Brussels")
    hot = core.build_pv_forecast(_weather(35.0, 1.0), loc, tz="Europe/Brussels")
    windy = core.build_pv_forecast(_weather(35.0, 6.0), loc, tz="Europe/Brussels")
    assert float(hot["pv_total_kwh"].sum()) < float(cool["pv_total_kwh"].sum())
    assert float(windy["pv_total_kwh"].sum()) > float(hot["pv_total_kwh"].sum())
