import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import planner_core as core


def _weather(wind_ms: float) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp("2026-06-20 09:00", tz="Europe/Brussels"), periods=6, freq="h")
    return pd.DataFrame(
        {
            "ghi_wm2": 800.0,
            "dni_wm2": 600.0,
            "dhi_wm2": 200.0,
            "temp_air_c": 30.0,
            "wind_speed_ms": wind_ms,
            "cloud_cover_pct": 10.0,
        },
        index=idx,
    )


def test_faiman_uses_module_height_wind_not_raw_10m(monkeypatch) -> None:
    captured: dict[str, pd.Series] = {}

    def _fake_faiman(*, poa_global, temp_air, wind_speed, u0, u1):
        captured["wind_speed"] = pd.to_numeric(wind_speed, errors="coerce")
        return pd.Series(temp_air, index=poa_global.index, dtype=float)

    monkeypatch.setattr(core.pvlib.temperature, "faiman", _fake_faiman)

    df = _weather(5.0)
    loc = core.Location("BE", 50.85, 4.35, elevation_m=100.0)
    core.build_pv_forecast(df, loc, tz="Europe/Brussels")

    expected = core.wind_speed_module_height_from_10m(df["wind_speed_ms"])
    assert "wind_speed" in captured
    pd.testing.assert_series_equal(captured["wind_speed"], expected, check_names=False)
    assert not captured["wind_speed"].equals(pd.to_numeric(df["wind_speed_ms"], errors="coerce"))


def test_more_10m_wind_increases_output_via_effective_module_wind() -> None:
    loc = core.Location("BE", 50.85, 4.35, elevation_m=100.0)
    low_wind = core.build_pv_forecast(_weather(1.0), loc, tz="Europe/Brussels")
    high_wind = core.build_pv_forecast(_weather(8.0), loc, tz="Europe/Brussels")

    assert float(high_wind["pv_total_kwh"].sum()) > float(low_wind["pv_total_kwh"].sum())
    assert low_wind.attrs["temperature_wind_input_source"] == "module_height_from_10m"
    assert float(low_wind.attrs["effective_module_wind_height_m"]) == core.PV_EFFECTIVE_MODULE_WIND_HEIGHT_M
