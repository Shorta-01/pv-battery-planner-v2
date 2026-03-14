import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt

import planner_core as core
import weather_ensemble as we


def test_location_elevation_is_required_in_config_validation() -> None:
    cfg = core.get_effective_config()
    cfg["location"]["elevation_m"] = None
    try:
        core.validate_config(cfg)
    except ValueError as exc:
        assert "location.elevation_m is required" in str(exc)
    else:
        raise AssertionError("Expected elevation validation failure")


def test_open_meteo_request_includes_elevation(monkeypatch) -> None:
    captured = {}

    def _fake_request(*, service, url, params, **kwargs):
        captured.update(params)
        return {
            "hourly": {
                "time": ["2026-01-02T00:00"],
                "temperature_2m": [5.0],
                "cloud_cover": [50.0],
                "shortwave_radiation": [0.0],
                "direct_normal_irradiance": [0.0],
                "diffuse_radiation": [0.0],
                "wind_speed_10m": [1.0],
            },
            "daily": {"sunrise": ["2026-01-02T08:00"], "sunset": ["2026-01-02T16:00"]},
        }

    monkeypatch.setattr(core, "_request_json", _fake_request)
    loc = core.Location("BE", 50.85, 4.35, elevation_m=123.0)
    core._fetch_weather_payload(loc, dt.date(2026, 1, 2), "Europe/Brussels")
    assert captured.get("elevation") == 123


def test_weather_cache_key_changes_with_elevation() -> None:
    day = dt.date(2026, 1, 2)
    assert we._cache_key("ecmwf_ifs", 50.85, 4.35, "Europe/Brussels", day, 10.0) != we._cache_key(
        "ecmwf_ifs", 50.85, 4.35, "Europe/Brussels", day, 200.0
    )


def test_belgium_geolocation_still_affects_solar_elevation() -> None:
    idx = core.pd.date_range("2026-06-21 12:00", periods=1, freq="h", tz="Europe/Brussels")
    brussels = core.Location("Brussels", 50.85, 4.35, elevation_m=20.0)
    arlon = core.Location("Arlon", 49.68, 5.81, elevation_m=400.0)
    elev_bxl = core.compute_solar_elevation_series(idx, brussels)
    elev_arl = core.compute_solar_elevation_series(idx, arlon)
    assert float(elev_bxl.iloc[0]) != float(elev_arl.iloc[0])
