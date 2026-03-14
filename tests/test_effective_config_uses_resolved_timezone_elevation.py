import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt

import planner_core as core


def test_weather_request_uses_effective_timezone_and_elevation(monkeypatch):
    captured = {}

    def fake_request(*, service, url, params, **kwargs):
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

    monkeypatch.setattr(core, "_request_json", fake_request)
    loc = core.Location("BE", 50.85, 4.35, elevation_m=222.0)
    core._fetch_weather_payload(loc, dt.date(2026, 1, 2), "Europe/Brussels")

    assert captured["timezone"] == "Europe/Brussels"
    assert captured["elevation"] == 222
