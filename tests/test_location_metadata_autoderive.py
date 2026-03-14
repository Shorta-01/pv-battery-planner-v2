import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import planner_core as core


def test_resolve_location_metadata_uses_coordinates(monkeypatch):
    def fake_request(*, service, url, params, **kwargs):
        if service == "open-meteo-timezone":
            return {"timezone": "Europe/Brussels"}
        if service == "open-meteo-elevation":
            return {"elevation": [118.0]}
        raise AssertionError(service)

    monkeypatch.setattr(core, "_request_json", fake_request)
    res = core.resolve_location_metadata(
        latitude=50.85,
        longitude=4.35,
        fallback_timezone="UTC",
        fallback_elevation_m=0.0,
        force_refresh=True,
    )

    assert res["timezone"] == "Europe/Brussels"
    assert res["elevation_m"] == 118.0


def test_effective_config_contains_resolved_location_metadata(monkeypatch):
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["location"].update({"latitude": 50.85, "longitude": 4.35, "timezone": "", "elevation_m": None, "auto_resolve_metadata": True})

    def fake_request(*, service, **kwargs):
        if service == "open-meteo-timezone":
            return {"timezone": "Europe/Brussels"}
        if service == "open-meteo-elevation":
            return {"elevation": 123.0}
        raise AssertionError(service)

    monkeypatch.setattr(core, "_request_json", fake_request)
    out = core.build_effective_config(cfg)
    assert out["location"]["timezone"] == "Europe/Brussels"
    assert out["location"]["elevation_m"] == 123.0
