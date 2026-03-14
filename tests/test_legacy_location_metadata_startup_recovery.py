import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api
import planner_core as core


def _new_state(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(backend_api, "LOCAL_STATE_DIR", tmp_path / "local_state")
    monkeypatch.setattr(backend_api, "SETTINGS_PATH", backend_api.LOCAL_STATE_DIR / "settings.json")
    monkeypatch.setattr(backend_api, "INPUTS_PATH", backend_api.LOCAL_STATE_DIR / "last_inputs.json")
    monkeypatch.setattr(backend_api, "LATEST_RESULT_PATH", backend_api.LOCAL_STATE_DIR / "latest_result.json")
    monkeypatch.setattr(backend_api, "HISTORY_PATH", backend_api.LOCAL_STATE_DIR / "results_history.json")
    monkeypatch.setattr(backend_api, "SQLITE_PATH", backend_api.LOCAL_STATE_DIR / "planner_history.sqlite")
    monkeypatch.setattr(backend_api, "TOKEN_PATH", backend_api.LOCAL_STATE_DIR / "api_token.txt")
    monkeypatch.setattr(backend_api, "RUN_HISTORY_PATH", tmp_path / "run_history_log.json")
    monkeypatch.setattr(backend_api, "init_db", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_api, "fetch_latest_full_run", lambda *_args, **_kwargs: None)
    return backend_api.BackendState()


def test_build_effective_config_uses_merged_auto_resolve_default_for_missing_location_metadata(monkeypatch):
    user_cfg = {
        "location": {
            "latitude": 50.85,
            "longitude": 4.35,
            "timezone": "",
            "elevation_m": None,
        }
    }

    force_refresh_calls = []

    def fake_resolve_location_metadata(**kwargs):
        force_refresh_calls.append(kwargs["force_refresh"])
        assert kwargs["force_refresh"] is False
        return {"timezone": "Europe/Brussels", "elevation_m": 99.5, "warnings": []}

    monkeypatch.setattr(core, "resolve_location_metadata", fake_resolve_location_metadata)

    merged = core.build_effective_config(user_cfg)

    assert force_refresh_calls.count(False) == 1
    assert merged["location"]["auto_resolve_metadata"] is True
    assert merged["location"]["timezone"] == "Europe/Brussels"
    assert merged["location"]["elevation_m"] == 99.5


def test_backend_state_load_settings_repairs_legacy_missing_elevation(monkeypatch, tmp_path: Path):
    legacy_cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    legacy_cfg["location"]["elevation_m"] = None
    legacy_cfg["location"].pop("auto_resolve_metadata", None)

    settings_payload = {
        "config": legacy_cfg,
        "nightly_run_time": "22:00",
        "timezone": "Europe/Brussels",
        "max_ac_charge_power_kw_default": 5.0,
    }

    (tmp_path / "local_state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "local_state" / "settings.json").write_text(json.dumps(settings_payload), encoding="utf-8")

    force_refresh_calls = []

    def fake_resolve_location_metadata(**kwargs):
        force_refresh_calls.append(kwargs["force_refresh"])
        return {"timezone": "Europe/Brussels", "elevation_m": 123.0, "warnings": []}

    monkeypatch.setattr(backend_api.core, "resolve_location_metadata", fake_resolve_location_metadata)

    state = _new_state(monkeypatch, tmp_path)

    assert force_refresh_calls.count(False) == 1
    assert state.settings["config"]["location"]["elevation_m"] == 123.0
    assert state.settings["config"]["location"]["auto_resolve_metadata"] is True


def test_backend_state_load_settings_repairs_legacy_blank_timezone(monkeypatch, tmp_path: Path):
    legacy_cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    legacy_cfg["location"]["timezone"] = "   "
    legacy_cfg["location"].pop("auto_resolve_metadata", None)

    settings_payload = {
        "config": legacy_cfg,
        "nightly_run_time": "22:00",
        "timezone": "Europe/Brussels",
        "max_ac_charge_power_kw_default": 5.0,
    }

    (tmp_path / "local_state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "local_state" / "settings.json").write_text(json.dumps(settings_payload), encoding="utf-8")

    force_refresh_calls = []

    def fake_resolve_location_metadata(**kwargs):
        force_refresh_calls.append(kwargs["force_refresh"])
        return {"timezone": "Europe/Brussels", "elevation_m": 88.0, "warnings": []}

    monkeypatch.setattr(backend_api.core, "resolve_location_metadata", fake_resolve_location_metadata)

    state = _new_state(monkeypatch, tmp_path)

    assert force_refresh_calls.count(False) == 1
    assert state.settings["config"]["location"]["timezone"] == "Europe/Brussels"
    assert state.settings["config"]["location"]["auto_resolve_metadata"] is True


def test_backend_state_load_settings_keeps_valid_config_unchanged(monkeypatch, tmp_path: Path):
    valid_cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    valid_cfg["location"].update(
        {
            "latitude": 50.85,
            "longitude": 4.35,
            "timezone": "Europe/Brussels",
            "elevation_m": 42.0,
            "auto_resolve_metadata": True,
        }
    )

    settings_payload = {
        "config": valid_cfg,
        "nightly_run_time": "22:00",
        "timezone": "Europe/Brussels",
        "max_ac_charge_power_kw_default": 5.0,
    }

    (tmp_path / "local_state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "local_state" / "settings.json").write_text(json.dumps(settings_payload), encoding="utf-8")

    force_refresh_calls = []

    def fake_resolve_location_metadata(**kwargs):
        force_refresh_calls.append(kwargs["force_refresh"])
        return {
            "timezone": kwargs.get("fallback_timezone"),
            "elevation_m": kwargs.get("fallback_elevation_m"),
            "warnings": [],
        }

    monkeypatch.setattr(backend_api.core, "resolve_location_metadata", fake_resolve_location_metadata)

    state = _new_state(monkeypatch, tmp_path)

    assert force_refresh_calls.count(False) == 0
    assert state.settings["config"]["location"]["timezone"] == "Europe/Brussels"
    assert state.settings["config"]["location"]["elevation_m"] == 42.0
