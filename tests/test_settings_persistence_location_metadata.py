import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy

import backend_api


def test_settings_save_reload_preserves_resolved_location_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(backend_api, "LOCAL_STATE_DIR", tmp_path)
    monkeypatch.setattr(backend_api, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(backend_api, "INPUTS_PATH", tmp_path / "last_inputs.json")
    monkeypatch.setattr(backend_api, "LATEST_RESULT_PATH", tmp_path / "latest_result.json")
    monkeypatch.setattr(backend_api, "HISTORY_PATH", tmp_path / "results_history.json")
    monkeypatch.setattr(backend_api, "RUN_HISTORY_PATH", tmp_path / "run_history_log.json")
    monkeypatch.setattr(backend_api, "TOKEN_PATH", tmp_path / "api_token.txt")
    monkeypatch.setattr(backend_api, "SQLITE_PATH", tmp_path / "planner_history.sqlite")

    monkeypatch.setattr(
        backend_api.core,
        "resolve_location_metadata",
        lambda **kwargs: {"timezone": "Europe/Brussels", "elevation_m": 99.0, "warnings": []},
    )

    state = backend_api.BackendState()
    payload_cfg = copy.deepcopy(state.settings["config"])
    payload_cfg["location"].update({"latitude": 50.85, "longitude": 4.35, "timezone": "UTC", "elevation_m": 1.0})

    settings_payload = backend_api.SettingsPayload(config=payload_cfg, nightly_run_time="22:00", timezone="UTC", max_ac_charge_power_kw_default=5.0)
    state.update_settings(settings_payload)

    reloaded = backend_api.BackendState()
    assert reloaded.settings["config"]["location"]["timezone"] == "Europe/Brussels"
    assert reloaded.settings["config"]["location"]["elevation_m"] == 99.0
