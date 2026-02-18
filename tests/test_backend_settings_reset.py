import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api
import planner_core as core


def _new_state(monkeypatch, tmp_path):
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


def test_reset_to_repo_defaults_replaces_settings_file(monkeypatch, tmp_path):
    repo_config_path = tmp_path / "config.json"
    repo_config = {
        "location": {
            "latitude": 50.9,
            "longitude": 4.4,
            "timezone": "Europe/Brussels",
            "name": "Repo Defaults",
        },
        "battery": {
            "battery_kwh": 14.0,
        },
    }
    repo_config_path.write_text(json.dumps(repo_config), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_PATH", repo_config_path)

    state = _new_state(monkeypatch, tmp_path)
    state.update_settings(
        backend_api.SettingsPayload(
            config={
                "location": {
                    "latitude": 48.8,
                    "longitude": 2.3,
                    "timezone": "Europe/Paris",
                    "name": "Modified Settings",
                },
            },
            nightly_run_time="21:30",
            timezone="Europe/Paris",
            max_ac_charge_power_kw_default=3.2,
        )
    )

    assert state.settings["settings_source"] == backend_api.SETTINGS_SOURCE_SETTINGS_JSON

    reset_payload = state.reset_settings_to_repo_defaults()

    expected_config = core.set_user_config(repo_config)
    assert reset_payload["config"] == expected_config
    assert reset_payload["settings_source"] == backend_api.SETTINGS_SOURCE_CONFIG_DEFAULTS
    assert reset_payload["nightly_run_time"] == backend_api.DEFAULT_NIGHTLY_TIME
    assert reset_payload["max_ac_charge_power_kw_default"] == backend_api.DEFAULT_MAX_AC_CAP

    persisted = json.loads(backend_api.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert persisted == reset_payload

    reloaded = _new_state(monkeypatch, tmp_path)
    assert reloaded.settings == reset_payload
