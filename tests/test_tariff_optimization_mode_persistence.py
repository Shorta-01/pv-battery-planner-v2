import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api


def _isolated_state(monkeypatch, tmp_path: Path) -> backend_api.BackendState:
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
    return backend_api.BackendState()


def test_optimization_mode_survives_save_and_reload(monkeypatch, tmp_path: Path):
    state = _isolated_state(monkeypatch, tmp_path)
    payload_cfg = copy.deepcopy(state.settings["config"])
    payload_cfg["tariff"]["optimization_mode"] = "price_aware"

    state.update_settings(
        backend_api.SettingsPayload(
            config=payload_cfg,
            nightly_run_time="22:00",
            timezone="Europe/Brussels",
            max_ac_charge_power_kw_default=5.0,
        )
    )

    reloaded = backend_api.BackendState()
    assert reloaded.settings["config"]["tariff"]["optimization_mode"] == "price_aware"


def test_unrelated_save_does_not_reset_optimization_mode(monkeypatch, tmp_path: Path):
    state = _isolated_state(monkeypatch, tmp_path)

    first_cfg = copy.deepcopy(state.settings["config"])
    first_cfg["tariff"]["optimization_mode"] = "price_aware"
    state.update_settings(
        backend_api.SettingsPayload(
            config=first_cfg,
            nightly_run_time="22:00",
            timezone="Europe/Brussels",
            max_ac_charge_power_kw_default=5.0,
        )
    )

    second_cfg = copy.deepcopy(state.settings["config"])
    second_cfg["location"]["latitude"] = 50.85
    state.update_settings(
        backend_api.SettingsPayload(
            config=second_cfg,
            nightly_run_time="22:00",
            timezone="Europe/Brussels",
            max_ac_charge_power_kw_default=6.0,
        )
    )

    reloaded = backend_api.BackendState()
    assert reloaded.settings["config"]["tariff"]["optimization_mode"] == "price_aware"
    assert reloaded.settings["config"]["location"]["latitude"] == 50.85
