import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api
import planner_core as core


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


def test_night_load_from_battery_is_persisted_runtime_setting(monkeypatch, tmp_path: Path):
    state = _isolated_state(monkeypatch, tmp_path)

    cfg = copy.deepcopy(state.settings["config"])
    cfg["tariff"]["optimization_mode"] = "price_aware"
    cfg["tariff"]["night_load_from_battery"] = True
    state.update_settings(
        backend_api.SettingsPayload(
            config=cfg,
            nightly_run_time="22:00",
            timezone="Europe/Brussels",
            max_ac_charge_power_kw_default=5.0,
        )
    )

    reloaded = backend_api.BackendState()
    tariff_cfg = reloaded.settings["config"]["tariff"]
    assert tariff_cfg["night_load_from_battery"] is True
    assert core.should_use_battery_for_offpeak_load(tariff_cfg) is True


def test_night_load_from_battery_is_ignored_in_window_only_mode():
    assert core.should_use_battery_for_offpeak_load(
        {"optimization_mode": "window_only", "night_load_from_battery": True}
    ) is False
