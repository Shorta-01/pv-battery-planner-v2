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


def test_battery_max_soc_survives_save_reload_and_unrelated_saves(monkeypatch, tmp_path: Path):
    state = _isolated_state(monkeypatch, tmp_path)

    cfg = copy.deepcopy(state.settings["config"])
    cfg["battery"]["battery_max_soc_percent"] = 88.0
    cfg["battery"]["max_cutoff_soc_percent"] = 88.0
    state.update_settings(
        backend_api.SettingsPayload(
            config=cfg,
            nightly_run_time="22:00",
            timezone="Europe/Brussels",
            max_ac_charge_power_kw_default=5.0,
        )
    )

    unrelated = copy.deepcopy(state.settings["config"])
    unrelated["tariff"]["peak_grid_price_eur_per_kwh"] = 0.31
    state.update_settings(
        backend_api.SettingsPayload(
            config=unrelated,
            nightly_run_time="22:00",
            timezone="Europe/Brussels",
            max_ac_charge_power_kw_default=5.0,
        )
    )

    reloaded = backend_api.BackendState()
    assert reloaded.settings["config"]["battery"]["battery_max_soc_percent"] == 88.0
