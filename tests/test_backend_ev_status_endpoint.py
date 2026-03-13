import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api


class _FakeBmwService:
    def __init__(self):
        self.last_update_config = None

    def update_config(self, config):
        self.last_update_config = dict(config)

    def provider_status(self):
        return {"provider_status": "healthy", "data_status": "fresh"}

    def vehicles(self):
        return {"veh": {"soc_pct": 71.0, "freshness_seconds": 10, "is_plugged": True, "is_charging": True, "range_km": 300.0, "charge_power_kw": None, "time_to_full_min": 35}}

    def manual_refresh(self, *, force_reprobe=False):
        _ = force_reprobe
        return {"ok": True}

    def device_flow_debug_info(self):
        return {}


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
    monkeypatch.setattr(backend_api, "insert_forecast_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_api, "fetch_latest_full_run", lambda *_args, **_kwargs: None)
    return backend_api.BackendState()


def test_resolved_ev_runtime_config_includes_charger_cap(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.settings["max_ac_charge_power_kw_default"] = 6.5
    state.settings["config"]["ev_vehicle_data"] = {"enabled": True, "bmw_client_id": "abc"}
    runtime = state._resolved_ev_runtime_config(state.settings["config"])

    assert runtime["charger_max_power_kw"] == 6.5
    assert runtime["ev_vehicle_data"]["enabled"] is True


def test_get_unified_ev_status(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.bmw_service = _FakeBmwService()
    state.settings["config"]["ev_vehicle_data"] = {"enabled": True, "bmw_healthcheck_seconds": 300}
    state.settings["config"]["car_charger"] = {"enabled": True}
    monkeypatch.setattr(backend_api, "evse_mgr", type("X", (), {"status_dict": lambda _self: {"connected": True, "is_plugged": True, "is_charging": True, "power_kw": 4.4}})())

    out = state.get_unified_ev_status()

    assert out["soc_pct"] == 71.0
    assert out["charge_power_kw"] is None
    assert out["charge_power_source"] == "unavailable"
    assert out["field_states"]["charge_power_kw"] == "waiting_for_bmw_power"
    assert out["time_to_full_min"] == 35
    assert out["expected_full_charge_source"] == "bmw_time_to_full"
    assert out["sources"]["deadline_time"] == "config"
