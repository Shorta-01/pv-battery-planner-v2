import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api


class _FakeBmwService:
    def __init__(self, vehicles_seq, refresh_result=None):
        self._vehicles_seq = list(vehicles_seq)
        self._refresh_result = refresh_result if refresh_result is not None else {"ok": True}
        self.manual_refresh_calls = 0

    def vehicles(self):
        if self._vehicles_seq:
            return self._vehicles_seq.pop(0)
        return {}

    def manual_refresh(self, *, force_reprobe=False):
        _ = force_reprobe
        self.manual_refresh_calls += 1
        return self._refresh_result


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


def _base_vehicle(*, soc=61.0, freshness=120, status="fresh"):
    return {
        "vehicle-1": {
            "soc_pct": soc,
            "freshness_seconds": freshness,
            "data_status": status,
            "last_update_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    }


def test_ev_disabled_skips_refresh(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.settings["config"]["ev_vehicle_data"] = {"enabled": False}
    fake = _FakeBmwService([_base_vehicle()])
    state.bmw_service = fake

    out = state._get_planning_ready_ev_state()

    assert out["refresh_attempted"] is False
    assert out["refresh_reason"] == "ev_disabled"
    assert fake.manual_refresh_calls == 0


def test_fresh_cache_still_refreshes_before_planning(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.settings["config"]["ev_vehicle_data"] = {"enabled": True, "bmw_healthcheck_seconds": 300}
    fake = _FakeBmwService([_base_vehicle(soc=54.0, freshness=120), _base_vehicle(soc=54.0, freshness=10)])
    state.bmw_service = fake

    out = state._get_planning_ready_ev_state()

    assert out["refresh_attempted"] is True
    assert out["refresh_reason"] == "run_forecast"
    assert out["refresh_succeeded"] is True
    assert out["vehicles"]["vehicle-1"]["soc_pct"] == 54.0
    assert fake.manual_refresh_calls == 1


def test_stale_cache_refreshes_and_uses_refreshed_state(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.settings["config"]["ev_vehicle_data"] = {"enabled": True, "bmw_healthcheck_seconds": 300}
    stale = _base_vehicle(soc=20.0, freshness=900, status="stale")
    fresh = _base_vehicle(soc=73.0, freshness=20, status="fresh")
    fake = _FakeBmwService([stale, fresh], refresh_result={"ok": True})
    state.bmw_service = fake

    out = state._get_planning_ready_ev_state()

    assert out["refresh_attempted"] is True
    assert out["refresh_reason"] == "run_forecast"
    assert out["refresh_succeeded"] is True
    assert out["vehicles"]["vehicle-1"]["soc_pct"] == 73.0
    assert "live refresh succeeded" in out["warning"]


def test_no_cache_refresh_fails_with_warning(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.settings["config"]["ev_vehicle_data"] = {"enabled": True}
    fake = _FakeBmwService([{}, {}], refresh_result={"ok": False, "reason": "auth_required"})
    state.bmw_service = fake

    out = state._get_planning_ready_ev_state()

    assert out["refresh_attempted"] is True
    assert out["vehicles"] == {}
    assert "no BMW vehicle data is available" in out["warning"]
    assert "auth_required" in out["warning"]


def test_stale_cache_refresh_fails_keeps_last_known_vehicle(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.settings["config"]["ev_vehicle_data"] = {"enabled": True, "bmw_healthcheck_seconds": 300}
    stale = _base_vehicle(soc=31.0, freshness=1200, status="stale")
    fake = _FakeBmwService([stale, stale], refresh_result={"ok": False, "reason": "poll_failed"})
    state.bmw_service = fake

    out = state._get_planning_ready_ev_state()

    assert out["refresh_attempted"] is True
    assert out["vehicles"]["vehicle-1"]["soc_pct"] == 31.0
    assert "using last known EV state" in out["warning"]


def test_default_threshold_falls_back_to_300(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.settings["config"]["ev_vehicle_data"] = {"enabled": True, "bmw_healthcheck_seconds": ""}
    fake = _FakeBmwService([_base_vehicle(freshness=350), _base_vehicle(freshness=10)], refresh_result={"ok": True})
    state.bmw_service = fake

    out = state._get_planning_ready_ev_state()

    assert out["threshold_seconds"] == 300
    assert out["refresh_attempted"] is True


def test_run_now_and_nightly_both_call_run(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        return {"warnings": [], "inputs_used": {}, "metrics": {}, "target_date": args[0].isoformat()}

    monkeypatch.setattr(state, "_run", fake_run)
    state.last_inputs = {"soc_now_percent": 55.0, "yesterday_consumption_kwh": 10.0, "last_inputs_updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}

    out_now = state.run_now(backend_api.RunNowPayload(yesterday_consumption_kwh=10.0))
    out_nightly = state.run_nightly_tick(backend_api.NightlyTickPayload(force=True))

    assert out_now["ran"] is True
    assert out_nightly["ran"] is True
    assert calls["count"] == 2


def test_missing_soc_refreshes_before_planning(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.settings["config"]["ev_vehicle_data"] = {"enabled": True, "bmw_healthcheck_seconds": 300}
    missing_soc = _base_vehicle(soc=None, freshness=20, status="fresh")
    refreshed = _base_vehicle(soc=66.0, freshness=10, status="fresh")
    fake = _FakeBmwService([missing_soc, refreshed], refresh_result={"ok": True})
    state.bmw_service = fake

    out = state._get_planning_ready_ev_state()

    assert out["refresh_attempted"] is True
    assert out["refresh_reason"] == "run_forecast"
    assert out["vehicles"]["vehicle-1"]["soc_pct"] == 66.0
