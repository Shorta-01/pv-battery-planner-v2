import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api


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


def test_run_now_records_response_payload_size(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)

    def fake_run(*_args, **_kwargs):
        return {"run_id": "r1", "status": "ok", "run_duration_ms": 1, "run_diagnostics": {"stage_timings_ms": {}, "total_run_ms": 1.0}}

    monkeypatch.setattr(state, "_run", fake_run)

    out = state.run_now(backend_api.RunNowPayload())
    assert out["ran"] is True
    payload_sizes = state.latest_diagnostics.get("payload_sizes_bytes", {})
    assert isinstance(payload_sizes.get("/v1/run/now"), int)
    assert payload_sizes["/v1/run/now"] >= 0


def test_results_endpoints_record_payload_sizes(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_api, "state", state)
    monkeypatch.setattr(backend_api, "_require_token", lambda *_args, **_kwargs: None)

    latest_payload = {"run_id": "r2", "status": "ok"}
    monkeypatch.setattr(backend_api, "fetch_latest_full_run", lambda *_args, **_kwargs: latest_payload)
    monkeypatch.setattr(backend_api, "fetch_history_latest_per_day", lambda *_args, **_kwargs: [{"target_date": "2026-01-01"}])

    out_latest = backend_api.latest_result()
    out_history = backend_api.history(days=5, show_all_runs=False)

    assert out_latest == latest_payload
    assert "items" in out_history

    payload_sizes = state.latest_diagnostics.get("payload_sizes_bytes", {})
    assert isinstance(payload_sizes.get("/v1/results/latest"), int)
    assert isinstance(payload_sizes.get("/v1/results/history"), int)
    assert payload_sizes["/v1/results/latest"] >= 0
    assert payload_sizes["/v1/results/history"] >= 0
