from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api


def _new_state(monkeypatch, tmp_path: Path) -> backend_api.BackendState:
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


def test_latest_and_history_persistence_skip_identical_rewrite(monkeypatch, tmp_path: Path) -> None:
    state = _new_state(monkeypatch, tmp_path)

    latest_payload = {"target_date": "2026-01-01", "status": "ok"}
    state.latest_result = latest_payload
    state._persist_latest_result_json()

    latest_stat_before = backend_api.LATEST_RESULT_PATH.stat()
    state.latest_result = dict(latest_payload)
    state._persist_latest_result_json()
    latest_stat_after = backend_api.LATEST_RESULT_PATH.stat()

    assert latest_stat_after.st_mtime_ns == latest_stat_before.st_mtime_ns
    assert json.loads(backend_api.LATEST_RESULT_PATH.read_text(encoding="utf-8")) == latest_payload

    history_payload = [{"target_date": "2026-01-01", "status": "ok"}]
    state.history = history_payload
    state._persist_history_json()

    history_stat_before = backend_api.HISTORY_PATH.stat()
    state.history = [dict(history_payload[0])]
    state._persist_history_json()
    history_stat_after = backend_api.HISTORY_PATH.stat()

    assert history_stat_after.st_mtime_ns == history_stat_before.st_mtime_ns
    assert json.loads(backend_api.HISTORY_PATH.read_text(encoding="utf-8")) == history_payload


def test_settings_and_inputs_persistence_skip_identical_rewrite(monkeypatch, tmp_path: Path) -> None:
    state = _new_state(monkeypatch, tmp_path)

    state._save_settings()
    settings_stat_before = backend_api.SETTINGS_PATH.stat()
    state.settings = json.loads(json.dumps(state.settings))
    state._save_settings()
    settings_stat_after = backend_api.SETTINGS_PATH.stat()

    assert settings_stat_after.st_mtime_ns == settings_stat_before.st_mtime_ns

    state.last_inputs = {
        "soc_now_percent": 50.0,
        "yesterday_consumption_kwh": 10.0,
        "last_inputs_updated_at": "2026-01-01T00:00:00+00:00",
    }
    state._save_inputs()
    inputs_stat_before = backend_api.INPUTS_PATH.stat()
    state.last_inputs = json.loads(json.dumps(state.last_inputs))
    state._save_inputs()
    inputs_stat_after = backend_api.INPUTS_PATH.stat()

    assert inputs_stat_after.st_mtime_ns == inputs_stat_before.st_mtime_ns


def test_changed_payload_still_rewrites(monkeypatch, tmp_path: Path) -> None:
    state = _new_state(monkeypatch, tmp_path)

    state.latest_result = {"target_date": "2026-01-01", "status": "ok"}
    state._persist_latest_result_json()
    stat_before = backend_api.LATEST_RESULT_PATH.stat()

    state.latest_result = {"target_date": "2026-01-01", "status": "changed"}
    state._persist_latest_result_json()
    stat_after = backend_api.LATEST_RESULT_PATH.stat()

    assert stat_after.st_mtime_ns >= stat_before.st_mtime_ns
    assert json.loads(backend_api.LATEST_RESULT_PATH.read_text(encoding="utf-8"))["status"] == "changed"
