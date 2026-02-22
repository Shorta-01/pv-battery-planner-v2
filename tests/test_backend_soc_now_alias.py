import datetime as dt
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
    monkeypatch.setattr(backend_api, "insert_forecast_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_api, "fetch_latest_full_run", lambda *_args, **_kwargs: None)
    return backend_api.BackendState()


def test_run_now_soc_priority_and_clamping(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    state.last_inputs = {
        "soc_now_percent": 40.0,
        "soc_at_22_percent": 41.0,
        "yesterday_consumption_kwh": 12.0,
        "last_inputs_updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    captured = {}

    def fake_run(target_date, soc_percent, yesterday_kwh, *_args, **_kwargs):
        captured["soc_percent"] = soc_percent
        return {
            "target_date": target_date.isoformat(),
            "warnings": [],
            "inputs_used": {
                "soc_now_percent": soc_percent,
                "soc_at_22_percent": soc_percent,
                "yesterday_consumption_kwh": yesterday_kwh,
            },
            "weather_ensemble": {},
        }

    monkeypatch.setattr(state, "_run", fake_run)

    payload = backend_api.RunNowPayload(
        soc_now_percent=120.0,
        soc_at_22_percent=10.0,
        yesterday_consumption_kwh=15.0,
    )
    out = state.run_now(payload)

    assert out["ran"] is True
    assert captured["soc_percent"] == 100.0
    warnings = out["result"].get("warnings", [])
    assert any("soc_now_percent: clamped to 100.0" in w for w in warnings)


def test_update_inputs_persists_soc_now_percent(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)

    payload = backend_api.InputsPayload(soc_now_percent=37.5, yesterday_consumption_kwh=9.8)
    saved = state.update_inputs(payload)

    assert saved["soc_now_percent"] == 37.5
    assert saved["soc_at_22_percent"] == 37.5
    assert state.last_inputs["soc_now_percent"] == 37.5
