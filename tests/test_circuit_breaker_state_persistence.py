import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_write_json_file_concurrent_writers_use_unique_tmp(tmp_path: Path) -> None:
    target = tmp_path / "circuit_breaker_state.json"
    rounds = 5

    for round_idx in range(rounds):
        exceptions: list[Exception] = []

        def write_payload(worker_idx: int) -> None:
            try:
                we._write_json_file(target, {"round": round_idx, "worker": worker_idx})
            except Exception as exc:  # pragma: no cover - assertion below validates no exceptions
                exceptions.append(exc)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_payload, idx) for idx in range(20)]
            for future in futures:
                future.result()

        assert exceptions == []
        assert target.exists()
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["round"] == round_idx
        assert isinstance(payload["worker"], int)


def test_circuit_breaker_concurrent_updates_are_thread_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_path = tmp_path / "circuit_breaker_state.json"
    monkeypatch.setattr(we, "PROVIDER_CIRCUIT_STATE_PATH", state_path)
    monkeypatch.setattr(we, "_CIRCUIT_BREAKER_STATE", {})
    monkeypatch.setattr(we, "_CIRCUIT_BREAKER_LOADED", False)

    models = ["ecmwf_ifs", "dwd_icon_d2", "gfs", "knmi_harmonie_arome"]

    def update_model(model_id: str, idx: int) -> None:
        if idx % 2 == 0:
            we._mark_provider_failure(model_id)
        else:
            we._mark_provider_success(model_id)

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(update_model, models[idx % len(models)], idx)
            for idx in range(120)
        ]
        for future in futures:
            future.result()

    assert state_path.exists()
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert isinstance(persisted, dict)
    assert any(model in persisted for model in models)


def test_persistence_failure_is_logged_and_non_fatal_for_success_mark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state_path = tmp_path / "circuit_breaker_state.json"
    monkeypatch.setattr(we, "PROVIDER_CIRCUIT_STATE_PATH", state_path)
    monkeypatch.setattr(we, "_CIRCUIT_BREAKER_STATE", {
        "ecmwf_ifs": {
            "consecutive_failures": 2.0,
            "last_failure_ts": 123.0,
            "circuit_open_until_ts": 9999999999.0,
        }
    })
    monkeypatch.setattr(we, "_CIRCUIT_BREAKER_LOADED", True)

    def boom(_: Path, __: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(we, "_write_json_file", boom)

    with caplog.at_level("WARNING"):
        we._mark_provider_success("ecmwf_ifs")

    assert any("[weather_ensemble][circuit_breaker] failed to persist success state" in rec.message for rec in caplog.records)
    assert we._CIRCUIT_BREAKER_STATE["ecmwf_ifs"]["consecutive_failures"] == 0
