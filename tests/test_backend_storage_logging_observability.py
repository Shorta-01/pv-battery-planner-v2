from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from starlette.requests import Request

import backend_api
import db_sqlite


def _request(path: str = "/v1/test") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_log_backend_error_event_logs_persist_failures(monkeypatch, caplog) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(backend_api, "insert_error_event", _boom)

    with caplog.at_level("WARNING"):
        backend_api._log_backend_error_event(
            request=_request("/v1/boom"),
            exc=ValueError("bad input"),
            error_type="validation",
            severity="warning",
            title="Backend validation error: GET /v1/boom",
            extra={"x": 1},
        )

    messages = [rec.message for rec in caplog.records]
    assert any("backend_api error_event source=backend" in msg for msg in messages)
    assert any("backend_api error_event_persist_failed" in msg for msg in messages)


def test_insert_forecast_run_logs_provider_payload_insert_failures(tmp_path: Path, monkeypatch, caplog) -> None:
    db_path = tmp_path / "planner.sqlite"
    db_sqlite.init_db(str(db_path))

    real_connect = db_sqlite._connect

    class _WrappedConn:
        def __init__(self, conn: sqlite3.Connection):
            self._conn = conn

        def executemany(self, sql, params):
            if "INSERT OR REPLACE INTO provider_payloads" in sql:
                raise RuntimeError("provider payload write failed")
            return self._conn.executemany(sql, params)

        def __getattr__(self, item):
            return getattr(self._conn, item)

    @contextlib.contextmanager
    def _wrapped_connect(path: str):
        with real_connect(path) as conn:
            yield _WrappedConn(conn)

    monkeypatch.setattr(db_sqlite, "_connect", _wrapped_connect)

    payload = {
        "run_id": "run-provider-fail-1",
        "target_date": "2026-03-01",
        "run_at_utc": "2026-02-29T23:10:00+00:00",
        "metrics": {"pv_forecast_kwh": 3.0, "cons_forecast_kwh": 10.0},
        "provider_payloads_by_model": {
            "ecmwf_ifs": {
                "fetched_at_utc": "2026-02-29T23:10:01+00:00",
                "endpoint": "https://api.open-meteo.com/v1/ecmwf",
                "params": {"latitude": 50.9},
                "response_headers": {"content-type": "application/json"},
                "response_json": {"hourly": {"time": ["2026-03-01T00:00"]}},
                "http_status": 200,
                "latency_ms": 187,
            }
        },
    }

    with caplog.at_level("ERROR"):
        db_sqlite.insert_forecast_run(str(db_path), payload)

    with sqlite3.connect(db_path) as conn:
        run_count = conn.execute("SELECT COUNT(*) FROM forecast_runs WHERE run_id = ?", ("run-provider-fail-1",)).fetchone()[0]
        provider_count = conn.execute("SELECT COUNT(*) FROM provider_payloads WHERE run_id = ?", ("run-provider-fail-1",)).fetchone()[0]

    assert run_count == 1
    assert provider_count == 0
    assert any("db_sqlite provider_payloads_persist_failed" in rec.message for rec in caplog.records)
