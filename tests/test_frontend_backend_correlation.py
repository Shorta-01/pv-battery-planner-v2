import backend_api


def _request_with_headers(path: str, headers: dict[str, str]):
    raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return backend_api.Request(scope)


def test_extract_ui_request_context_reads_headers():
    request = _request_with_headers(
        "/v1/run/now",
        {
            "X-UI-Correlation-Id": "ui-run-123",
            "X-UI-Action": "run_forecast",
        },
    )

    context = backend_api._extract_ui_request_context(request)

    assert context == {
        "ui_correlation_id": "ui-run-123",
        "ui_action": "run_forecast",
    }


def test_log_backend_error_event_persists_ui_correlation_context(monkeypatch, tmp_path):
    captured = {}

    def _fake_insert_error_event(_sqlite_path, **kwargs):
        captured.update(kwargs)
        return "err-1"

    monkeypatch.setattr(backend_api, "SQLITE_PATH", tmp_path / "planner_history.sqlite")
    monkeypatch.setattr(backend_api, "insert_error_event", _fake_insert_error_event)

    request = _request_with_headers(
        "/v1/run/now",
        {
            "X-UI-Correlation-Id": "ui-run-abc",
            "X-UI-Action": "run_forecast",
        },
    )

    backend_api._log_backend_error_event(
        request=request,
        exc=RuntimeError("boom"),
        error_type="exception",
        severity="error",
        title="Backend exception: POST /v1/run/now",
        extra={"foo": "bar"},
    )

    assert isinstance(captured.get("context"), dict)
    assert captured["context"]["foo"] == "bar"
    assert captured["context"]["ui_correlation_id"] == "ui-run-abc"
    assert captured["context"]["ui_action"] == "run_forecast"


def test_post_error_merges_header_correlation_without_overwriting_payload_context(monkeypatch, tmp_path):
    captured = {}

    def _fake_insert_error_event(_sqlite_path, **kwargs):
        captured.update(kwargs)
        return "err-2"

    monkeypatch.setattr(backend_api, "SQLITE_PATH", tmp_path / "planner_history.sqlite")
    monkeypatch.setattr(backend_api, "insert_error_event", _fake_insert_error_event)

    payload = backend_api.ErrorEventPayload(
        source="frontend",
        severity="error",
        error_type="network",
        where="app.py:run_forecast",
        title="Frontend error: backend unreachable",
        body="x",
        context={"ui_correlation_id": "ui-payload", "existing": "yes"},
    )

    request = _request_with_headers(
        "/v1/errors",
        {
            "X-UI-Correlation-Id": "ui-header",
            "X-UI-Action": "frontend_error_event",
        },
    )

    out = backend_api.post_error(payload=payload, request=request, authorization=f"Bearer {backend_api.state.api_token}")

    assert out == {"error_id": "err-2"}
    assert captured["context"]["ui_correlation_id"] == "ui-payload"
    assert captured["context"]["ui_action"] == "frontend_error_event"
    assert captured["context"]["existing"] == "yes"
