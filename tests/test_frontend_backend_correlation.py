import backend_api
import bmw_logging


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



def test_bmw_device_flow_start_propagates_ui_correlation_context(monkeypatch):
    captured = {}

    class _FakeBmwService:
        def start_device_flow(self):
            captured.update(bmw_logging.get_bmw_request_context())
            return {"ok": True}

    monkeypatch.setattr(backend_api.state, "bmw_service", _FakeBmwService())

    request = _request_with_headers(
        "/v1/ev/bmw/device_flow/start",
        {
            "X-UI-Correlation-Id": "ui-bmw-start-1",
            "X-UI-Action": "bmw_device_flow_start",
        },
    )

    out = backend_api.ev_bmw_device_flow_start(
        authorization=f"Bearer {backend_api.state.api_token}",
        request=request,
    )

    assert out == {"ok": True}
    assert captured == {
        "ui_correlation_id": "ui-bmw-start-1",
        "ui_action": "bmw_device_flow_start",
    }


def test_bmw_device_flow_poll_propagates_ui_correlation_context(monkeypatch):
    captured = {}

    class _FakeBmwService:
        def poll_device_token(self, _device_code):
            captured.update(bmw_logging.get_bmw_request_context())
            return {"ok": True, "token_status": "valid", "expires_at": None}

    monkeypatch.setattr(backend_api.state, "bmw_service", _FakeBmwService())

    request = _request_with_headers(
        "/v1/ev/bmw/device_flow/poll",
        {
            "X-UI-Correlation-Id": "ui-bmw-poll-1",
            "X-UI-Action": "bmw_device_flow_poll",
        },
    )

    out = backend_api.ev_bmw_device_flow_poll(
        payload=backend_api.BmwDeviceTokenPayload(device_code="dev-code"),
        authorization=f"Bearer {backend_api.state.api_token}",
        request=request,
    )

    assert out == {"ok": True, "token_status": "valid", "expires_at": None}
    assert captured == {
        "ui_correlation_id": "ui-bmw-poll-1",
        "ui_action": "bmw_device_flow_poll",
    }



def test_bmw_manual_refresh_propagates_ui_correlation_context(monkeypatch):
    captured = {}

    class _FakeBmwService:
        def manual_refresh(self, *, force_reprobe=False):
            _ = force_reprobe
            captured.update(bmw_logging.get_bmw_request_context())
            return {"ok": True, "refreshed": True}

    monkeypatch.setattr(backend_api.state, "bmw_service", _FakeBmwService())

    request = _request_with_headers(
        "/v1/ev/manual_refresh",
        {
            "X-UI-Correlation-Id": "ui-bmw-refresh-1",
            "X-UI-Action": "bmw_manual_refresh",
        },
    )

    out = backend_api.ev_manual_refresh(
        force_reprobe=False,
        authorization=f"Bearer {backend_api.state.api_token}",
        request=request,
    )

    assert out == {"ok": True, "refreshed": True}
    assert captured == {
        "ui_correlation_id": "ui-bmw-refresh-1",
        "ui_action": "bmw_manual_refresh",
    }
