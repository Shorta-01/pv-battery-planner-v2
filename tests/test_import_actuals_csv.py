from __future__ import annotations

import importlib.util
import json
import textwrap
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "import_actuals_csv.py"
_SPEC = importlib.util.spec_from_file_location("import_actuals_csv", _MODULE_PATH)
assert _SPEC and _SPEC.loader
import_actuals_csv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(import_actuals_csv)


class DummyResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


class DummySession:
    def __init__(self, response: DummyResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post(self, url: str, headers: dict, data: str, timeout: int):
        self.calls.append({"url": url, "headers": headers, "data": data, "timeout": timeout})
        return self.response


def _write_csv(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def test_valid_csv_with_exact_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "actuals.csv"
    _write_csv(
        csv_path,
        """
        ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct
        2026-03-01T10:00:00,1.1,0.8,0.2,0.5,57
        """,
    )

    rows = import_actuals_csv.load_and_validate_csv(csv_path)

    assert rows == [
        {
            "ts_local": "2026-03-01T10:00:00",
            "pv_kwh": 1.1,
            "load_kwh": 0.8,
            "grid_import_kwh": 0.2,
            "grid_export_kwh": 0.5,
            "soc_pct": 57.0,
        }
    ]


def test_missing_required_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "actuals.csv"
    _write_csv(
        csv_path,
        """
        ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh
        2026-03-01T10:00:00,1.1,0.8,0.2,0.5
        """,
    )

    with pytest.raises(ValueError, match="CSV headers must be exactly"):
        import_actuals_csv.load_and_validate_csv(csv_path)


def test_malformed_timestamp(tmp_path: Path) -> None:
    csv_path = tmp_path / "actuals.csv"
    _write_csv(
        csv_path,
        """
        ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct
        2026-03-01 10:00:00,1.1,0.8,0.2,0.5,57
        """,
    )

    with pytest.raises(ValueError, match="ts_local must match"):
        import_actuals_csv.load_and_validate_csv(csv_path)


def test_successful_request_payload_generation() -> None:
    session = DummySession(DummyResponse(status_code=200, payload={"inserted": 2, "source": "manual_csv"}))
    rows = [
        {
            "ts_local": "2026-03-01T10:00:00",
            "pv_kwh": 1.0,
            "load_kwh": 2.0,
            "grid_import_kwh": 0.0,
            "grid_export_kwh": 0.0,
            "soc_pct": 50.0,
        }
    ]

    import_actuals_csv.post_actual_rows(
        session=session,
        api_base="http://127.0.0.1:8787",
        token="abc",
        source="ops_csv",
        rows=rows,
        timeout_s=30,
    )

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "http://127.0.0.1:8787/v1/actuals/hourly"
    assert call["headers"]["Authorization"] == "Bearer abc"
    assert call["headers"]["Content-Type"] == "application/json"
    payload = json.loads(call["data"])
    assert payload == {"source": "ops_csv", "rows": rows}


def test_non_destructive_behavior_on_bad_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "actuals.csv"
    _write_csv(
        csv_path,
        """
        ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct
        bad-ts,1.1,0.8,0.2,0.5,57
        """,
    )

    posted = {"count": 0}

    def _unexpected_post(*args, **kwargs):
        posted["count"] += 1
        raise AssertionError("POST should not be called for invalid CSV")

    monkeypatch.setattr(import_actuals_csv, "_resolve_token", lambda explicit: "abc")
    monkeypatch.setattr(import_actuals_csv, "post_actual_rows", _unexpected_post)
    monkeypatch.setattr("sys.argv", ["import_actuals_csv.py", str(csv_path), "--token", "abc"])

    rc = import_actuals_csv.main()

    assert rc == 2
    assert posted["count"] == 0
