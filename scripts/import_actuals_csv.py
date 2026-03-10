from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

DEFAULT_API_BASE = "http://127.0.0.1:8787"
DEFAULT_SOURCE = "manual_csv"
REQUIRED_COLUMNS = (
    "ts_local",
    "pv_kwh",
    "load_kwh",
    "grid_import_kwh",
    "grid_export_kwh",
    "soc_pct",
)
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
ISO_HOURLY_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:00:00$")


def _resolve_token(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()

    env_token = os.getenv("PVBP_API_TOKEN", "").strip()
    if env_token:
        return env_token

    token_path = Path("local_state/api_token.txt")
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()

    raise RuntimeError("Missing API token. Use --token, PVBP_API_TOKEN, or local_state/api_token.txt")


def _validate_hourly_timestamp(value: str) -> bool:
    if not ISO_HOURLY_TS_RE.match(value):
        return False
    try:
        dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return False
    return True


def load_and_validate_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(REQUIRED_COLUMNS):
            raise ValueError(f"CSV headers must be exactly: {','.join(REQUIRED_COLUMNS)}")

        rows: list[dict[str, Any]] = []
        for row_num, row in enumerate(reader, start=2):
            normalized: dict[str, Any] = {}
            ts_local = str(row.get("ts_local", "")).strip()
            if not _validate_hourly_timestamp(ts_local):
                raise ValueError(f"Row {row_num}: ts_local must match YYYY-MM-DDTHH:00:00")
            normalized["ts_local"] = ts_local

            for column in NUMERIC_COLUMNS:
                raw = row.get(column)
                try:
                    normalized[column] = float(raw)
                except (TypeError, ValueError):
                    raise ValueError(f"Row {row_num}: {column} must be numeric") from None

            rows.append(normalized)

    return rows


def post_actual_rows(
    *,
    session: requests.Session,
    api_base: str,
    token: str,
    source: str,
    rows: list[dict[str, Any]],
    timeout_s: int = 60,
) -> requests.Response:
    payload = {"source": source, "rows": rows}
    return session.post(
        f"{api_base.rstrip('/')}/v1/actuals/hourly",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout_s,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Import local hourly actuals CSV into /v1/actuals/hourly")
    parser.add_argument("input_csv", help="Path to CSV with required header")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"Source tag to store (default: {DEFAULT_SOURCE})")
    parser.add_argument("--api-base", default=os.getenv("PVBP_BACKEND_URL", DEFAULT_API_BASE), help="Backend base URL")
    parser.add_argument("--token", default=None, help="Bearer token (optional if env/file present)")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        return 2

    try:
        token = _resolve_token(args.token)
        rows = load_and_validate_csv(input_path)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        with requests.Session() as session:
            resp = post_actual_rows(
                session=session,
                api_base=args.api_base,
                token=token,
                source=str(args.source),
                rows=rows,
                timeout_s=max(1, int(args.timeout)),
            )
    except requests.RequestException as exc:
        print(
            f"Import failed: rows_read={len(rows)} rows_posted=0 source={args.source} error={exc}",
            file=sys.stderr,
        )
        return 1

    if resp.status_code >= 400:
        print(
            f"Import failed: rows_read={len(rows)} rows_posted=0 source={args.source} "
            f"status={resp.status_code} body={resp.text}",
            file=sys.stderr,
        )
        return 1

    body = resp.json()
    rows_posted = int(body.get("inserted", 0)) if isinstance(body, dict) else 0
    print(f"Import succeeded: rows_read={len(rows)} rows_posted={rows_posted} source={args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
