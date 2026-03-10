from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests


DEFAULT_API_BASE = "http://127.0.0.1:8787"
DEFAULT_SOURCE = "manual_csv"


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


def _default_score_date() -> str:
    return (dt.date.today() - dt.timedelta(days=1)).isoformat()


def _post_with_retry(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str],
    data: str | None = None,
    retries: int = 3,
    timeout: int = 60,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(url, headers=headers, data=data, timeout=timeout)
            if response.status_code < 500:
                return response
        except requests.RequestException as exc:
            last_exc = exc

        if attempt < retries:
            time.sleep(2)

    if last_exc is not None:
        raise RuntimeError(f"Request failed after {retries} attempts: {url} ({last_exc})") from last_exc
    raise RuntimeError(f"Request failed after {retries} attempts: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest hourly actuals CSV and compute daily score for a date.",
    )
    parser.add_argument("--actuals-csv", required=True, help="Path to CSV with required header.")
    parser.add_argument("--date", default=_default_score_date(), help="Score date YYYY-MM-DD (default: yesterday).")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"Actuals source tag (default: {DEFAULT_SOURCE}).")
    parser.add_argument("--api-base", default=os.getenv("PVBP_BACKEND_URL", DEFAULT_API_BASE), help="Backend base URL.")
    parser.add_argument("--token", default=None, help="Bearer token (optional if env/file present).")
    parser.add_argument("--retries", type=int, default=3, help="Retries for each API call.")
    args = parser.parse_args()

    actuals_path = Path(args.actuals_csv)
    if not actuals_path.exists():
        print(f"CSV file not found: {actuals_path}", file=sys.stderr)
        return 2

    try:
        dt.date.fromisoformat(args.date)
    except ValueError:
        print("--date must be YYYY-MM-DD", file=sys.stderr)
        return 2

    try:
        token = _resolve_token(args.token)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    csv_text = actuals_path.read_text(encoding="utf-8")

    auth_headers = {"Authorization": f"Bearer {token}"}

    with requests.Session() as session:
        ingest_resp = _post_with_retry(
            session,
            f"{args.api_base}/v1/actuals/hourly",
            headers={**auth_headers, "Content-Type": "text/csv"},
            data=csv_text,
            retries=max(1, int(args.retries)),
        )
        if ingest_resp.status_code >= 400:
            print(f"Actuals ingest failed ({ingest_resp.status_code}): {ingest_resp.text}", file=sys.stderr)
            return 1

        score_url = f"{args.api_base}/v1/score/day?date={args.date}&source={args.source}"
        score_resp = _post_with_retry(
            session,
            score_url,
            headers=auth_headers,
            retries=max(1, int(args.retries)),
        )
        if score_resp.status_code >= 400:
            print(f"Score failed ({score_resp.status_code}): {score_resp.text}", file=sys.stderr)
            return 1

    print(json.dumps({
        "ingest": ingest_resp.json(),
        "score": score_resp.json(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
