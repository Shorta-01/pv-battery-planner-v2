"""One-time migration helper.

Run with:
    python scripts/migrate_history_to_sqlite.py
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_sqlite import init_db, insert_forecast_run

LOCAL_STATE_DIR = Path("local_state")
DB_PATH = LOCAL_STATE_DIR / "planner_history.sqlite"
RESULTS_HISTORY = LOCAL_STATE_DIR / "results_history.json"
LATEST_RESULT = LOCAL_STATE_DIR / "latest_result.json"
RUN_HISTORY_LOG = Path("run_history_log.json")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _count_existing_runs() -> int:
    if not DB_PATH.exists():
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()
        return int(row[0]) if row else 0


def main() -> None:
    LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
    init_db(str(DB_PATH))

    payloads: list[dict] = []
    history = _read_json(RESULTS_HISTORY, [])
    if isinstance(history, list):
        payloads.extend([item for item in history if isinstance(item, dict)])

    latest = _read_json(LATEST_RESULT, {})
    if isinstance(latest, dict) and latest:
        payloads.append(latest)

    fallback_log = _read_json(RUN_HISTORY_LOG, [])
    fallback_by_date: dict[str, dict] = {}
    if isinstance(fallback_log, list):
        for row in fallback_log:
            if isinstance(row, dict) and row.get("Date"):
                fallback_by_date[str(row["Date"])] = row

    total_payloads = len(payloads)
    inserted_or_replaced = 0
    skipped = 0

    before_count = _count_existing_runs()

    for payload in payloads:
        if not isinstance(payload, dict):
            skipped += 1
            continue

        target_date = payload.get("target_date")
        if not target_date:
            skipped += 1
            continue

        payload.setdefault("run_id", str(uuid.uuid4()))
        payload.setdefault("run_at_utc", dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat())

        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        payload["metrics"] = metrics

        fallback = fallback_by_date.get(str(target_date), {})
        if "charge_kw" not in metrics and fallback:
            metrics["charge_kw"] = float(fallback.get("Allowed AC charge power (kW)", 0.0) or 0.0)
        if "cutoff_soc" not in metrics and fallback:
            cutoff_pct = float(fallback.get("AC charge cutoff SOC (%)", 0.0) or 0.0)
            metrics["cutoff_soc"] = cutoff_pct / 100.0

        try:
            insert_forecast_run(str(DB_PATH), payload)
            inserted_or_replaced += 1
        except Exception:
            skipped += 1

    after_count = _count_existing_runs()
    inserted_count = max(0, after_count - before_count)
    replaced_count = max(0, inserted_or_replaced - inserted_count)

    print(f"Total payloads found: {total_payloads}")
    print(f"Inserted count: {inserted_count}")
    print(f"Updated/replaced count: {replaced_count}")
    print(f"Skipped count: {skipped}")


if __name__ == "__main__":
    main()
