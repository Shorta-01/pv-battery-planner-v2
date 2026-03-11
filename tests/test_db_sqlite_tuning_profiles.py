import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_sqlite import get_sqlite_pragma_snapshot, init_db, insert_forecast_run


def _assert_common(snapshot: dict):
    assert snapshot["journal_mode"].lower() == "wal"
    assert snapshot["synchronous"] == 1
    assert snapshot["busy_timeout"] == 5000
    assert snapshot["foreign_keys"] == 1


def test_sqlite_tuning_defaults_to_laptop_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("PVBP_SQLITE_PROFILE", raising=False)
    db_path = tmp_path / "planner.sqlite"

    init_db(str(db_path))
    snapshot = get_sqlite_pragma_snapshot(str(db_path))

    _assert_common(snapshot)
    assert snapshot["profile"] == "laptop"
    assert snapshot["cache_size"] == -131072
    assert snapshot["mmap_size"] == 268435456
    assert snapshot["wal_autocheckpoint"] == 1000
    assert snapshot["journal_size_limit"] == 134217728


def test_sqlite_tuning_pi_profile_applies_expected_pragmas(tmp_path, monkeypatch):
    monkeypatch.setenv("PVBP_SQLITE_PROFILE", "pi")
    db_path = tmp_path / "planner.sqlite"

    init_db(str(db_path))
    snapshot = get_sqlite_pragma_snapshot(str(db_path))

    _assert_common(snapshot)
    assert snapshot["profile"] == "pi"
    assert snapshot["cache_size"] == -32768
    assert snapshot["mmap_size"] == 33554432
    assert snapshot["wal_autocheckpoint"] == 256
    assert snapshot["journal_size_limit"] == 67108864


def test_sqlite_init_and_persistence_still_work_with_tuning(tmp_path, monkeypatch):
    monkeypatch.setenv("PVBP_SQLITE_PROFILE", "laptop")
    db_path = tmp_path / "planner.sqlite"

    init_db(str(db_path))
    insert_forecast_run(
        str(db_path),
        {
            "run_id": "run-tuning-check",
            "target_date": "2026-04-01",
            "run_at_utc": "2026-03-31T22:00:00+00:00",
            "metrics": {"pv_forecast_kwh": 2.0, "cons_forecast_kwh": 8.0},
        },
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT run_id FROM forecast_runs WHERE run_id = ?", ("run-tuning-check",)).fetchone()
        assert row is not None

    snapshot = get_sqlite_pragma_snapshot(str(db_path))
    _assert_common(snapshot)
