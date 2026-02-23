import datetime as dt
import tempfile

from db_sqlite import fetch_effective_daily_kwh, init_db


def test_fetch_effective_daily_kwh_works_without_zoneinfo_nameerror():
    with tempfile.NamedTemporaryFile(suffix='.sqlite') as tmp:
        init_db(tmp.name)

        import sqlite3

        now = dt.datetime.now(dt.timezone.utc)
        rows = [
            (
                'run-1',
                now.isoformat(),
                8.5,
            ),
            (
                'run-2',
                (now - dt.timedelta(days=1)).isoformat(),
                9.0,
            ),
        ]

        with sqlite3.connect(tmp.name) as conn:
            conn.executemany(
                """
                INSERT INTO forecast_runs (
                    run_id, target_date, run_at_utc, status, yesterday_kwh_used
                ) VALUES (?, ?, ?, 'ok', ?)
                """,
                [(rid, run_at[:10], run_at, ykwh) for rid, run_at, ykwh in rows],
            )
            conn.commit()

        value, meta = fetch_effective_daily_kwh(tmp.name, lookback_runs=14)

        assert (value is None) or isinstance(value, float)
        assert isinstance(meta, dict)
        assert 'method' in meta
