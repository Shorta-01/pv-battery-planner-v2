# Tests and smoke checks

## Added focused tests
- New test module: `tests/test_db_sqlite_tuning_profiles.py`

Coverage included:
1. Connection opens and returns PRAGMA snapshot successfully.
2. WAL mode is active.
3. `synchronous` is NORMAL (`1`).
4. `busy_timeout` is 5000.
5. Profile selection (`pi` vs `laptop`) changes cache/mmap/checkpoint/journal size settings.
6. DB init + persistence still work (`insert_forecast_run` + row fetch).
7. Tests are backend-only and do not involve frontend dependencies.

## Commands executed
1. `pytest -q tests/test_db_sqlite_tuning_profiles.py`
   - Result: `3 passed`
2. `pytest -q tests/test_db_sqlite_insert_enriched_fields.py tests/test_db_sqlite_model_hourly_tables.py tests/test_db_sqlite_zoneinfo_import.py`
   - Result: `7 passed`

## Smoke validation notes
- Profile behavior validated through direct PRAGMA reads using helper `get_sqlite_pragma_snapshot(...)`.
- Existing DB persistence tests continued to pass after tuning changes.
