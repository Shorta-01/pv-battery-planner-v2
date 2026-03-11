# Current SQLite lifecycle inspection

## Files inspected
- `db_sqlite.py`
- `backend_api.py`
- `scripts/start_backend.py`
- `scripts/migrate_history_to_sqlite.py`
- DB-focused tests under `tests/`

## 1) Where SQLite connections are created
- Primary runtime connection path is `db_sqlite._connect(db_path)`, used by nearly all DB operations in `db_sqlite.py`.
- Before this change, `_connect` directly called `sqlite3.connect(db_path)` and immediately applied:
  - `PRAGMA journal_mode=WAL;`
  - `PRAGMA synchronous=NORMAL;`
- Secondary direct connection path exists in migration and tests:
  - `scripts/migrate_history_to_sqlite.py` uses `sqlite3.connect` in `_count_existing_runs()`.
  - Several tests use direct `sqlite3.connect` to assert persisted rows.

## 2) Where schema init runs
- `init_db(db_path)` in `db_sqlite.py` is the schema/migration bootstrap.
- Runtime startup path:
  - `backend_api.BackendState.__init__` calls `init_db(str(SQLITE_PATH))` early in process initialization.
- One-time migration script path:
  - `scripts/migrate_history_to_sqlite.py` calls `init_db(str(DB_PATH))` before importing JSON payloads.

## 3) Where PRAGMAs were currently applied (before this task)
- In `db_sqlite._connect` only:
  - `journal_mode=WAL`
  - `synchronous=NORMAL`
- No centralized profile mechanism existed.
- No explicit `busy_timeout`, `temp_store`, `foreign_keys`, `cache_size`, `mmap_size`, `wal_autocheckpoint`, or `journal_size_limit` tuning existed.

## 4) Whether multiple connection paths exist
- Yes.
  - Main path: `db_sqlite._connect` (used everywhere in backend DB operations).
  - Additional direct SQLite connections in migration/test code for read/assert usage.

## 5) Whether tuning was already partially present
- Yes (partial): WAL + synchronous NORMAL already existed centrally in `_connect`.
- No environment-specific profile tuning existed.
- No explicit optimize strategy was present.
