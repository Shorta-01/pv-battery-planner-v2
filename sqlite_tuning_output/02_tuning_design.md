# Minimal SQLite tuning design

## Design goals met
1. Centralize PRAGMA application in one place.
2. Apply safe common baseline to all backend DB connections.
3. Support Pi vs laptop explicit profiles.
4. Keep code changes additive and non-invasive.

## Chosen architecture

### Centralized connection tuning
- Keep all tuning inside `db_sqlite.py`.
- Introduce:
  - `_resolve_sqlite_profile()`
  - `_apply_sqlite_pragmas(conn, profile=...)`
- `db_sqlite._connect()` calls `_apply_sqlite_pragmas(...)` for every connection.

### Profile selection mechanism
- Environment variable: `PVBP_SQLITE_PROFILE`
- Supported values:
  - `pi`
  - `laptop`
- Friendly aliases accepted (`default`, `rpi`, `raspberrypi`, etc.) and normalized.
- Safe default when unset or invalid: `laptop`.

## PRAGMA plan

### Common safe baseline (always applied)
- `journal_mode=WAL`
- `synchronous=NORMAL`
- `busy_timeout=5000`
- `temp_store=MEMORY`
- `foreign_keys=ON`

### Pi profile
- `cache_size=-32768` (~32MB)
- `mmap_size=33554432` (32MB)
- `wal_autocheckpoint=256`
- `journal_size_limit=67108864` (64MB)

### Laptop profile
- `cache_size=-131072` (~128MB)
- `mmap_size=268435456` (256MB)
- `wal_autocheckpoint=1000`
- `journal_size_limit=134217728` (128MB)

## Optimize strategy
- Add a minimal `PRAGMA optimize;` call at the end of `init_db(...)`.
- Rationale:
  - Safe, SQLite-native maintenance hint.
  - Runs after schema setup/migrations.
  - Does not alter planner/forecasting business behavior.

## Why this is minimal and safe
- No API contract changes.
- No planner/ensemble logic touched.
- No schema semantic redesign.
- Existing `_connect` pattern preserved; only tuning behavior expanded.
- Explicit profiles improve operator clarity while preserving conservative defaults.
