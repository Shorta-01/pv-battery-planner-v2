# Evidence and operator usage

## 1) Exact files changed
- `db_sqlite.py`
- `tests/test_db_sqlite_tuning_profiles.py`
- `sqlite_tuning_output/01_current_sqlite_lifecycle.md`
- `sqlite_tuning_output/02_tuning_design.md`
- `sqlite_tuning_output/03_tests_and_smoke_checks.md`
- `sqlite_tuning_output/04_evidence_and_usage.md`
- `sqlite_tuning_output/05_final_verdict.md`

## 2) Exact env var for Pi
```bash
export PVBP_SQLITE_PROFILE=pi
```

## 3) Exact env var for laptop
```bash
export PVBP_SQLITE_PROFILE=laptop
```

## 4) Verify active PRAGMA settings

### Via Python helper (recommended)
```python
from db_sqlite import get_sqlite_pragma_snapshot
print(get_sqlite_pragma_snapshot("local_state/planner_history.sqlite"))
```

### Via sqlite3 shell
```sql
PRAGMA journal_mode;
PRAGMA synchronous;
PRAGMA busy_timeout;
PRAGMA temp_store;
PRAGMA cache_size;
PRAGMA mmap_size;
PRAGMA wal_autocheckpoint;
PRAGMA journal_size_limit;
PRAGMA foreign_keys;
```

## 5) Remaining risks / not proven
- Not benchmarked under real Pi 3 production load in this task.
- No long-duration WAL growth benchmark included beyond `journal_size_limit`/checkpoint configuration.
- External scripts using raw `sqlite3.connect(...)` outside `db_sqlite._connect` will not automatically inherit tuning unless they use the centralized helper path.
