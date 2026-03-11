# Final verdict

## 1) What tuning was added
- Centralized SQLite tuning in `db_sqlite._connect()` via dedicated helper functions.
- Added safe baseline PRAGMAs across all backend DB connections:
  - WAL
  - synchronous NORMAL
  - busy timeout
  - memory temp store
  - foreign key enforcement
- Added explicit profile-based tuning for Pi and laptop.
- Added minimal `PRAGMA optimize` strategy in DB initialization.

## 2) How Pi and laptop now differ
- Pi profile uses lower memory footprint and more frequent WAL auto-checkpointing.
- Laptop profile uses larger cache + mmap and a larger checkpoint threshold for SSD-friendly read performance.

## 3) Why this should help Pi microSD + swap pressure
- Smaller cache/mmap reduces RAM pressure on Pi 3-class hardware.
- Lower `wal_autocheckpoint` encourages smaller/frequent WAL consolidation, reducing sharp write spikes.
- `journal_size_limit` caps journal growth behavior more explicitly.

## 4) What this does NOT solve
- It does not eliminate all IO latency or filesystem bottlenecks.
- It does not guarantee performance improvements for every workload shape.
- It does not replace capacity planning (RAM, swap, SD endurance, IO scheduler concerns).

## 5) Is this safe to use now on both environments?
- Yes, based on implemented safeguards and passing focused tests.
- The change is additive, centralized, and does not alter planner/forecasting business logic.
- Operationally, profile selection is explicit and reversible via one environment variable.
