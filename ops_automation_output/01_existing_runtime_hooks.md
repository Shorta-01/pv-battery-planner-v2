# 01 — Existing runtime hooks (reusable as-is)

## Backend startup + SQLite bootstrap
- `BackendState.__init__()` creates `local_state/` and calls `init_db("local_state/planner_history.sqlite")` on process start.
- This means **starting FastAPI backend once is enough to create the SQLite file + tables** if missing.
- Startup command already documented in repo: `python -m uvicorn backend_api:app --host 127.0.0.1 --port 8787`.

## Forecast hook (nightly)
- Endpoint exists: `POST /v1/run/nightly`.
- It executes `state.run_nightly_tick(payload)`.
- Existing behavior is already scheduler-safe:
  - returns `{"ran": false, "reason": "before_window"}` when called before configured nightly time
  - returns `{"ran": false, "reason": "already_ran"}` if already completed for same target day
  - returns `{"ran": true, "reason": "ran"}` when forecast is executed
- Existing helper script exists and can be reused directly:
  - `scripts/nightly_tick.py`
  - posts `{"force": false}` to `/v1/run/nightly`
  - if backend unreachable, it attempts to start backend and retries once.

## Actuals ingestion hook
- Endpoint exists: `POST /v1/actuals/hourly`.
- Supports:
  - `Content-Type: text/csv` body with exact header:
    `ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct`
  - or JSON payload with `rows: []`.
- Persists using `insert_actual_hourly_rows(...)` with strict required columns.
- `INSERT OR REPLACE` behavior means reruns/corrections are operationally safe.

## Daily score hook
- Endpoint exists: `POST /v1/score/day?date=YYYY-MM-DD&source=manual_csv`.
- It computes score using latest forecast run for that target date + actual rows from same date/source.
- Writes to `daily_scores` via upsert.
- Readback endpoint exists: `GET /v1/score/day?date=YYYY-MM-DD&source=manual_csv`.

## Reuse decision
Existing runtime hooks are sufficient for backtest accumulation without FusionSolar:
1. backend startup auto-creates DB
2. nightly forecast endpoint/script already available
3. actual ingest endpoint already available (CSV/manual)
4. score endpoint already available

Only missing piece was a simple operator helper to chain **actuals ingest + score** reliably for daily operations.
