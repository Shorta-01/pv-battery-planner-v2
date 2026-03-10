# 02 — Minimum daily loop design

## Objective
Accumulate historical forecast-vs-actual data and daily scoring with the smallest reliable process, without FusionSolar integration.

## Daily sequence (minimal)

### 0) Keep backend running
- Process: `uvicorn backend_api:app --host 127.0.0.1 --port 8787`
- Why: backend startup guarantees DB/tables exist and exposes all required endpoints.

### 1) Forecast trigger (nightly)
- Time: after configured `nightly_run_time` (default `22:00` local timezone in settings).
- Action: run `python scripts/nightly_tick.py`.
- Expected result:
  - `ran=true` once per target day
  - idempotent skips (`before_window` / `already_ran`) are non-fail states.

### 2) Actuals ingestion (next day, when full day is available)
- Time: once prior day hourly actuals are ready (typically after midnight local, e.g. 00:05–00:30).
- Action: submit CSV for that previous date to `/v1/actuals/hourly`.
- Source tag: keep stable (`manual_csv`) to align ingest and score queries.

### 3) Daily scoring
- Time: immediately after successful actuals ingestion for that date.
- Action: `POST /v1/score/day?date=<prior_day>&source=manual_csv`.
- Expected result: row upserted in `daily_scores` and retrievable with GET endpoint.

## Suggested timing window
- 22:05 local: nightly forecast trigger
- 00:10 local: ingest yesterday actuals + compute yesterday score

This creates a clean two-job operating model suitable for Task Scheduler/cron.

## Failure/retry notes
- Forecast job:
  - `scripts/nightly_tick.py` already retries after backend auto-start attempt.
  - `before_window` and `already_ran` are operationally expected; do not treat as hard failure.
- Actuals+score job:
  - Retry API calls up to a few times for transient backend/network issues.
  - Keep ingestion idempotent by reusing same `source` + timestamps (DB replace semantics).
- Data readiness:
  - If scoring fails with missing overlap, verify target date alignment and full hourly rows.
- Backfill:
  - Repeat ingest+score for historical dates as needed; upserts keep latest corrected values.
