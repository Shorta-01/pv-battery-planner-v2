# 03 — Minimal automation helpers implemented

## Added helper script
- `scripts/daily_actuals_and_score.py`

Purpose:
- scheduler-friendly, single command that:
  1) ingests hourly actuals CSV (`POST /v1/actuals/hourly`)
  2) computes daily score (`POST /v1/score/day`)
- loads token from `--token`, `PVBP_API_TOKEN`, or `local_state/api_token.txt`
- defaults score date to **yesterday**
- includes lightweight retry logic for transient failures

## Existing helper reused
- `scripts/nightly_tick.py` (already present)
- used for nightly forecast trigger

## Command examples

### A) Nightly forecast (existing)
```bash
python scripts/nightly_tick.py
```

### B) Daily close (new helper)
```bash
python scripts/daily_actuals_and_score.py \
  --actuals-csv data/actuals_2026-03-01.csv \
  --date 2026-03-01 \
  --source manual_csv
```

### C) Daily close with default date (yesterday)
```bash
python scripts/daily_actuals_and_score.py --actuals-csv data/actuals_yesterday.csv
```

## CSV format requirement
Exact header required by backend:
```csv
ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct
2026-03-01T00:00:00,0.0,0.6,0.4,0.0,53.0
2026-03-01T01:00:00,0.0,0.5,0.3,0.0,52.0
```

## Scheduler-ready entrypoints

### Windows Task Scheduler (minimal)
1. Task A (nightly forecast):
   - Program/script: `python`
   - Arguments: `scripts/nightly_tick.py`
   - Start in: repo root
2. Task B (daily close):
   - Program/script: `python`
   - Arguments: `scripts/daily_actuals_and_score.py --actuals-csv <path> --date <yyyy-mm-dd> --source manual_csv`
   - Start in: repo root

### Cron (minimal)
```cron
# Nightly forecast after configured nightly time
5 22 * * * cd /workspace/pv-battery-planner-v2 && /usr/bin/python3 scripts/nightly_tick.py >> logs/nightly_tick.log 2>&1

# Daily close for previous day (requires upstream CSV drop)
10 0 * * * cd /workspace/pv-battery-planner-v2 && /usr/bin/python3 scripts/daily_actuals_and_score.py --actuals-csv data/actuals_yesterday.csv >> logs/daily_close.log 2>&1
```

(If your CSV filename contains date, generate date-specific command in wrapper script or scheduler variable expansion.)
