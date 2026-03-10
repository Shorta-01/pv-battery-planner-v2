# 04 — Operator runbook (backtest-ready loop without FusionSolar)

## 1) First-time startup

1. Create venv + install deps:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

2. Start backend:
```bash
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8787
```

## 2) Verify DB creation
After backend starts once, verify local state files:
```bash
test -f local_state/planner_history.sqlite && echo "DB OK"
test -f local_state/api_token.txt && echo "TOKEN OK"
```

Optional health check:
```bash
TOKEN="$(cat local_state/api_token.txt)"
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8787/v1/health
```

## 3) Trigger forecast (nightly)
Run after `nightly_run_time` (default 22:00 local):
```bash
python scripts/nightly_tick.py
```

Interpret output:
- `"ran": true` => forecast executed
- `"reason": "before_window"` => too early; rerun later
- `"reason": "already_ran"` => already done for today’s target day

## 4) Post actuals (manual/file source)
Prepare CSV for completed day with exact header:
```csv
ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct
```

Run ingestion + scoring together (recommended):
```bash
python scripts/daily_actuals_and_score.py \
  --actuals-csv data/actuals_YYYY-MM-DD.csv \
  --date YYYY-MM-DD \
  --source manual_csv
```

If needed, ingest-only via curl:
```bash
TOKEN="$(cat local_state/api_token.txt)"
curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/csv" \
  --data-binary @data/actuals_YYYY-MM-DD.csv \
  http://127.0.0.1:8787/v1/actuals/hourly
```

## 5) Run scoring (if not using helper)
```bash
TOKEN="$(cat local_state/api_token.txt)"
curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8787/v1/score/day?date=YYYY-MM-DD&source=manual_csv"
```

## 6) Verify scored days are accumulating
Check single day:
```bash
TOKEN="$(cat local_state/api_token.txt)"
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8787/v1/score/day?date=YYYY-MM-DD&source=manual_csv"
```

Optional DB check:
```bash
sqlite3 local_state/planner_history.sqlite \
  "select score_date, source, pv_mae_kwh, pv_rmse_kwh, pv_daily_error_kwh from daily_scores order by score_date desc limit 10;"
```

## 7) Common failure checks

1. `401 Invalid bearer token`
- Ensure token matches `local_state/api_token.txt` or `PVBP_API_TOKEN`.

2. `No forecast run found for date ...`
- Nightly forecast for that score date was not executed.
- Run forecast for appropriate day first.

3. `No overlapping forecast and actual rows for this day`
- Date mismatch between forecast target date and actual CSV timestamps.
- Ensure `ts_local` rows belong to score date and are hourly.

4. CSV header/columns rejected
- Must match exactly:
  `ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct`

5. Backend not reachable
- Start/restart uvicorn backend process.
- `scripts/nightly_tick.py` auto-starts once, but daily close helper expects backend reachable.

## Minimal always-on operating model
- Keep backend service running continuously.
- Schedule:
  - Nightly forecast trigger after configured nightly time.
  - Daily close (actuals ingest + score) shortly after midnight for previous day.
- Repeat daily to accumulate backtest history.
