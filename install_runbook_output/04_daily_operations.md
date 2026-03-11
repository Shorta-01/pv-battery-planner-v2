# 04 — Daily operations guide

## 1) Start / stop backend

### Start (recommended helper)

```bash
python scripts/start_backend.py
```

Optional bind override:

```bash
PVBP_BACKEND_HOST=0.0.0.0 PVBP_BACKEND_PORT=8787 python scripts/start_backend.py
```

### Stop
- In foreground terminal: `Ctrl+C`.

### Health check after start

```bash
TOKEN=$(cat local_state/api_token.txt)
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8787/v1/health
```

---

## 2) Restart after code update

1. Pull/update code.
2. Re-activate venv.
3. Reinstall dependencies (safe/idempotent):

```bash
python -m pip install -r requirements-backend.txt
```

4. Restart backend:

```bash
python scripts/start_backend.py
```

5. Re-run health check.

---

## 3) Confirm forecast history is being written

### Trigger one run (API)

```bash
TOKEN=$(cat local_state/api_token.txt)
curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"soc_now_percent":50,"yesterday_consumption_kwh":18}' \
  http://127.0.0.1:8787/v1/run/now
```

### Confirm history via API

```bash
curl -s -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:8787/v1/results/history?days=7&show_all_runs=true"
```

### Confirm history in SQLite

```bash
sqlite3 local_state/planner_history.sqlite "SELECT COUNT(*) AS forecast_runs FROM forecast_runs;"
```

Count should increase after successful runs.

---

## 4) Ingest actuals

## Option A — strict CSV helper script (recommended)

```bash
python scripts/import_actuals_csv.py ./actuals.csv --api-base http://127.0.0.1:8787
```

Required CSV header (exact order):

```text
ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct
```

Timestamp format per row:

```text
YYYY-MM-DDTHH:00:00
```

## Option B — direct API with CSV body

```bash
TOKEN=$(cat local_state/api_token.txt)
curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/csv" \
  --data-binary @actuals.csv \
  http://127.0.0.1:8787/v1/actuals/hourly
```

---

## 5) Run daily scoring

### Direct API scoring call

```bash
TOKEN=$(cat local_state/api_token.txt)
DATE=$(date -d 'yesterday' +%F 2>/dev/null || python - <<'PY'
import datetime as dt
print((dt.date.today()-dt.timedelta(days=1)).isoformat())
PY
)
curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8787/v1/score/day?date=$DATE&source=manual_csv"
```

### Combined ingest + scoring helper

```bash
python scripts/daily_actuals_and_score.py --actuals-csv ./actuals.csv --date 2026-01-15 --api-base http://127.0.0.1:8787
```

---

## 6) Confirm backtest data is accumulating

Daily scoring writes backtest rows by model/date/source.

Check counts:

```bash
sqlite3 local_state/planner_history.sqlite "SELECT COUNT(*) AS backtest_rows FROM backtest_daily_scores;"
```

Check latest entries:

```bash
sqlite3 local_state/planner_history.sqlite "SELECT score_date, model_id, source, pv_mae_kwh, created_at_utc FROM backtest_daily_scores ORDER BY score_date DESC, model_id LIMIT 20;"
```

Check daily_scores:

```bash
sqlite3 local_state/planner_history.sqlite "SELECT score_date, source, pv_mae_kwh, pv_rmse_kwh, created_at_utc FROM daily_scores ORDER BY created_at_utc DESC LIMIT 10;"
```

---

## Optional nightly automation trigger

```bash
python scripts/nightly_tick.py
```

This posts to `/v1/run/nightly`; if backend is down, it attempts a local backend start and retries.

