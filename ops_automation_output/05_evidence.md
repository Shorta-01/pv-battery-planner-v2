# 05 — Evidence

## Files added/changed

### Added
1. `scripts/daily_actuals_and_score.py`
   - New minimal automation helper for daily close (ingest actuals + score).
2. `ops_automation_output/01_existing_runtime_hooks.md`
3. `ops_automation_output/02_daily_loop_design.md`
4. `ops_automation_output/03_automation_helpers.md`
5. `ops_automation_output/04_runbook.md`
6. `ops_automation_output/05_evidence.md`

### Changed
- No planner or ensemble code changed.
- No backend API contract changed.

## Exact local commands to run

### One-time startup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8787
```

### Nightly forecast job
```bash
python scripts/nightly_tick.py
```

### Daily actuals + scoring job
```bash
python scripts/daily_actuals_and_score.py \
  --actuals-csv data/actuals_YYYY-MM-DD.csv \
  --date YYYY-MM-DD \
  --source manual_csv
```

## Operator verification steps

1. Verify DB exists after backend starts:
```bash
test -f local_state/planner_history.sqlite && echo "planner_history.sqlite created"
```

2. Verify nightly run creates/updates forecast data:
```bash
TOKEN="$(cat local_state/api_token.txt)"
curl -s -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:8787/v1/results/history?limit_days=3&latest_per_day=true"
```

3. Verify actuals ingestion returns inserted count:
- output includes `{"ingest": {"inserted": <n>, ...}}` from helper script.

4. Verify daily score exists:
```bash
TOKEN="$(cat local_state/api_token.txt)"
curl -s -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:8787/v1/score/day?date=YYYY-MM-DD&source=manual_csv"
```

5. Verify accumulation across days:
```bash
sqlite3 local_state/planner_history.sqlite "select score_date, source, pv_mae_kwh from daily_scores order by score_date desc limit 15;"
```

If rows continue increasing day-over-day, backtest history accumulation is operational.
