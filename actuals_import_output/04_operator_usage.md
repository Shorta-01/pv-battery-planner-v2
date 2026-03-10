# Operator usage guide (actuals CSV import)

## 1) Expected CSV format
Header must be exactly:

```csv
ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct
2026-03-01T10:00:00,1.20,0.95,0.20,0.45,57
2026-03-01T11:00:00,1.35,1.05,0.18,0.48,58
```

Rules:
- `ts_local` must be hourly local shape: `YYYY-MM-DDTHH:00:00`
- Numeric columns must be numeric.

## 2) Exact import command
```bash
python scripts/import_actuals_csv.py ./data/actuals_hourly.csv --source manual_csv --api-base http://127.0.0.1:8787
```

Token resolution (in order):
1. `--token`
2. `PVBP_API_TOKEN`
3. `local_state/api_token.txt`

## 3) Verify rows landed in `actual_hourly`
```bash
sqlite3 local_state/planner_history.sqlite "SELECT source, ts_local, pv_kwh, load_kwh, grid_import_kwh, grid_export_kwh, soc_pct FROM actual_hourly WHERE source='manual_csv' ORDER BY ts_local DESC LIMIT 20;"
```

## 4) Common mistakes
- Wrong header order or missing column.
- Timestamp uses space (`2026-03-01 10:00:00`) instead of `T` format.
- Non-numeric values in kWh/soc fields.
- Missing bearer token.
- Pointing `--api-base` at wrong backend instance/port.
