# 05 — Troubleshooting

## 1) Import / module errors

### Symptom
- `ModuleNotFoundError` for `fastapi`, `uvicorn`, `streamlit`, etc.

### Fix
1. Ensure venv is active.
2. Reinstall matching requirement set:

```bash
python -m pip install -U pip
python -m pip install -r requirements-backend.txt
# For full UI:
python -m pip install -r requirements-frontend.txt
```

3. Verify interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

---

## 2) Missing SQLite file (`local_state/planner_history.sqlite`)

### Symptom
- DB file not found after startup.

### Checks
- Did backend process actually start?
- Are you in repo root when launching?

### Fix
```bash
python scripts/start_backend.py
```

Then verify:

```bash
ls -l local_state/planner_history.sqlite
```

If still missing, run smoke:

```bash
python scripts/smoke_backend_runtime.py
```

---

## 3) Backend does not start

### Common causes
- Port already in use.
- Broken venv/deps.
- Wrong working directory.

### Checks

```bash
python scripts/start_backend.py
```

Read immediate traceback.

### Port conflict resolution

Linux/macOS:
```bash
ss -ltnp | rg 8787
```

Windows PowerShell:
```powershell
Get-NetTCPConnection -LocalPort 8787 -State Listen
```

Then either stop conflicting process or use another port:

```bash
PVBP_BACKEND_PORT=8799 python scripts/start_backend.py
```

---

## 4) Frontend cannot reach backend

### Symptom
- UI shows backend unreachable.

### Checks
1. Backend alive?

```bash
TOKEN=$(cat local_state/api_token.txt)
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8787/v1/health
```

2. Frontend points to right backend URL?
- Default: `http://127.0.0.1:8787`
- Override with `PVBP_BACKEND_URL`.

3. Token available to frontend?
- `PVBP_API_TOKEN` or `local_state/api_token.txt`.

### Fix
- Restart backend and streamlit terminals.
- Set explicit URL/token env vars before launching frontend.

---

## 5) Actuals ingest rejects CSV/JSON

### CSV rejection causes
- Header mismatch (must be exact order):
  `ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct`
- Bad timestamp format (`YYYY-MM-DDTHH:00:00`).
- Non-numeric values in numeric columns.

### Validate with helper

```bash
python scripts/import_actuals_csv.py ./actuals.csv --api-base http://127.0.0.1:8787
```

Helper prints row-level validation errors.

### JSON rejection causes
- Payload is not list/object.
- Missing `rows` list in object payload.

Use expected JSON shape:

```json
{"source":"manual_csv","rows":[{"ts_local":"2026-01-15T00:00:00","pv_kwh":0,"load_kwh":0.8,"grid_import_kwh":0.5,"grid_export_kwh":0,"soc_pct":42}]}
```

---

## 6) Port conflicts

### Symptom
- `Address already in use` on startup.

### Fix options
1. Stop process using port.
2. Move backend to another port:

```bash
PVBP_BACKEND_PORT=8799 python scripts/start_backend.py
```

3. Update all clients/scripts:
- `PVBP_BACKEND_URL=http://127.0.0.1:8799`

---

## 7) Raspberry Pi 3 performance/resource issues

### Symptoms
- Slow API responses.
- OOM kills or swapping.

### Mitigations
- Use backend-only profile (`requirements-backend.txt`)—no streamlit.
- Keep backend local bind unless needed externally.
- Avoid concurrent heavy runs.
- Monitor memory/CPU:

```bash
free -h
top
```

- If unstable, reboot and relaunch backend cleanly.

---

## 8) Missing dependency / wheel install issues

### Symptom
- Pip fails building wheels on Pi.

### Fix sequence

```bash
sudo apt update
sudo apt install -y python3-dev build-essential
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-backend.txt
```

If one package still fails, capture full pip error and pin/adjust only that package locally (do not change planner logic).

