# 01 — Current runnable paths (as implemented on branch `work`)

## Scope inspected
- `README.md`
- `scripts/start_backend.py`
- `scripts/smoke_backend_runtime.py`
- `requirements-backend.txt`
- `requirements-frontend.txt`
- `requirements.txt`
- `backend_api.py`
- `app.py`

---

## A) Backend-only install/start path (supported)

### Install profile
- Backend-only dependency profile is `requirements-backend.txt`:
  - `requests`, `numpy`, `pandas`, `pvlib`, `fastapi`, `uvicorn`.
- This avoids `streamlit`/`plotly` frontend dependencies.

### Start command (preferred helper)
- Supported startup helper:

```bash
python scripts/start_backend.py
```

- Helper resolves repository root, then launches:

```bash
python -m uvicorn backend_api:app --host <host> --port <port>
```

- Host/port override support (from env vars consumed by the helper):
  - `PVBP_BACKEND_HOST` (default `127.0.0.1`)
  - `PVBP_BACKEND_PORT` (default `8787`)

Example:

```bash
PVBP_BACKEND_HOST=0.0.0.0 PVBP_BACKEND_PORT=8787 python scripts/start_backend.py
```

### Auth behavior
- Backend API endpoints require bearer token.
- Token is created on first backend start at:

```text
local_state/api_token.txt
```

---

## B) Frontend install/start path (supported)

### Install profile
- Frontend dependency profile is `requirements-frontend.txt`, which includes:
  - all backend requirements via `-r requirements-backend.txt`
  - `streamlit`
  - `plotly`

### Start command

```bash
python -m streamlit run app.py
```

### Frontend ↔ backend connection defaults
- Frontend backend base URL default:

```text
http://127.0.0.1:8787
```

- Override with env var:

```text
PVBP_BACKEND_URL
```

- Frontend token lookup order:
  1. `PVBP_API_TOKEN` env var
  2. `local_state/api_token.txt`

---

## C) DB creation verification path (supported)

### What creates DB
- Backend startup (`BackendState()` initialization) runs `init_db(...)` against:

```text
local_state/planner_history.sqlite
```

- Therefore DB file creation is guaranteed during backend initialization (if startup succeeds).

### Programmatic smoke check already provided

```bash
python scripts/smoke_backend_runtime.py
```

This smoke script verifies:
- backend module import succeeds
- sqlite file is initialized
- token file is created
- streamlit is not imported during backend startup

---

## D) Local API verification path (supported)

### Minimal health endpoint
- Endpoint: `GET /v1/health`
- Requires header: `Authorization: Bearer <token>`

### Practical verification command pattern
1) read token from local file,
2) call health endpoint,
3) expect JSON with `status: ok` and current time.

Linux/macOS:

```bash
TOKEN=$(cat local_state/api_token.txt)
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8787/v1/health
```

PowerShell:

```powershell
$token = Get-Content .\local_state\api_token.txt -Raw
$token = $token.Trim()
Invoke-RestMethod -Headers @{ Authorization = "Bearer $token" } -Uri "http://127.0.0.1:8787/v1/health"
```

Expected response shape:

```json
{"status":"ok","time":"<iso-datetime>"}
```

