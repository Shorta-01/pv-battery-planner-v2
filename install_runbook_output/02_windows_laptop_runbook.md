# 02 — Windows laptop runbook (frontend + backend)

> Target: run full app locally on Windows (PowerShell).

## 0) Open PowerShell in repo root

```powershell
cd C:\path\to\pv-battery-planner-v2
```

Verify:

```powershell
Get-ChildItem .\app.py, .\backend_api.py, .\scripts\start_backend.py
```

---

## 1) Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Verify active interpreter:

```powershell
python -c "import sys; print(sys.executable)"
```

Expected: path contains `.venv`.

---

## 2) Install backend requirements (required)

```powershell
python -m pip install -r requirements-backend.txt
```

Quick verify imports:

```powershell
python -c "import fastapi, uvicorn, pandas, requests, pvlib; print('backend deps ok')"
```

---

## 3) Install frontend requirements (optional but needed for full app)

```powershell
python -m pip install -r requirements-frontend.txt
```

Quick verify:

```powershell
python -c "import streamlit, plotly; print('frontend deps ok')"
```

---

## 4) Start backend (Terminal 1)

```powershell
python .\scripts\start_backend.py
```

Expected log includes uvicorn startup and listener on `127.0.0.1:8787`.

---

## 5) Verify `local_state/planner_history.sqlite` is created

In a second PowerShell terminal (same repo, venv active):

```powershell
Test-Path .\local_state\planner_history.sqlite
Get-Item .\local_state\planner_history.sqlite | Format-List FullName,Length,LastWriteTime
```

Expected:
- `Test-Path` returns `True`
- file metadata is shown.

---

## 6) Verify backend API responds

Read token then call health endpoint:

```powershell
$token = (Get-Content .\local_state\api_token.txt -Raw).Trim()
Invoke-RestMethod -Headers @{ Authorization = "Bearer $token" } -Uri "http://127.0.0.1:8787/v1/health"
```

Expected response has:
- `status = ok`
- `time = <timestamp>`

Optional deeper check (settings):

```powershell
Invoke-RestMethod -Headers @{ Authorization = "Bearer $token" } -Uri "http://127.0.0.1:8787/v1/settings" | ConvertTo-Json -Depth 3
```

---

## 7) Start frontend (Terminal 2)

If backend runs on default URL, no extra env var needed:

```powershell
python -m streamlit run .\app.py
```

If backend URL differs, set it before launch:

```powershell
$env:PVBP_BACKEND_URL = "http://127.0.0.1:8787"
python -m streamlit run .\app.py
```

---

## 8) Verify frontend can connect to backend

1. Open Streamlit URL shown in terminal (typically `http://localhost:8501`).
2. Confirm app loads without “backend unreachable” error.
3. If prompted for token, ensure token exists at `local_state/api_token.txt` or set env var:

```powershell
$env:PVBP_API_TOKEN = (Get-Content .\local_state\api_token.txt -Raw).Trim()
```

4. In UI, open/refresh once; backend logs should show requests to `/v1/settings` and `/v1/health`.

---

## Stop commands

- Stop backend: press `Ctrl+C` in backend terminal.
- Stop frontend: press `Ctrl+C` in streamlit terminal.

