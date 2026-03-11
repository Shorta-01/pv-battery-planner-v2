# 06 — Quick start (copy/paste)

## A) Windows laptop — full app (backend + frontend)

### Terminal 1 (backend)
```powershell
cd C:\path\to\pv-battery-planner-v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements-backend.txt
python .\scripts\start_backend.py
```

### Terminal 2 (frontend)
```powershell
cd C:\path\to\pv-battery-planner-v2
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-frontend.txt
python -m streamlit run .\app.py
```

### Verify backend quickly
```powershell
$token = (Get-Content .\local_state\api_token.txt -Raw).Trim()
Invoke-RestMethod -Headers @{ Authorization = "Bearer $token" } -Uri "http://127.0.0.1:8787/v1/health"
```

---

## B) Raspberry Pi 3 — backend only

```bash
cd ~/pv-battery-planner-v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-backend.txt
python scripts/start_backend.py
```

### LAN bind (optional)
```bash
PVBP_BACKEND_HOST=0.0.0.0 PVBP_BACKEND_PORT=8787 python scripts/start_backend.py
```

### Verify backend quickly
```bash
TOKEN=$(cat local_state/api_token.txt)
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8787/v1/health
```

