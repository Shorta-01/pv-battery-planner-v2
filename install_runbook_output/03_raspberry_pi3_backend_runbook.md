# 03 — Raspberry Pi 3 runbook (backend-only profile)

> Target: Raspberry Pi 3 runs backend API only (no Streamlit frontend).

## 1) System prerequisites

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip curl sqlite3
```

Verify:

```bash
python3 --version
```

Recommendation: use Python 3.10+ if available.

---

## 2) Clone repo and create venv

```bash
git clone <YOUR_REPO_URL> pv-battery-planner-v2
cd pv-battery-planner-v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Verify active venv:

```bash
python -c "import sys; print(sys.executable)"
```

---

## 3) Install backend-only dependencies

> Use backend-only profile explicitly.

```bash
python -m pip install -r requirements-backend.txt
```

Sanity check (no Streamlit needed):

```bash
python scripts/smoke_backend_runtime.py
```

Expected PASS lines include sqlite + token creation in a temp directory.

---

## 4) Start backend via supported startup helper

Default local-only bind:

```bash
python scripts/start_backend.py
```

LAN-access bind override (supported by helper):

```bash
PVBP_BACKEND_HOST=0.0.0.0 PVBP_BACKEND_PORT=8787 python scripts/start_backend.py
```

Keep this process running.

---

## 5) Verify SQLite creation in repo local_state

In another shell (same repo):

```bash
ls -l local_state/planner_history.sqlite
```

Expected: file exists.

Optional table list:

```bash
sqlite3 local_state/planner_history.sqlite '.tables'
```

---

## 6) Verify API reachable locally

```bash
TOKEN=$(cat local_state/api_token.txt)
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8787/v1/health
```

Expected JSON contains `"status":"ok"`.

Also verify settings endpoint:

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8787/v1/settings | head
```

---

## 7) Host/port override usage (supported)

The startup helper supports:
- `PVBP_BACKEND_HOST`
- `PVBP_BACKEND_PORT`

Examples:

```bash
# Different port
PVBP_BACKEND_PORT=8799 python scripts/start_backend.py

# LAN bind + custom port
PVBP_BACKEND_HOST=0.0.0.0 PVBP_BACKEND_PORT=8799 python scripts/start_backend.py
```

Update clients/scripts to matching base URL, e.g. `http://<pi-ip>:8799`.

---

## Backend-only profile reminder

For Pi 3, keep installs to backend profile unless you explicitly need UI tools:

```bash
python -m pip install -r requirements-backend.txt
```

Do **not** install `requirements-frontend.txt` for the backend-only target unless troubleshooting requires it.

