# Backend runtime smoke checks

All checks were run from repository root.

## 1) Backend module import
Command:
- `python -c "import backend_api; print('import backend_api OK')"`

Result:
- PASS (`import backend_api OK`)

## 2) DB init path + local_state init
Command:
- `python scripts/smoke_backend_runtime.py`

What this validates:
- imports `backend_api`
- patches runtime paths to a temp local_state directory
- constructs `BackendState()`
- verifies SQLite DB file creation
- verifies API token creation
- verifies streamlit is not imported during backend startup

Result:
- PASS
- Output included:
  - `PASS: import backend_api`
  - `PASS: sqlite initialized .../local_state/planner_history.sqlite`
  - `PASS: token created .../local_state/api_token.txt`
  - `PASS: streamlit not imported during backend startup`

## 3) Startup helper command path
Command:
- `python -c "import scripts.start_backend as sb; print('start helper import OK')"`

Result:
- PASS (`start helper import OK`)

Notes:
- Startup helper is intended runtime command:
  - `python scripts/start_backend.py`
- Helper ensures uvicorn is launched from repo root and supports host/port env overrides.

## 4) Backend-only install path documented
Validated in README changes:
- Windows backend-only install uses `requirements-backend.txt`
- Raspberry Pi backend-only section uses `requirements-backend.txt`
- Frontend install marked optional

## 5) No frontend dependency required for backend start
Command:
- `pytest -q tests/test_runtime_split_imports.py`

Result:
- PASS (`2 passed`)
- Confirms backend import path does not require `streamlit`/`plotly`.
