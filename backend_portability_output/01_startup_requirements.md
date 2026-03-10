# Backend startup requirements (Windows + Raspberry Pi 3)

## Scope inspected
- `backend_api.py`
- `scripts/nightly_tick.py`
- `scripts/start_backend.py` (added)
- `README.md`
- `requirements-backend.txt`
- `requirements-frontend.txt`
- `requirements.txt`

## 1) Minimum Python version for backend-only
- Practical minimum is **Python 3.10+**.
- Reasoning:
  - Project README declares Python 3.10+.
  - Backend and DB code use modern typing syntax (`list[str]`, `float | None`) and `zoneinfo`, which are consistent with the 3.10 baseline.

## 2) Backend startup command
- Direct command:
  - `python -m uvicorn backend_api:app --host 127.0.0.1 --port 8787`
- New helper command (platform-neutral):
  - `python scripts/start_backend.py`
- Host/port can be controlled with:
  - `PVBP_BACKEND_HOST` (default `127.0.0.1`)
  - `PVBP_BACKEND_PORT` (default `8787`)

## 3) File/path assumptions
- Backend stores runtime state under relative `local_state/`:
  - `settings.json`
  - `last_inputs.json`
  - `latest_result.json`
  - `results_history.json`
  - `planner_history.sqlite`
  - `api_token.txt`
- Backend creates `local_state/` and DB parent paths automatically at startup.
- Additional relative path assumption:
  - `run_history_log.json` at repo root.
- Weather cache path assumption:
  - `local_state/provider_cache/`.

## 4) Likely Windows failure points
- No clear backend hard blocker found in current backend startup path.
- Path handling uses `pathlib.Path` and relative paths, which are cross-platform.
- Potential operational pitfall:
  - Starting from an unexpected working directory can place `local_state/` elsewhere. Helper script now forces repo-root cwd.

## 5) Likely Raspberry Pi 3 failure points
- No backend code-level blocker for startup found.
- Main risk is dependency/runtime cost on Pi 3 (see blockers report):
  - `numpy`, `pandas`, `pvlib` install/build time and memory footprint.
  - Forecast execution may be CPU-heavy for Pi 3 compared to laptop.
- Backend-only startup path is viable when only `requirements-backend.txt` is installed.
