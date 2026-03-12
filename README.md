# PV Battery Planner v2

PV Battery Planner v2 is a **two-process local application** for day-ahead battery charging advice:

- **Streamlit client** in `app.py` for inputs, visualization, and history browsing.
- **FastAPI backend** in `backend_api.py` for forecast execution, planning logic, token auth, and persistence.

The backend combines weather model forecasts, estimates PV production, computes battery planning metrics, and stores run history locally.

## Features

- **Client/server architecture**
  - `app.py` calls backend API endpoints for health, settings, run execution, and history.
  - `backend_api.py` runs the forecast + planning pipeline and returns structured results.

- **Weather ensemble (5 models)**
  - `ecmwf_ifs` — ECMWF IFS **[Core]**
  - `dwd_icon_d2` — DWD ICON-D2 **[Core]**
  - `knmi_harmonie_arome` — KNMI Harmonie-Arome **[Core]**
  - `dwd_icon_eu` — DWD ICON-EU **[Extra]**
  - `meteofrance_seamless` — Météo-France Seamless **[Extra]**
  - **Core** models are enabled by default for Belgium; **Extra** models are optional for experimentation/robustness.
  - `ensemble_method` supports `weighted`, `mean`, or `median`; backend default is `weighted`.

- **PV uncertainty range**
  - Optional PV uncertainty returns ensemble PV totals including `p10`, `p50`, and `p90`.
  - When available, hourly output can include low/high curves (`pv_total_low_kwh` / `pv_total_high_kwh`).

- **History and persistence**
  - Forecast runs are stored in SQLite at `local_state/planner_history.sqlite`.
  - UI history uses backend endpoint `/v1/results/history`.

## Quickstart (Windows)

> Requires **Python 3.10+**.

1. Create and activate virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install backend dependencies:

```powershell
python -m pip install -U pip
python -m pip install -r requirements-backend.txt
```

3. Start backend (Terminal 1):

```powershell
python scripts/start_backend.py
```

> After any backend code change (including BMW auth/device-flow files), stop and restart this backend process so the running server picks up the latest source.

4. (Optional) Install/start Streamlit UI (Terminal 2):

```powershell
python -m pip install -r requirements-frontend.txt
python -m streamlit run app.py
```

Defaults:
- Backend base URL: `http://127.0.0.1:8787`
- Streamlit UI: `http://localhost:8501`

## Backend-only run (Raspberry Pi 3)

> Backend-only path (no frontend) uses `requirements-backend.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-backend.txt
python scripts/start_backend.py
```

Environment variables supported by `scripts/start_backend.py`:
- `PVBP_BACKEND_HOST` (default `127.0.0.1`)
- `PVBP_BACKEND_PORT` (default `8787`)

Example for LAN access on Pi:

```bash
PVBP_BACKEND_HOST=0.0.0.0 PVBP_BACKEND_PORT=8787 python scripts/start_backend.py
```


## BMW CarData Phase 1 notes

- Current backend Phase 1 supports BMW OAuth device-flow + CarData REST polling.
- REST polling uses `access_token` on `https://api-cardata.bmwgroup.com`; stream/MQTT is not implemented yet.
- BMW onboarding is backend/API-driven in this phase (not full GUI guided flow yet).
- Saving BMW settings updates the live backend runtime immediately (auth/provider are rebuilt from the saved config).
- Debug endpoint `GET /v1/ev/bmw/device_flow/debug` (when `DEBUG=1`) reports masked active runtime values, including client-id mask and active auth/REST URLs.
- After first successful BMW refresh, raw endpoint payload captures are written to `local_state/bmw_payload_captures/` as `bmw_cardata_live_<endpoint>_<timestamp>.json` for mapping validation and fixture promotion.

## Security / API token

- On first backend start, the backend auto-creates:
  - `local_state/api_token.txt`
- The UI reads the API token from:
  1. `PVBP_API_TOKEN` environment variable, or
  2. `local_state/api_token.txt`

`local_state/` also stores:
- `settings.json`
- `last_inputs.json`
- `latest_result.json`
- `results_history.json` (summary list)
- `planner_history.sqlite` (SQLite DB with full runs)

### Reset local state

To fully reset token/settings/history:
1. Stop backend and Streamlit.
2. Delete the `local_state/` folder.
3. Restart backend and UI.

## Nightly scheduler

Use `scripts/nightly_tick.py` to trigger nightly execution.

What it does:
- POSTs to `/v1/run/nightly` with `{"force": false}`.
- If backend is not reachable, it attempts to auto-start backend and retries.

Manual run:

```powershell
python scripts/nightly_tick.py
```

You can run this from **Windows Task Scheduler** after your configured nightly time.

## Repository structure

- `app.py` — Streamlit client UI (calls backend)
- `backend_api.py` — FastAPI backend, token auth, forecast orchestration, history endpoints
- `weather_ensemble.py` — provider catalog, fetching, normalization, ensemble aggregation, PV uncertainty
- `planner_core.py` — PV + battery planning functions used by backend
- `db_sqlite.py` — SQLite schema and insert/query helpers
- `config.json` — default user/system config
