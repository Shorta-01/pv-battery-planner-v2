# Dependency inventory

## Inspected files
- `requirements.txt`
- `app.py`
- `backend_api.py`
- `planner_core.py`
- `weather_ensemble.py`
- `db_sqlite.py`

## Current dependency list (before split)
From `requirements.txt`:
- `requests`
- `numpy`
- `pandas`
- `pvlib`
- `streamlit`
- `plotly`
- `pytest`
- `fastapi`
- `uvicorn`

## Backend-required dependencies
These are required by backend and core runtime paths (`backend_api.py`, `planner_core.py`, `weather_ensemble.py`, `db_sqlite.py`):
- `requests`
- `numpy`
- `pandas`
- `pvlib`
- `fastapi`
- `uvicorn`

## Frontend-only dependencies
These are only required by UI runtime (`app.py`):
- `streamlit`
- `plotly`

## Dev/test-only dependency
- `pytest`

## Import-coupling findings
- `app.py` imports both `streamlit` and `plotly.graph_objects` at module import time.
- `backend_api.py`, `planner_core.py`, `weather_ensemble.py`, and `db_sqlite.py` do **not** import `streamlit` or `plotly`.
- This indicates dependency coupling exists in packaging (`requirements.txt`), not in backend module imports.

## Practical implication
Backend runtime can be separated safely by dependency profile (without planner or ensemble logic changes), because backend import paths already avoid UI modules.
