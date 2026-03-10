# Minimal runtime split design

## Objective
Enable two deployment profiles with minimal change:
1. **Backend-only (Raspberry Pi 3)**
2. **Frontend + backend (Windows laptop)**

## Minimal design

### 1) Add backend requirements profile
Create `requirements-backend.txt` containing only backend/core runtime dependencies:
- `requests`
- `numpy`
- `pandas`
- `pvlib`
- `fastapi`
- `uvicorn`

### 2) Add frontend requirements profile
Create `requirements-frontend.txt` that layers frontend deps over backend deps:
- `-r requirements-backend.txt`
- `streamlit`
- `plotly`

### 3) Keep umbrella requirements file
Retain `requirements.txt` as a convenience umbrella for full laptop/dev installs:
- `-r requirements-frontend.txt`
- `pytest`

## Why this is minimal and safe
- No changes to planner logic.
- No changes to ensemble logic.
- No API contract changes.
- No startup-path refactor required because backend modules already avoid UI imports.
- Only dependency-profile packaging is split.

## Deployment matrix
- **Raspberry Pi 3 (backend-only):** install `requirements-backend.txt`, run FastAPI/uvicorn backend.
- **Windows laptop (full):** install `requirements-frontend.txt` (or umbrella `requirements.txt`) and run backend + Streamlit frontend.
