# Raspberry Pi 3 practical blockers and risks

## 1) Heavy packages still present in backend requirements
Backend profile currently includes:
- `numpy`
- `pandas`
- `pvlib`
- `fastapi`
- `uvicorn`
- `requests`

Most significant on Pi 3:
- `numpy`, `pandas`, `pvlib` (install and runtime cost)

## 2) Wheel/build availability risk
- On Pi 3 (ARMv7, older OS/Python combos), prebuilt wheels may be missing for pinned ranges, especially for `numpy`/`pandas`.
- If wheels are unavailable, source builds can be slow and memory-intensive, and may fail without system build toolchain.

## 3) Memory/CPU risk
- Ensemble/weather+PV processing is computationally non-trivial.
- Pi 3 (limited CPU/RAM) may have noticeably slower run latency.
- Uvicorn + heavy forecast work can still run, but concurrency should be kept low and expectations set for slower responses.

## 4) Backend-only viability
- Yes: backend startup/import path does not require Streamlit/Plotly.
- The backend can be installed from `requirements-backend.txt` only and started independently.

## 5) Optional packages that can be avoided
- For backend-only deployment, avoid installing frontend packages:
  - `streamlit`
  - `plotly`
- `requirements-frontend.txt` should be skipped on Pi backend-only setups.

## 6) Import-path drag-in risk
- Startup/import inspection indicates backend modules do not import frontend modules.
- Existing runtime split tests and smoke check confirm `streamlit` is not required for backend module import/startup.

## Practical Pi 3 conclusion
- **Main blockers are operational/dependency-related**, not backend architecture.
- Backend-only path is practical enough to proceed to real-device testing using the new startup helper and backend dependency profile.
