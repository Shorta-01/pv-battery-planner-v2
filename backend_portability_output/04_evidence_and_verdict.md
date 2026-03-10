# Evidence and verdict

## 1) Exact files added/changed
Added:
- `scripts/start_backend.py`
- `scripts/smoke_backend_runtime.py`
- `backend_portability_output/01_startup_requirements.md`
- `backend_portability_output/02_pi3_blockers.md`
- `backend_portability_output/03_smoke_checks.md`
- `backend_portability_output/04_evidence_and_verdict.md`

Changed:
- `README.md`

## 2) Backend start command for Windows
Recommended backend-only path:
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements-backend.txt
python scripts/start_backend.py
```

Optional overrides:
```powershell
$env:PVBP_BACKEND_HOST="127.0.0.1"
$env:PVBP_BACKEND_PORT="8787"
python scripts/start_backend.py
```

## 3) Backend start command for Raspberry Pi 3
Backend-only path:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-backend.txt
python scripts/start_backend.py
```

For LAN/API access from other devices:
```bash
PVBP_BACKEND_HOST=0.0.0.0 PVBP_BACKEND_PORT=8787 python scripts/start_backend.py
```

## 4) Remaining risks NOT PROVEN
- Pi 3 dependency install success is not proven in this environment (ARM-specific wheel/build behavior not exercised here).
- Pi 3 forecast runtime latency under real workload not benchmarked here.
- End-to-end networked client calls against Pi-hosted backend not proven here.

## 5) Ready for real laptop testing?
- **Yes.**
- Backend-only install/startup path is explicit and smoke-checked.
- Backend import/db init path validated.
- Frontend dependency is not required for backend startup.

## 6) Ready for real Pi 3 testing?
- **Yes, with expected dependency/performance caveats.**
- The path is practical enough to test now:
  - backend-only dependencies
  - explicit startup helper
  - no frontend startup requirement
- Remaining uncertainty is operational on-device packaging/performance, not architecture.

## Change scope confirmation
- No planner logic changed.
- No ensemble logic changed.
- No API contract changes introduced.
- Changes are startup/runtime/documentation focused for backend portability.
