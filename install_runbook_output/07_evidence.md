# 07 — Evidence

## 1) Exact files added/changed

Added:
- `install_runbook_output/01_current_run_paths.md`
- `install_runbook_output/02_windows_laptop_runbook.md`
- `install_runbook_output/03_raspberry_pi3_backend_runbook.md`
- `install_runbook_output/04_daily_operations.md`
- `install_runbook_output/05_troubleshooting.md`
- `install_runbook_output/06_quick_start.md`
- `install_runbook_output/07_evidence.md`

No planner logic files changed.
No ensemble logic files changed.
No backend/frontend core behavior changed.

---

## 2) Exact commands validated in this environment

Validated command:

```bash
python --version && python scripts/smoke_backend_runtime.py
```

Observed result:
- Python 3.10.19
- PASS import backend_api
- PASS sqlite initialized
- PASS token created
- PASS streamlit not imported during backend startup

---

## 3) Commands not validated here (and why)

Not validated due environment mismatch (this environment is Linux CI container, not actual Windows laptop / Raspberry Pi runtime target):
- Windows PowerShell activation commands (`.\\.venv\\Scripts\\Activate.ps1`)
- Windows `Invoke-RestMethod` checks
- Streamlit browser UX connectivity checks
- Raspberry Pi apt/package/wheel behavior on ARM hardware
- LAN exposure checks (`PVBP_BACKEND_HOST=0.0.0.0`) on real Pi network

Not validated to avoid long-running foreground process in this task session:
- Full backend uvicorn server lifecycle with manual Ctrl+C interaction
- End-to-end real forecast run + real actuals import + daily score against user data

---

## 4) Remaining risks NOT PROVEN yet

1. **Windows local policy friction**
   - PowerShell execution policy may block venv activation script on some machines.

2. **Raspberry Pi package/wheel variability**
   - Some dependency builds can be slower/fail depending on OS image and available build tools.

3. **Network/firewall constraints**
   - LAN access (`0.0.0.0`) may still be blocked by host firewall/router policy.

4. **Token handling in multi-user setups**
   - `local_state/api_token.txt` file permissions may need tightening in shared environments.

5. **Operational data quality**
   - Actuals CSV formatting/timezone alignment can still cause ingest failures despite runbook validation steps.

