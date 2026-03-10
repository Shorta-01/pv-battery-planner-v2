# Backend import safety verification

## Verification approach
Added focused smoke tests in `tests/test_runtime_split_imports.py`.

### Test 1: backend import path with frontend deps blocked
- Inject a temporary import hook that raises `ImportError` for `streamlit*` and `plotly*`.
- Under that condition, import:
  - `planner_core`
  - `weather_ensemble`
  - `db_sqlite`
  - `backend_api`
- Expected result: all imports succeed.

### Test 2: frontend deps importable in full environment
- In normal environment, import:
  - `streamlit`
  - `plotly.graph_objects`
- Expected result: imports succeed.

## Result
`pytest -q tests/test_runtime_split_imports.py` passed:
- `2 passed`

## Conclusion
Backend startup paths do not require frontend-only modules (`streamlit`, `plotly`).
The runtime split is valid without changing planner or ensemble logic.
