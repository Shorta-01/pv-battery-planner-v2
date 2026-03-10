# Minimal actuals importer design

## Objective
Implement the smallest operational local importer for hourly actuals without FusionSolar integration.

## Script
- File: `scripts/import_actuals_csv.py`

## CLI interface
- Positional: `input_csv`
- Optional: `--source` (default `manual_csv`)
- Optional: `--api-base` (default `http://127.0.0.1:8787` or `PVBP_BACKEND_URL`)
- Optional: `--token` (fallback: `PVBP_API_TOKEN`, then `local_state/api_token.txt`)
- Optional: `--timeout`

## Validation flow (strict, pre-POST)
1. File exists.
2. CSV header equals exact required columns and order.
3. Each row timestamp matches `YYYY-MM-DDTHH:00:00` and parses as datetime.
4. Each numeric field parses as `float`.
5. If any row fails, exit with code `2` and clear error message.
6. No network call if validation fails.

## POST behavior
- Build payload exactly as:
  ```json
  {"source": "<source>", "rows": [ ...required-schema rows... ]}
  ```
- `POST {api_base}/v1/actuals/hourly`
- Headers:
  - `Authorization: Bearer <token>`
  - `Content-Type: application/json`

## Output behavior
- Success: `Import succeeded: rows_read=X rows_posted=Y source=Z`
- Request failure: includes status/body or request exception
- Return codes:
  - `0` success
  - `1` request-level failure
  - `2` local validation/input/token failures

## Why this is minimal
- Single script, no planner/ensemble touch.
- Mirrors existing backend contract as-is.
- Uses only local CSV and existing API.
