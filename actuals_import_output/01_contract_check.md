# `/v1/actuals/hourly` contract check

## Endpoint and auth
- Route: `POST /v1/actuals/hourly`
- Requires Bearer auth via `_require_token`; missing/invalid token returns `401`.

## Accepted request formats

### 1) CSV (`Content-Type: text/csv`)
- Server parses with `csv.DictReader`.
- CSV header must be **exactly**:
  `ts_local,pv_kwh,load_kwh,grid_import_kwh,grid_export_kwh,soc_pct`
- Any header mismatch returns `400` with detail:
  `CSV headers must be exactly: ...`
- For CSV ingest, source defaults to `manual_csv`.

### 2) JSON
- List payload accepted: `[row, ...]` (uses default source `manual_csv`).
- Object payload accepted: `{"source": "...", "rows": [row, ...]}`.
- If object payload omits `rows[]`, server returns `400`.

## Required row schema
Rows are validated strictly in DB layer (`normalize_actual_hourly_row(strict=True)`):
- Required keys (exact set, no extras):
  - `ts_local`
  - `pv_kwh`
  - `load_kwh`
  - `grid_import_kwh`
  - `grid_export_kwh`
  - `soc_pct`
- Missing or extra keys raise `ValueError("Invalid actual row columns: ...")`, surfaced as `400`.
- Non-object rows raise `ValueError("Each actual row must be an object")`, surfaced as `400`.

## Timestamp expectations
- `ts_local` is normalized by `_to_local_hour_str` and stored as `%Y-%m-%dT%H:00:00`.
- Invalid/unparseable timestamps raise `ValueError("Invalid ts_local value")`, surfaced as `400`.
- Input can be parseable datetime-like, but canonical stored shape is hourly local timestamp string.

## Source behavior
- Default source: `manual_csv`.
- JSON object payload can override with `source`.
- Insert path uses `(source, ts_local)` as primary key in `actual_hourly` (`INSERT OR REPLACE`).
- Response shape: `{"inserted": <int>, "source": <source>}`.

## Importer constraints to obey
1. Ensure CSV header matches exact expected columns/order.
2. Ensure every row has exactly the required fields.
3. Validate timestamps to canonical hourly local format (`YYYY-MM-DDTHH:00:00`) before POST.
4. Validate numeric columns before POST.
5. POST JSON object payload (`source`, `rows`) to preserve optional source behavior.
6. Fail fast and avoid any POST on local validation failure.
