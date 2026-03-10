# Tests and validation

## Added focused tests
File: `tests/test_import_actuals_csv.py`

1. `test_valid_csv_with_exact_schema`
   - Valid CSV with exact columns and one row parses correctly.
2. `test_missing_required_column`
   - Header mismatch raises clear validation error.
3. `test_malformed_timestamp`
   - Bad timestamp shape rejected before POST.
4. `test_successful_request_payload_generation`
   - Verifies exact URL, auth header, content-type, and generated payload body.
5. `test_non_destructive_behavior_on_bad_input`
   - Ensures `main()` exits with validation error and never attempts POST.

## Command used
- `pytest -q tests/test_import_actuals_csv.py`

## Result
- `5 passed`
