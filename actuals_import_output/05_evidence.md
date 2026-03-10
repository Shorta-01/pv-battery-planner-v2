# Evidence

## 1) Files added/changed
- Added `scripts/import_actuals_csv.py`
- Added `tests/test_import_actuals_csv.py`
- Added `actuals_import_output/01_contract_check.md`
- Added `actuals_import_output/02_importer_design.md`
- Added `actuals_import_output/03_tests_and_validation.md`
- Added `actuals_import_output/04_operator_usage.md`
- Added `actuals_import_output/05_evidence.md`
- Added `scripts/__init__.py`

## 2) Example command
```bash
python scripts/import_actuals_csv.py ./data/actuals_hourly.csv --source manual_csv --api-base http://127.0.0.1:8787
```

## 3) Test output
```text
$ pytest -q tests/test_import_actuals_csv.py
.....                                                                    [100%]
5 passed in 0.11s
```

## 4) Example success output
```text
Import succeeded: rows_read=24 rows_posted=24 source=manual_csv
```
