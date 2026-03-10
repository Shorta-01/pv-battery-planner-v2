# Evidence package

## 1) Files added/changed

### Added
- `requirements-backend.txt`
- `requirements-frontend.txt`
- `tests/test_runtime_split_imports.py`
- `portability_split_output/01_dependency_inventory.md`
- `portability_split_output/02_runtime_split_design.md`
- `portability_split_output/03_backend_import_safety.md`
- `portability_split_output/04_evidence.md`

### Changed
- `requirements.txt` (now umbrella profile)

## 2) Backend-only install command (Raspberry Pi 3)
```bash
pip install -r requirements-backend.txt
```

## 3) Frontend+laptop install command (full runtime)
```bash
pip install -r requirements-frontend.txt
```

Optional dev umbrella:
```bash
pip install -r requirements.txt
```

## 4) Test/smoke output
Command run:
```bash
pytest -q tests/test_runtime_split_imports.py
```
Observed output:
```text
..                                                                       [100%]
2 passed in 1.90s
```
