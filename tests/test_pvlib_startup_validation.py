import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import weather_ensemble as we


def test_validate_pvlib_runtime_passes_when_available(monkeypatch) -> None:
    monkeypatch.setattr(we.core, "PVLIB_AVAILABLE", True)
    assert we.validate_pvlib_runtime(require_production_quality=True) is True


def test_validate_pvlib_runtime_fails_for_production(monkeypatch) -> None:
    monkeypatch.setattr(we.core, "PVLIB_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        we.validate_pvlib_runtime(require_production_quality=True)
