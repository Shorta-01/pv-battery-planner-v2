import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_belgium_week_ahead_selection_is_deterministic_and_global() -> None:
    models_a = we.select_week_ahead_models(requested_days=7, lat=50.85, lon=4.35)
    models_b = we.select_week_ahead_models(requested_days=7, lat=50.85, lon=4.35)
    assert models_a == models_b
    assert models_a[:3] == ["ecmwf_ifs", "ecmwf_aifs", "gfs"]
