import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_madrid_short_horizon_excludes_dwd_icon_d2() -> None:
    candidates = we.get_candidate_models_for_location(latitude=40.4168, longitude=-3.7038, horizon_hours=24)
    assert "dwd_icon_d2" not in candidates


def test_brussels_short_horizon_keeps_regional_models() -> None:
    candidates = we.get_candidate_models_for_location(latitude=50.8503, longitude=4.3517, horizon_hours=24)
    assert "knmi_harmonie_arome" in candidates
    assert "dwd_icon_d2" in candidates


def test_sydney_short_horizon_global_only() -> None:
    candidates = we.get_candidate_models_for_location(latitude=-33.8688, longitude=151.2093, horizon_hours=24)
    assert candidates
    assert all(we.MODEL_COVERAGE_HINTS.get(m, "global") == "global" for m in candidates)
