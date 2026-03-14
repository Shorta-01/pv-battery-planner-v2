import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_brussels_tomorrow_auto_includes_global_anchor() -> None:
    selected, _reason = we.auto_select_models_for_location_and_horizon(
        latitude=50.8503,
        longitude=4.3517,
        horizon_hours=24,
    )
    assert any(we.MODEL_COVERAGE_HINTS.get(m, "global") == "global" for m in selected)


def test_brussels_short_horizon_keeps_ifs_anchor() -> None:
    selected, _reason = we.auto_select_models_for_location_and_horizon(
        latitude=50.8503,
        longitude=4.3517,
        horizon_hours=24,
    )
    assert "ecmwf_ifs" in selected


def test_global_only_short_horizon_prefers_ifs_over_aifs_and_gfs() -> None:
    selected, _reason = we.auto_select_models_for_location_and_horizon(
        latitude=-33.8688,
        longitude=151.2093,
        horizon_hours=24,
    )
    assert selected[0] == "ecmwf_ifs"
