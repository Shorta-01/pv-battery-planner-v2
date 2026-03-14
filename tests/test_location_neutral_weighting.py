import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_location_neutral_default_weight_alias_kept() -> None:
    assert we.DEFAULT_WEIGHTED_AUTO == we.DEFAULT_WEIGHTED_BELGIUM


def test_week_selection_not_using_recommended_for_be_flag() -> None:
    selected = we.select_week_ahead_models(requested_days=7, lat=10.0, lon=10.0)
    assert selected
    assert selected[0] in {"ecmwf_ifs", "ecmwf_aifs", "gfs"}
