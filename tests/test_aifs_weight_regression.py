import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_aifs_has_explicit_base_weight() -> None:
    assert "ecmwf_aifs" in we.DEFAULT_WEIGHTED_AUTO
    assert we.DEFAULT_WEIGHTED_AUTO["ecmwf_aifs"] < 1.0


def test_unknown_weight_fallback_is_conservative() -> None:
    assert we._base_weight_for_model("unknown_model_xyz") == we.UNKNOWN_MODEL_WEIGHT_FALLBACK
    assert we._base_weight_for_model("unknown_model_xyz") < we.DEFAULT_WEIGHTED_AUTO["ecmwf_ifs"]


def test_day_type_weights_do_not_oversize_aifs() -> None:
    weights = we._weights_for_day_type(
        base_weights=None,
        model_ids=["ecmwf_ifs", "ecmwf_aifs", "gfs"],
        day_type="variable_cloudy",
        expert_mode=False,
    )
    assert weights["ecmwf_ifs"] > weights["ecmwf_aifs"] > weights["gfs"]
