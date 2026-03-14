import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_auto_tomorrow_prefers_regional_in_benelux() -> None:
    models = we.auto_select_models_for_location(50.85, 4.35, requested_days=1)
    assert models
    assert models[0] in {"knmi_harmonie_arome", "dwd_icon_d2", "dwd_icon_eu"}


def test_auto_tomorrow_global_fallback_outside_regional_domain() -> None:
    models = we.auto_select_models_for_location(-33.86, 151.21, requested_days=1)
    assert models
    assert models[0] in {"ecmwf_ifs", "ecmwf_aifs", "gfs"}


def test_auto_selection_deterministic_order() -> None:
    a = we.auto_select_models_for_location(48.86, 2.35, requested_days=2)
    b = we.auto_select_models_for_location(48.86, 2.35, requested_days=2)
    assert a == b
