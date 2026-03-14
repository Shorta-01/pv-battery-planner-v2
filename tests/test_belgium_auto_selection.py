import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_belgium_tomorrow_auto_selection_is_strong_and_deterministic() -> None:
    models_a = we.auto_select_models_for_location(50.8503, 4.3517, requested_days=1)
    models_b = we.auto_select_models_for_location(50.8503, 4.3517, requested_days=1)
    assert models_a == models_b
    assert models_a[:4] == ["knmi_harmonie_arome", "dwd_icon_d2", "dwd_icon_eu", "ecmwf_ifs"]
    assert any(we.MODEL_COVERAGE_HINTS.get(m) == "global" for m in models_a)
