import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_stable_available_model_order_respects_selected_then_sorted_extras() -> None:
    selected = ["dwd_icon_d2", "ecmwf_ifs"]
    mapping = {
        "gfs": object(),
        "ecmwf_ifs": object(),
        "knmi_harmonie_arome": object(),
        "dwd_icon_d2": object(),
    }

    out = we._stable_available_model_order(selected, mapping)

    assert out == ["dwd_icon_d2", "ecmwf_ifs", "gfs", "knmi_harmonie_arome"]


def test_stable_first_available_model_ignores_dict_insertion_order() -> None:
    selected = ["ecmwf_ifs", "dwd_icon_d2"]
    mapping_a = {"dwd_icon_d2": object(), "ecmwf_ifs": object()}
    mapping_b = {"ecmwf_ifs": object(), "dwd_icon_d2": object()}

    assert we._stable_first_available_model(selected, mapping_a) == "ecmwf_ifs"
    assert we._stable_first_available_model(selected, mapping_b) == "ecmwf_ifs"


def test_stable_first_available_model_fallback_is_sorted() -> None:
    selected = ["ecmwf_ifs"]
    mapping = {"knmi_harmonie_arome": object(), "dwd_icon_d2": object()}

    assert we._stable_first_available_model(selected, mapping) == "dwd_icon_d2"
