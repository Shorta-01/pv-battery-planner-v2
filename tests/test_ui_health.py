from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_health import model_indicators, should_hard_stop


def test_should_hard_stop_error() -> None:
    result = {
        "status": "error",
        "warnings": ["all weather model requests failed"],
        "warnings_count": 1,
    }

    hard_stop, _, warnings = should_hard_stop(result)

    assert hard_stop is True
    assert "all weather model requests failed" in warnings


def test_should_hard_stop_ok() -> None:
    result = {"status": "ok", "warnings": [], "warnings_count": 0}

    hard_stop, _, _ = should_hard_stop(result)

    assert hard_stop is False


def test_model_indicators_not_selected() -> None:
    weather_ensemble = {"selected_models": ["a"], "failed_model_reasons": {}}

    indicators = model_indicators("b", weather_ensemble)

    assert indicators["fetch"] == "na"


def test_model_indicators_failed() -> None:
    weather_ensemble = {
        "selected_models": ["a"],
        "failed_model_reasons": {"a": "403 Forbidden"},
    }

    indicators = model_indicators("a", weather_ensemble)

    assert indicators["fetch"] == "fail"
    assert indicators["data"] == "fail"


def test_model_indicators_data_ok_strict() -> None:
    weather_ensemble = {
        "selected_models": ["a"],
        "failed_model_reasons": {},
        "fetch_meta_by_model": {"a": {"missing_hours_overlap": 0}},
        "missing_vars_by_model": {"a": []},
    }

    indicators = model_indicators("a", weather_ensemble)

    assert indicators["fetch"] == "ok"
    assert indicators["data"] == "ok"


def test_model_indicators_data_warn_when_overlap_nonzero() -> None:
    weather_ensemble = {
        "selected_models": ["a"],
        "failed_model_reasons": {},
        "fetch_meta_by_model": {"a": {"missing_hours_overlap": 1}},
        "missing_vars_by_model": {"a": []},
    }

    indicators = model_indicators("a", weather_ensemble)

    assert indicators["data"] == "warn"
