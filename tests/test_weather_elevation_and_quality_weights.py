import datetime as dt

import pandas as pd

import backend_api
import weather_ensemble as we


def test_quality_weights_penalize_derived_hours() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    s_a = pd.Series([1.0, 1.2, 1.4], index=idx)
    s_b = pd.Series([1.0, 1.2, 1.4], index=idx)

    _, weights_used, quality_factors = we._weighted_ensemble(
        {"ecmwf_ifs": s_a, "dwd_icon_eu": s_b},
        ["ecmwf_ifs", "dwd_icon_eu"],
        dynamic_weights={"ecmwf_ifs": 0.5, "dwd_icon_eu": 0.5},
        derived_irradiance_by_model={"ecmwf_ifs": False, "dwd_icon_eu": False},
        derived_irradiance_hours_by_model={"ecmwf_ifs": 0, "dwd_icon_eu": 12},
    )

    assert weights_used is not None
    assert quality_factors["dwd_icon_eu"] < quality_factors["ecmwf_ifs"]
    assert weights_used["dwd_icon_eu"] < weights_used["ecmwf_ifs"]


def test_quality_weights_penalize_missing_wet_day_signals() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    s_a = pd.Series([1.0, 1.2, 1.4], index=idx)
    s_b = pd.Series([1.0, 1.2, 1.4], index=idx)

    _, weights_used, quality_factors = we._weighted_ensemble(
        {"ecmwf_ifs": s_a, "dwd_icon_eu": s_b},
        ["ecmwf_ifs", "dwd_icon_eu"],
        dynamic_weights={"ecmwf_ifs": 0.5, "dwd_icon_eu": 0.5},
        missing_vars_by_model={"dwd_icon_eu": ["precipitation_probability", "precipitation", "rain"]},
        derived_weather_code_by_model={"dwd_icon_eu": True},
    )
    assert weights_used is not None
    assert quality_factors["dwd_icon_eu"] < quality_factors["ecmwf_ifs"]
    assert weights_used["dwd_icon_eu"] < weights_used["ecmwf_ifs"]


def test_cache_key_includes_elevation() -> None:
    day = dt.date(2026, 1, 2)
    key_low = we._cache_key("ecmwf_ifs", 50.85, 4.35, "Europe/Brussels", day, elevation_m=10.0)
    key_high = we._cache_key("ecmwf_ifs", 50.85, 4.35, "Europe/Brussels", day, elevation_m=210.0)

    assert key_low != key_high


def test_elevation_parse_helper() -> None:
    assert backend_api._parse_elevation_m({"elevation": 54.2}) == 54.2
    assert backend_api._parse_elevation_m({"elevation": [123.0]}) == 123.0
    assert backend_api._parse_elevation_m({"elevation": []}) is None
    assert backend_api._parse_elevation_m({"elevation": "oops"}) is None
    assert backend_api._parse_elevation_m({"foo": 1}) is None
    assert backend_api._parse_elevation_m(None) is None
