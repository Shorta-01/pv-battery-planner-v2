import datetime as dt

import pandas as pd

import backend_api
import weather_ensemble as we


def test_derived_hours_penalty() -> None:
    idx = pd.date_range("2026-01-01", periods=4, freq="h")
    s_a = pd.Series([1.0, 1.1, 1.2, 1.3], index=idx)
    s_b = pd.Series([1.0, 1.1, 1.2, 1.3], index=idx)

    _, weights_used, quality_factors = we._weighted_ensemble(
        {"ecmwf_ifs": s_a, "dwd_icon_eu": s_b},
        ["ecmwf_ifs", "dwd_icon_eu"],
        dynamic_weights={"ecmwf_ifs": 0.5, "dwd_icon_eu": 0.5},
        derived_irradiance_hours_by_model={"ecmwf_ifs": 0, "dwd_icon_eu": 12},
    )

    assert weights_used is not None
    assert quality_factors["dwd_icon_eu"] < quality_factors["ecmwf_ifs"]
    assert weights_used["dwd_icon_eu"] < weights_used["ecmwf_ifs"]


def test_cache_key_includes_elevation() -> None:
    day = dt.date(2026, 1, 2)
    key_none = we._cache_key("ecmwf_ifs", 50.85, 4.35, "Europe/Brussels", day, elevation_m=None)
    key_120 = we._cache_key("ecmwf_ifs", 50.85, 4.35, "Europe/Brussels", day, elevation_m=120.0)

    assert key_none != key_120


def test_parse_elevation_payload() -> None:
    assert backend_api._parse_elevation_m({"elevation": [123.4]}) == 123.4
    assert backend_api._parse_elevation_m({"elevation": 123.4}) == 123.4
    assert backend_api._parse_elevation_m({"elevation": []}) is None
    assert backend_api._parse_elevation_m({"elevation": "bad"}) is None
    assert backend_api._parse_elevation_m({"foo": 1}) is None
