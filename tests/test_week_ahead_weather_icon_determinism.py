import datetime as dt

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api


class _FR:
    def __init__(self, df: pd.DataFrame):
        self.df = df


def _day_df(tz: str, midday_code: int, alt_code: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2026-01-10 00:00", periods=24, freq="h", tz=tz)
    values = [alt_code] * 24
    values[12] = midday_code
    return pd.DataFrame({"weather_code": values}, index=idx)


def test_build_pv_week_ahead_prefers_ensemble_weather_code_when_available() -> None:
    tz = "Europe/Brussels"
    target_date = dt.date(2026, 1, 10)
    idx = pd.date_range("2026-01-10 00:00", periods=24 * 7, freq="h", tz=tz)
    ensemble_df = pd.DataFrame({"weather_code": [3] * len(idx)}, index=idx)
    day_mask = (ensemble_df.index.date == target_date) & (ensemble_df.index.hour >= 8) & (ensemble_df.index.hour <= 18)
    ensemble_df.loc[day_mask, "weather_code"] = 61

    weather_by_model = {
        "ecmwf_ifs": _FR(_day_df(tz, midday_code=0)),
    }

    out = backend_api._build_pv_week_ahead(
        target_date=target_date,
        tz=tz,
        pv_totals_p50=[1.0] * 7,
        pv_totals_p10=[0.5] * 7,
        pv_totals_p90=[1.5] * 7,
        weather_by_model=weather_by_model,
        weights_used={"ecmwf_ifs": 1.0},
        weather_primary_model_id="ecmwf_ifs",
        weather_ensemble_df=ensemble_df,
    )

    assert out[0]["weather_code"] == 61
    assert out[0]["weather_code_source_model_id"] == "ensemble_weather"
    assert out[0]["weather_code_source_model_label"] == "Ensemble weather"


def test_build_pv_week_ahead_falls_back_to_source_model_picker_when_ensemble_missing() -> None:
    tz = "Europe/Brussels"
    target_date = dt.date(2026, 1, 10)
    weather_by_model = {
        "ecmwf_ifs": _FR(_day_df(tz, midday_code=2)),
    }

    out = backend_api._build_pv_week_ahead(
        target_date=target_date,
        tz=tz,
        pv_totals_p50=[1.0] * 7,
        pv_totals_p10=[0.5] * 7,
        pv_totals_p90=[1.5] * 7,
        weather_by_model=weather_by_model,
        weights_used={"ecmwf_ifs": 1.0},
        weather_primary_model_id="ecmwf_ifs",
        weather_ensemble_df=None,
    )

    assert out[0]["weather_code"] == 2
    assert out[0]["weather_code_source_model_id"] == "multi_model_vote"
    assert out[0]["weather_code_selection_policy"] == "multi_model_weighted_vote"


def test_pick_week_ahead_weather_code_is_order_independent_for_equal_candidates() -> None:
    tz = "Europe/Brussels"
    target_date = dt.date(2026, 1, 10)
    a = _FR(_day_df(tz, midday_code=3))
    b = _FR(_day_df(tz, midday_code=3))

    res1 = backend_api._pick_week_ahead_weather_code(
        0,
        target_date=target_date,
        tz=tz,
        weather_by_model={"ecmwf_ifs": a, "dwd_icon_d2": b},
        weights_used={"ecmwf_ifs": 1.0, "dwd_icon_d2": 1.0},
        primary_id=None,
        derived_weather_code_by_model={},
    )
    res2 = backend_api._pick_week_ahead_weather_code(
        0,
        target_date=target_date,
        tz=tz,
        weather_by_model={"dwd_icon_d2": b, "ecmwf_ifs": a},
        weights_used={"ecmwf_ifs": 1.0, "dwd_icon_d2": 1.0},
        primary_id=None,
        derived_weather_code_by_model={},
    )

    assert res1 == res2


def test_week_ahead_output_kwh_values_unchanged_by_weather_icon_source_policy() -> None:
    tz = "Europe/Brussels"
    target_date = dt.date(2026, 1, 10)
    totals_p50 = [float(i) for i in range(1, 8)]
    totals_p10 = [float(i) - 0.3 for i in range(1, 8)]
    totals_p90 = [float(i) + 0.3 for i in range(1, 8)]
    weather_by_model = {"ecmwf_ifs": _FR(_day_df(tz, midday_code=0))}

    no_ensemble = backend_api._build_pv_week_ahead(
        target_date=target_date,
        tz=tz,
        pv_totals_p50=totals_p50,
        pv_totals_p10=totals_p10,
        pv_totals_p90=totals_p90,
        weather_by_model=weather_by_model,
        weights_used={"ecmwf_ifs": 1.0},
        weather_primary_model_id="ecmwf_ifs",
        weather_ensemble_df=None,
    )

    idx = pd.date_range("2026-01-10 00:00", periods=24 * 7, freq="h", tz=tz)
    ensemble_df = pd.DataFrame({"weather_code": [61] * len(idx)}, index=idx)
    with_ensemble = backend_api._build_pv_week_ahead(
        target_date=target_date,
        tz=tz,
        pv_totals_p50=totals_p50,
        pv_totals_p10=totals_p10,
        pv_totals_p90=totals_p90,
        weather_by_model=weather_by_model,
        weights_used={"ecmwf_ifs": 1.0},
        weather_primary_model_id="ecmwf_ifs",
        weather_ensemble_df=ensemble_df,
    )

    for i in range(7):
        assert no_ensemble[i]["p50_kwh"] == with_ensemble[i]["p50_kwh"]
        assert no_ensemble[i]["p10_kwh"] == with_ensemble[i]["p10_kwh"]
        assert no_ensemble[i]["p90_kwh"] == with_ensemble[i]["p90_kwh"]
