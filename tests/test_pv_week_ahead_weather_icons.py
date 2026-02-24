import datetime as dt

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api


class _FR:
    def __init__(self, df: pd.DataFrame):
        self.df = df


def _day_df(tz: str, midday_code: int, *, fill_code: int = 3, coverage_hours: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2026-01-10 00:00", periods=24, freq="h", tz=tz)
    values = [float("nan")] * 24
    for i in range(max(0, min(24, coverage_hours))):
        values[i] = fill_code
    if coverage_hours > 12:
        values[12] = midday_code
    return pd.DataFrame({"weather_code": values}, index=idx)


def _base_build_kwargs(weather_by_model: dict[str, object], weights: dict[str, float], derived: dict[str, bool] | None = None):
    return dict(
        target_date=dt.date(2026, 1, 10),
        tz="Europe/Brussels",
        pv_totals_p50=[1.0] * 7,
        pv_totals_p10=[0.8] * 7,
        pv_totals_p90=[1.2] * 7,
        weather_by_model=weather_by_model,
        weights_used=weights,
        weather_primary_model_id="ecmwf_ifs",
        derived_weather_code_by_model=derived or {},
        weather_ensemble_df=None,
    )


def test_week_icon_vote_is_deterministic_for_same_inputs() -> None:
    tz = "Europe/Brussels"
    target_date = dt.date(2026, 1, 10)
    weather_by_model = {
        "ecmwf_ifs": _FR(_day_df(tz, midday_code=61)),
        "dwd_icon_d2": _FR(_day_df(tz, midday_code=3)),
        "metno_nordic": _FR(_day_df(tz, midday_code=61)),
    }
    kwargs = dict(
        day_offset=0,
        target_date=target_date,
        tz=tz,
        weather_by_model=weather_by_model,
        weights_used={"ecmwf_ifs": 1.0, "dwd_icon_d2": 1.0, "metno_nordic": 1.0},
        primary_id="ecmwf_ifs",
        derived_weather_code_by_model={},
    )

    first = backend_api._pick_week_ahead_weather_code_vote(**kwargs)
    second = backend_api._pick_week_ahead_weather_code_vote(**kwargs)

    assert first == second
    assert first[0] == 61


def test_week_icon_vote_stable_when_non_winning_model_missing() -> None:
    tz = "Europe/Brussels"
    target_date = dt.date(2026, 1, 10)
    full_models = {
        "ecmwf_ifs": _FR(_day_df(tz, midday_code=61)),
        "metno_nordic": _FR(_day_df(tz, midday_code=61)),
        "dwd_icon_d2": _FR(_day_df(tz, midday_code=3)),
    }
    reduced_models = {
        "ecmwf_ifs": full_models["ecmwf_ifs"],
        "metno_nordic": full_models["metno_nordic"],
    }

    full_code, _ = backend_api._pick_week_ahead_weather_code_vote(
        0,
        target_date=target_date,
        tz=tz,
        weather_by_model=full_models,
        weights_used={"ecmwf_ifs": 1.0, "metno_nordic": 1.0, "dwd_icon_d2": 0.5},
        primary_id="ecmwf_ifs",
        derived_weather_code_by_model={},
    )
    reduced_code, _ = backend_api._pick_week_ahead_weather_code_vote(
        0,
        target_date=target_date,
        tz=tz,
        weather_by_model=reduced_models,
        weights_used={"ecmwf_ifs": 1.0, "metno_nordic": 1.0},
        primary_id="ecmwf_ifs",
        derived_weather_code_by_model={},
    )

    assert full_code == reduced_code == 61


def test_week_icon_vote_prefers_non_derived_when_votes_close() -> None:
    tz = "Europe/Brussels"
    target_date = dt.date(2026, 1, 10)
    weather_by_model = {
        "ecmwf_ifs": _FR(_day_df(tz, midday_code=3)),
        "metno_nordic": _FR(_day_df(tz, midday_code=61)),
    }

    code, _ = backend_api._pick_week_ahead_weather_code_vote(
        0,
        target_date=target_date,
        tz=tz,
        weather_by_model=weather_by_model,
        weights_used={"ecmwf_ifs": 1.0, "metno_nordic": 1.0},
        primary_id=None,
        derived_weather_code_by_model={"metno_nordic": True},
    )

    assert code == 3


def test_build_pv_week_ahead_sets_multi_model_policy_fields() -> None:
    weather_by_model = {
        "ecmwf_ifs": _FR(_day_df("Europe/Brussels", midday_code=61)),
        "dwd_icon_d2": _FR(_day_df("Europe/Brussels", midday_code=3)),
    }

    out = backend_api._build_pv_week_ahead(
        **_base_build_kwargs(
            weather_by_model=weather_by_model,
            weights={"ecmwf_ifs": 1.0, "dwd_icon_d2": 1.0},
        )
    )

    day0 = out[0]
    assert day0["weather_code_selection_policy"] == "multi_model_weighted_vote"
    assert day0["weather_code_source_model_id"] == "multi_model_vote"
    assert isinstance(day0["weather_icon_models_considered"], int)
    assert isinstance(day0["weather_icon_models_used"], int)
    assert day0["weather_icon_models_used"] >= 1


def test_regression_multi_model_vote_stable_when_primary_disappears() -> None:
    tz = "Europe/Brussels"
    target_date = dt.date(2026, 1, 10)

    cold_models = {
        "ecmwf_ifs": _FR(_day_df(tz, midday_code=3)),
        "metno_nordic": _FR(_day_df(tz, midday_code=61)),
        "dwd_icon_d2": _FR(_day_df(tz, midday_code=61)),
    }
    warm_models = {
        "metno_nordic": cold_models["metno_nordic"],
        "dwd_icon_d2": cold_models["dwd_icon_d2"],
    }
    weights_all = {"ecmwf_ifs": 1.0, "metno_nordic": 1.0, "dwd_icon_d2": 1.0}

    old_cold = backend_api._pick_week_ahead_weather_code(
        0,
        target_date=target_date,
        tz=tz,
        weather_by_model=cold_models,
        weights_used=weights_all,
        primary_id="ecmwf_ifs",
        derived_weather_code_by_model={},
    )[0]
    old_warm = backend_api._pick_week_ahead_weather_code(
        0,
        target_date=target_date,
        tz=tz,
        weather_by_model=warm_models,
        weights_used=weights_all,
        primary_id="ecmwf_ifs",
        derived_weather_code_by_model={},
    )[0]

    new_cold = backend_api._pick_week_ahead_weather_code_vote(
        0,
        target_date=target_date,
        tz=tz,
        weather_by_model=cold_models,
        weights_used=weights_all,
        primary_id="ecmwf_ifs",
        derived_weather_code_by_model={},
    )[0]
    new_warm = backend_api._pick_week_ahead_weather_code_vote(
        0,
        target_date=target_date,
        tz=tz,
        weather_by_model=warm_models,
        weights_used=weights_all,
        primary_id="ecmwf_ifs",
        derived_weather_code_by_model={},
    )[0]

    assert old_cold != old_warm
    assert new_cold == new_warm == 61
