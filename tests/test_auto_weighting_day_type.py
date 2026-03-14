import pandas as pd
import weather_ensemble as we
import planner_core as core


def test_classify_day_types():
    stable_cloud = pd.Series([10, 12, 11, 10, 9, 11])
    stable_code = pd.Series([0, 1, 1, 0, 1, 0])
    assert we.classify_day_type(stable_cloud, stable_code) == "stable_clear"

    variable_cloud = pd.Series([5, 80, 20, 90, 10, 85])
    variable_code = pd.Series([2, 3, 2, 3, 2, 3])
    assert we.classify_day_type(variable_cloud, variable_code) == "variable_cloudy"

    wet_cloud = pd.Series([90, 95, 88, 92, 96, 91])
    wet_code = pd.Series([61, 63, 65, 80, 81, 82])
    assert we.classify_day_type(wet_cloud, wet_code) == "fronty_wet"


def test_weights_sum_and_bias():
    base = {"knmi_harmonie_arome": 0.4, "dwd_icon_d2": 0.3, "gfs": 0.3}
    models = list(base)
    stable = we._weights_for_day_type(base, models, day_type="stable_clear", expert_mode=False)
    assert abs(sum(stable.values()) - 1.0) < 1e-9
    assert stable["knmi_harmonie_arome"] > base["knmi_harmonie_arome"]

    expert = we._weights_for_day_type(base, models, day_type="fronty_wet", expert_mode=True)
    assert abs(sum(expert.values()) - 1.0) < 1e-9
    assert expert["knmi_harmonie_arome"] == base["knmi_harmonie_arome"]


def test_ensemble_day_type_fronty_wet_with_daylight_rain() -> None:
    idx = pd.date_range(pd.Timestamp("2026-01-10 00:00:00", tz="Europe/Brussels"), periods=24, freq="h")
    wet_df = pd.DataFrame(
        {
            "cloud_cover_pct": [92.0] * 24,
            "weather_code": [61.0] * 24,
            "precip_probability_pct": [85.0] * 24,
            "precip_mm": [0.6] * 24,
            "rain_mm": [0.4] * 24,
        },
        index=idx,
    )
    weather = {
        "knmi_harmonie_arome": core.ForecastResult(df=wet_df, sunrise=idx[7].to_pydatetime(), sunset=idx[17].to_pydatetime()),
        "dwd_icon_eu": core.ForecastResult(df=wet_df, sunrise=idx[7].to_pydatetime(), sunset=idx[17].to_pydatetime()),
    }
    assert we._classify_day_type_from_ensemble(weather, idx) == "fronty_wet"


def test_ensemble_day_type_not_dependent_on_primary_model_order() -> None:
    idx = pd.date_range(pd.Timestamp("2026-01-10 00:00:00", tz="Europe/Brussels"), periods=24, freq="h")
    clear_df = pd.DataFrame(
        {
            "cloud_cover_pct": [10.0] * 24,
            "weather_code": [1.0] * 24,
            "precip_probability_pct": [5.0] * 24,
            "precip_mm": [0.0] * 24,
            "rain_mm": [0.0] * 24,
        },
        index=idx,
    )
    wet_df = pd.DataFrame(
        {
            "cloud_cover_pct": [95.0] * 24,
            "weather_code": [63.0] * 24,
            "precip_probability_pct": [90.0] * 24,
            "precip_mm": [0.8] * 24,
            "rain_mm": [0.6] * 24,
        },
        index=idx,
    )
    a = {
        "clear": core.ForecastResult(df=clear_df, sunrise=idx[7].to_pydatetime(), sunset=idx[17].to_pydatetime()),
        "wet": core.ForecastResult(df=wet_df, sunrise=idx[7].to_pydatetime(), sunset=idx[17].to_pydatetime()),
    }
    b = {
        "wet": core.ForecastResult(df=wet_df, sunrise=idx[7].to_pydatetime(), sunset=idx[17].to_pydatetime()),
        "clear": core.ForecastResult(df=clear_df, sunrise=idx[7].to_pydatetime(), sunset=idx[17].to_pydatetime()),
    }
    assert we._classify_day_type_from_ensemble(a, idx) == we._classify_day_type_from_ensemble(b, idx)
