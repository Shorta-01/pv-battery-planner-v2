import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
import weather_ensemble as we


def _forecast(df: pd.DataFrame) -> core.ForecastResult:
    sunrise = df.index[0].to_pydatetime()
    sunset = df.index[-1].to_pydatetime()
    return core.ForecastResult(df=df, sunrise=sunrise, sunset=sunset)


def test_build_weather_ensemble_table_median_and_min_max() -> None:
    index = pd.date_range("2026-01-10 00:00:00", periods=3, freq="h", tz="Europe/Brussels")
    model_a = pd.DataFrame(
        {
            "temperature_2m": [10.0, 12.0, 14.0],
            "wind_speed_10m": [1.0, 2.0, 3.0],
            "cloud_cover": [10.0, 20.0, 30.0],
            "shortwave_radiation": [100.0, 200.0, 300.0],
            "direct_normal_irradiance": [50.0, 60.0, 70.0],
            "diffuse_radiation": [20.0, 30.0, 40.0],
        },
        index=index,
    )
    model_b = pd.DataFrame(
        {
            "temperature_2m": [14.0, 10.0, 16.0],
            "wind_speed_10m": [3.0, 2.0, 1.0],
            "cloud_cover": [30.0, 20.0, 10.0],
            "shortwave_radiation": [300.0, 200.0, 100.0],
            "direct_normal_irradiance": [70.0, 60.0, 50.0],
            "diffuse_radiation": [40.0, 30.0, 20.0],
        },
        index=index,
    )

    out = we.build_weather_ensemble_table(
        weather_ok={"a": _forecast(model_a), "b": _forecast(model_b)},
        index=index,
        ensemble_method="median",
        weights=None,
    )

    assert list(out.index) == list(index)
    assert out["temperature_2m"].tolist() == pytest.approx([12.0, 11.0, 15.0])
    assert out["temperature_2m_min"].tolist() == pytest.approx([10.0, 10.0, 14.0])
    assert out["temperature_2m_max"].tolist() == pytest.approx([14.0, 12.0, 16.0])


def test_build_weather_ensemble_table_mean_and_weighted() -> None:
    index = pd.date_range("2026-01-10 00:00:00", periods=2, freq="h", tz="Europe/Brussels")
    model_a = pd.DataFrame({"temperature_2m": [10.0, 20.0]}, index=index)
    model_b = pd.DataFrame({"temperature_2m": [30.0, 40.0]}, index=index)
    weather_ok = {"a": _forecast(model_a), "b": _forecast(model_b)}

    out_mean = we.build_weather_ensemble_table(weather_ok, index, "mean", None)
    out_weighted = we.build_weather_ensemble_table(weather_ok, index, "weighted", {"a": 0.75, "b": 0.25})

    assert out_mean["temperature_2m"].tolist() == pytest.approx([20.0, 30.0])
    assert out_weighted["temperature_2m"].tolist() == pytest.approx([15.0, 25.0])


def test_build_weather_ensemble_table_weighted_renormalizes_for_nans() -> None:
    index = pd.date_range("2026-01-10 00:00:00", periods=2, freq="h", tz="Europe/Brussels")
    model_a = pd.DataFrame({"temperature_2m": [10.0, 20.0]}, index=index)
    model_b = pd.DataFrame({"temperature_2m": [30.0, float("nan")]}, index=index)

    out = we.build_weather_ensemble_table(
        weather_ok={"a": _forecast(model_a), "b": _forecast(model_b)},
        index=index,
        ensemble_method="weighted",
        weights={"a": 0.25, "b": 0.75},
    )

    assert out["temperature_2m"].iloc[0] == pytest.approx((10.0 * 0.25 + 30.0 * 0.75) / (0.25 + 0.75))
    assert out["temperature_2m"].iloc[1] == pytest.approx(20.0)
    assert out["temperature_2m_min"].iloc[1] == pytest.approx(20.0)
    assert out["temperature_2m_max"].iloc[1] == pytest.approx(20.0)
