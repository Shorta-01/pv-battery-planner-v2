import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
import weather_ensemble as we


@pytest.fixture
def hourly_index() -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp("2026-01-10 00:00:00", tz="Europe/Brussels"), periods=24, freq="h")


def test_weighted_ensemble_renormalizes_per_timestamp(hourly_index: pd.DatetimeIndex) -> None:
    a = pd.Series([1.0, 2.0], index=hourly_index[:2])
    b = pd.Series([3.0, 4.0], index=hourly_index[:2])
    c = pd.Series([5.0, float("nan")], index=hourly_index[:2])

    out, weights = we._weighted_ensemble(
        {"knmi_harmonie_arome": a, "dwd_icon_d2": b, "ecmwf_ifs": c},
        ["knmi_harmonie_arome", "dwd_icon_d2", "ecmwf_ifs"],
    )

    assert weights == {
        "knmi_harmonie_arome": pytest.approx(0.45),
        "dwd_icon_d2": pytest.approx(0.35),
        "ecmwf_ifs": pytest.approx(0.20),
    }
    expected_h0 = (1.0 * 0.45 + 3.0 * 0.35 + 5.0 * 0.20) / (0.45 + 0.35 + 0.20)
    expected_h1 = (2.0 * 0.45 + 4.0 * 0.35) / (0.45 + 0.35)
    assert out.iloc[0] == pytest.approx(expected_h0)
    assert out.iloc[1] == pytest.approx(expected_h1)


def test_build_ensemble_mean_ignores_missing_model_hours(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    loc = core.Location(name="x", latitude=50.8, longitude=4.3)

    weather_df = pd.DataFrame(
        {
            "temp_air_c": [10.0] * len(hourly_index),
            "ghi_wm2": [0.0] * len(hourly_index),
            "dni_wm2": [0.0] * len(hourly_index),
            "dhi_wm2": [0.0] * len(hourly_index),
            "cloud_cover_pct": [0.0] * len(hourly_index),
            "wind_speed_ms": [1.0] * len(hourly_index),
        },
        index=hourly_index,
    )

    def fake_weather(model_id, *_args, **_kwargs):
        return core.ForecastResult(df=weather_df.copy(), sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()), [], False

    values = {
        "knmi_harmonie_arome": [1.0, 2.0, 3.0],
        "dwd_icon_d2": [3.0, float("nan"), 9.0],
    }
    call_seq = {"i": 0}

    def fake_build_pv(_df, _loc, tz=None):
        model_id = ["knmi_harmonie_arome", "dwd_icon_d2"][call_seq["i"]]
        call_seq["i"] += 1
        s = pd.Series(values[model_id], index=hourly_index[:3])
        return pd.DataFrame(
            {
                "pv_total_kwh": s,
                "pv_total_unclipped_kwh": s,
                "pv_east_kwh": s / 2,
                "pv_south_kwh": s / 2,
                "pv_clipped_kwh": [0.0, 0.0, 0.0],
            },
            index=hourly_index[:3],
        )

    monkeypatch.setattr(we, "fetch_open_meteo_weather", fake_weather)
    monkeypatch.setattr(core, "build_pv_forecast", fake_build_pv)

    out = we.build_ensemble_forecast(
        loc=loc,
        target_date=dt.date(2026, 1, 10),
        tz="Europe/Brussels",
        weather_models=["knmi_harmonie_arome", "dwd_icon_d2"],
        ensemble_method="mean",
        pv_uncertainty=False,
        accuracy_mode=True,
        fast_mode=False,
    )

    assert out.pv_ensemble_p50.iloc[0] == pytest.approx(2.0)
    assert out.pv_ensemble_p50.iloc[1] == pytest.approx(2.0)
    assert out.pv_ensemble_p50.iloc[2] == pytest.approx(6.0)


def test_fast_mode_limits_models(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    loc = core.Location(name="x", latitude=50.8, longitude=4.3)
    weather_df = pd.DataFrame(
        {
            "temp_air_c": [10.0] * len(hourly_index),
            "ghi_wm2": [0.0] * len(hourly_index),
            "dni_wm2": [0.0] * len(hourly_index),
            "dhi_wm2": [0.0] * len(hourly_index),
            "cloud_cover_pct": [0.0] * len(hourly_index),
            "wind_speed_ms": [1.0] * len(hourly_index),
        },
        index=hourly_index,
    )
    calls: list[str] = []

    def fake_weather(model_id, *_args, **_kwargs):
        calls.append(model_id)
        return core.ForecastResult(df=weather_df.copy(), sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()), [], False

    def fake_build_pv(df, _loc, tz=None):
        s = pd.Series([1.0] * len(df.index), index=df.index)
        return pd.DataFrame(
            {
                "pv_total_kwh": s,
                "pv_total_unclipped_kwh": s,
                "pv_east_kwh": s / 2,
                "pv_south_kwh": s / 2,
                "pv_clipped_kwh": [0.0] * len(df.index),
            },
            index=df.index,
        )

    monkeypatch.setattr(we, "fetch_open_meteo_weather", fake_weather)
    monkeypatch.setattr(core, "build_pv_forecast", fake_build_pv)

    out = we.build_ensemble_forecast(
        loc=loc,
        target_date=dt.date(2026, 1, 10),
        tz="Europe/Brussels",
        weather_models=["ecmwf_ifs", "dwd_icon_d2", "knmi_harmonie_arome"],
        ensemble_method="mean",
        pv_uncertainty=False,
        accuracy_mode=True,
        fast_mode=True,
    )

    assert calls == ["ecmwf_ifs", "dwd_icon_d2"]
    assert out.selected_models == ["ecmwf_ifs", "dwd_icon_d2"]


def test_fetch_open_meteo_sets_derived_flag_from_missing_dni_dhi(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    payload_with_native = {
        "hourly": {
            "time": [ts.isoformat() for ts in hourly_index],
            "temperature_2m": [10.0] * 24,
            "wind_speed_10m": [1.0] * 24,
            "shortwave_radiation": [100.0] * 24,
            "direct_normal_irradiance": [50.0] * 24,
            "diffuse_radiation": [20.0] * 24,
            "cloud_cover": [25.0] * 24,
        },
        "daily": {"sunrise": [hourly_index[7].isoformat()], "sunset": [hourly_index[17].isoformat()]},
    }

    monkeypatch.setattr(we, "_request_open_meteo", lambda *args, **kwargs: payload_with_native)
    _, _, derived = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )
    assert derived is False

    payload_missing = {
        "hourly": {
            "time": [ts.isoformat() for ts in hourly_index],
            "temperature_2m": [10.0] * 24,
            "wind_speed_10m": [1.0] * 24,
            "shortwave_radiation": [100.0] * 24,
            "cloud_cover": [25.0] * 24,
        },
        "daily": {"sunrise": [hourly_index[7].isoformat()], "sunset": [hourly_index[17].isoformat()]},
    }
    monkeypatch.setattr(we, "_request_open_meteo", lambda *args, **kwargs: payload_missing)
    _, _, derived_missing = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 11),
    )
    assert derived_missing is True


def test_request_open_meteo_error_categorization(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 429

        def raise_for_status(self):
            import requests

            raise requests.HTTPError(response=self)

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(we, "_SESSION", FakeSession())
    with pytest.raises(RuntimeError, match="rate_limited"):
        we._request_open_meteo("https://example.com", {}, model_id="ecmwf_ifs")


def test_request_open_meteo_timeout_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    class FakeSession:
        def get(self, *args, **kwargs):
            raise requests.Timeout("timed out")

    monkeypatch.setattr(we, "_SESSION", FakeSession())
    with pytest.raises(we.WeatherProviderError, match="timeout") as err:
        we._request_open_meteo("https://example.com", {}, model_id="ecmwf_ifs")
    assert err.value.category == "timeout"


def test_request_open_meteo_malformed_json_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("bad json")

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(we, "_SESSION", FakeSession())
    with pytest.raises(we.WeatherProviderError, match="Malformed JSON") as err:
        we._request_open_meteo("https://example.com", {}, model_id="ecmwf_ifs")
    assert err.value.category == "malformed_json"
    assert err.value.status == 200


def test_build_ensemble_surfaces_failed_model_reasons(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    loc = core.Location(name="x", latitude=50.8, longitude=4.3)
    weather_df = pd.DataFrame(
        {
            "temp_air_c": [10.0] * len(hourly_index),
            "ghi_wm2": [0.0] * len(hourly_index),
            "dni_wm2": [0.0] * len(hourly_index),
            "dhi_wm2": [0.0] * len(hourly_index),
            "cloud_cover_pct": [0.0] * len(hourly_index),
            "wind_speed_ms": [1.0] * len(hourly_index),
        },
        index=hourly_index,
    )

    def fake_weather(model_id, *_args, **_kwargs):
        if model_id == "dwd_icon_d2":
            raise we.WeatherProviderError(category="rate_limited", status=429, message="rate limited")
        return core.ForecastResult(df=weather_df.copy(), sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()), [], False

    def fake_build_pv(df, _loc, tz=None):
        s = pd.Series([1.0] * len(df.index), index=df.index)
        return pd.DataFrame(
            {
                "pv_total_kwh": s,
                "pv_total_unclipped_kwh": s,
                "pv_east_kwh": s / 2,
                "pv_south_kwh": s / 2,
                "pv_clipped_kwh": [0.0] * len(df.index),
            },
            index=df.index,
        )

    monkeypatch.setattr(we, "fetch_open_meteo_weather", fake_weather)
    monkeypatch.setattr(core, "build_pv_forecast", fake_build_pv)

    out = we.build_ensemble_forecast(
        loc=loc,
        target_date=dt.date(2026, 1, 10),
        tz="Europe/Brussels",
        weather_models=["dwd_icon_d2", "ecmwf_ifs"],
        ensemble_method="mean",
        pv_uncertainty=False,
        accuracy_mode=True,
        fast_mode=False,
    )

    assert out.failed_models == ["dwd_icon_d2"]
    assert out.failed_model_reasons == {
        "dwd_icon_d2": {"category": "rate_limited", "status": 429, "message": "rate limited"}
    }
    assert out.selected_models == ["ecmwf_ifs"]
