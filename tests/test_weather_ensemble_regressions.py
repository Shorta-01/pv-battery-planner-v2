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


@pytest.fixture(autouse=True)
def _isolate_provider_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(we, "PROVIDER_CACHE_DIR", tmp_path / "provider_cache")
    monkeypatch.setattr(we, "PROVIDER_CIRCUIT_STATE_PATH", we.PROVIDER_CACHE_DIR / "circuit_breaker_state.json")
    we._WEATHER_CACHE.clear()
    we._CIRCUIT_BREAKER_STATE.clear()
    we._CIRCUIT_BREAKER_LOADED = False


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
        out = weather_df.copy()
        out["temp_air_c"] = 10.0 if model_id == "knmi_harmonie_arome" else 11.0
        return core.ForecastResult(df=out, sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()), [], False, {"source": "live", "live_failed_used_cached": False}

    values = {
        "knmi_harmonie_arome": [1.0, 2.0, 3.0],
        "dwd_icon_d2": [3.0, float("nan"), 9.0],
    }

    def fake_build_pv(df, _loc, tz=None):
        if float(df["temp_air_c"].iloc[0]) > 10.5:
            model_id = "dwd_icon_d2"
        else:
            model_id = "knmi_harmonie_arome"
        s = pd.Series(values[model_id], index=hourly_index[:3]).reindex(df.index)
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
        out = weather_df.copy()
        out["temp_air_c"] = 10.0 if model_id == "ecmwf_ifs" else 11.0
        return core.ForecastResult(df=out, sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()), [], False, {"source": "live", "live_failed_used_cached": False}

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

    assert sorted(calls) == ["dwd_icon_d2", "ecmwf_ifs"]
    assert sorted(out.selected_models) == ["dwd_icon_d2", "ecmwf_ifs"]


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
    _, _, derived, _meta = we.fetch_open_meteo_weather(
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
    _, _, derived_missing, _meta = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 11),
    )
    assert derived_missing is True




def test_fetch_open_meteo_partially_missing_dni_dhi_backfills_gaps(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    payload_partial = {
        "hourly": {
            "time": [ts.isoformat() for ts in hourly_index],
            "temperature_2m": [10.0] * 24,
            "wind_speed_10m": [1.0] * 24,
            "shortwave_radiation": [100.0] * 24,
            "direct_normal_irradiance": [50.0 if i % 2 == 0 else None for i in range(24)],
            "diffuse_radiation": [20.0 if i % 3 != 0 else None for i in range(24)],
            "cloud_cover": [35.0] * 24,
        },
        "daily": {"sunrise": [hourly_index[7].isoformat()], "sunset": [hourly_index[17].isoformat()]},
    }

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", lambda *args, **kwargs: payload_partial)
    forecast, _, derived, _meta = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )
    assert derived is True
    assert forecast.df["dni_wm2"].isna().sum() == 0
    assert forecast.df["dhi_wm2"].isna().sum() == 0
    assert forecast.df["dni_wm2"].iloc[2] == pytest.approx(50.0)





def test_fetch_open_meteo_adds_ui_alias_columns(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    payload = {
        "hourly": {
            "time": [ts.isoformat() for ts in hourly_index],
            "temperature_2m": [11.0] * 24,
            "wind_speed_10m": [2.0] * 24,
            "shortwave_radiation": [120.0] * 24,
            "direct_normal_irradiance": [55.0] * 24,
            "diffuse_radiation": [25.0] * 24,
            "cloud_cover": [30.0] * 24,
        },
        "daily": {"sunrise": [hourly_index[7].isoformat()], "sunset": [hourly_index[17].isoformat()]},
    }

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", lambda *args, **kwargs: payload)
    forecast, _, _, _meta = we.fetch_open_meteo_weather(
        model_id="knmi_harmonie_arome",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )

    for column in [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "temperature_2m",
        "wind_speed_10m",
        "cloud_cover",
    ]:
        assert column in forecast.df.columns

    assert forecast.df["shortwave_radiation"].iloc[0] == pytest.approx(120.0)
    assert forecast.df["direct_normal_irradiance"].iloc[0] == pytest.approx(55.0)
    assert forecast.df["diffuse_radiation"].iloc[0] == pytest.approx(25.0)


def test_fetch_open_meteo_knmi_requests_dni_dhi_when_supported(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    payload = {
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

    calls: list[dict[str, object]] = []

    def fake_request(url: str, params: dict[str, object], model_id: str):
        calls.append(params)
        return payload

    monkeypatch.setattr(we, "_request_open_meteo", fake_request)

    we.fetch_open_meteo_weather(
        model_id="knmi_harmonie_arome",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )

    assert calls
    hourly_requested = str(calls[0]["hourly"])
    assert "direct_normal_irradiance" in hourly_requested
    assert "diffuse_radiation" in hourly_requested
def test_fetch_open_meteo_retries_404_with_forecast_endpoint(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    payload = {
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

    calls: list[tuple[str, str | None]] = []

    def fake_request(url: str, params: dict[str, object], model_id: str):
        calls.append((url, params.get("models") if isinstance(params, dict) else None))
        if len(calls) == 1:
            raise we.WeatherProviderError(
                category="http_error",
                status=404,
                message=f"Open-Meteo request failed (http_error) for {model_id} status=404",
            )
        return payload

    monkeypatch.setattr(we, "_request_open_meteo", fake_request)

    forecast, _, derived, _meta = we.fetch_open_meteo_weather(
        model_id="knmi_harmonie_arome",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )

    assert len(calls) == 2
    assert calls[0][0] == we.WEATHER_MODELS["knmi_harmonie_arome"]["endpoint"]
    assert calls[1][0] == "https://api.open-meteo.com/v1/forecast"
    assert calls[1][1] == "knmi_seamless"
    assert derived is False
    assert forecast.df["ghi_wm2"].iloc[0] == pytest.approx(100.0)


def test_fetch_open_meteo_retries_400_with_forecast_endpoint_for_model_support_errors(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    payload = {
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

    calls: list[tuple[str, str | None]] = []

    def fake_request(url: str, params: dict[str, object], model_id: str):
        calls.append((url, params.get("models") if isinstance(params, dict) else None))
        if len(calls) == 1:
            raise we.WeatherProviderError(
                category="http_error",
                status=400,
                provider_reason="Hourly parameter direct_normal_irradiance is not supported for this model",
                message=f"Open-Meteo request failed (http_error) for {model_id} status=400",
            )
        return payload

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", fake_request)

    forecast, _, derived, _meta = we.fetch_open_meteo_weather(
        model_id="knmi_harmonie_arome",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )

    assert len(calls) == 2
    assert calls[0][0] == we.WEATHER_MODELS["knmi_harmonie_arome"]["endpoint"]
    assert calls[1][0] == "https://api.open-meteo.com/v1/forecast"
    assert calls[1][1] == "knmi_seamless"
    assert derived is False
    assert forecast.df["ghi_wm2"].iloc[0] == pytest.approx(100.0)


def test_fetch_open_meteo_does_not_retry_400_for_invalid_request(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_request(url: str, params: dict[str, object], model_id: str):
        calls.append((url, params.get("models") if isinstance(params, dict) else None))
        raise we.WeatherProviderError(
            category="http_error",
            status=400,
            provider_reason="Invalid latitude parameter",
            message=f"Open-Meteo request failed (http_error) for {model_id} status=400",
        )

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", fake_request)

    with pytest.raises(we.WeatherProviderError):
        we.fetch_open_meteo_weather(
            model_id="knmi_harmonie_arome",
            loc=core.Location(name="x", latitude=50.8, longitude=4.3),
            tz="Europe/Brussels",
            target_date=dt.date(2026, 1, 10),
        )

    assert len(calls) == 1

def test_request_open_meteo_handles_429_without_json_method(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    class DummyResp:
        status_code = 429
        text = "Too Many Requests"

        def raise_for_status(self):
            raise requests.HTTPError(response=self)

    class FakeSession:
        def get(self, *args, **kwargs):
            return DummyResp()

    monkeypatch.setattr(we, "_SESSION", FakeSession())
    with pytest.raises(we.WeatherProviderError) as err:
        we._request_open_meteo("https://example.com", {}, model_id="ecmwf_ifs")

    assert err.value.category == "rate_limited"
    assert err.value.status == 429
    assert err.value.provider_reason == "Too Many Requests"


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
        return core.ForecastResult(df=weather_df.copy(), sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()), [], False, {"source": "live", "live_failed_used_cached": False}

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


def test_weather_provider_error_to_reason_includes_provider_reason() -> None:
    err = we.WeatherProviderError(
        category="http_error",
        status=400,
        provider_reason="unknown model",
        message="Open-Meteo request failed",
    )

    assert err.to_reason() == {
        "category": "http_error",
        "status": 400,
        "message": "Open-Meteo request failed",
        "provider_reason": "unknown model",
    }


def test_fetch_open_meteo_logs_structured_success(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, hourly_index: pd.DatetimeIndex) -> None:
    payload = {
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

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", lambda *args, **kwargs: payload)

    with caplog.at_level("INFO"):
        we.fetch_open_meteo_weather(
            model_id="dwd_icon_d2",
            loc=core.Location(name="x", latitude=50.8, longitude=4.3),
            tz="Europe/Brussels",
            target_date=dt.date(2026, 1, 10),
        )

    model_logs = [r.message for r in caplog.records if "model_fetch" in r.message]
    assert model_logs
    msg = model_logs[-1]
    assert "model=dwd_icon_d2" in msg
    assert "endpoint=https://api.open-meteo.com/v1/dwd-icon" in msg
    assert "status=200" in msg
    assert "elapsed_ms=" in msg
    assert "params_hash=" in msg
    assert "latitude=50.8" not in msg


def test_fetch_open_meteo_logs_structured_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path) -> None:
    def fake_request(_url: str, _params: dict[str, object], model_id: str):
        raise we.WeatherProviderError(
            category="rate_limited",
            status=429,
            message=f"Open-Meteo request failed (rate_limited) for {model_id} status=429",
        )

    we._WEATHER_CACHE.clear()
    monkeypatch.setattr(we, "_request_open_meteo", fake_request)

    with caplog.at_level("INFO"):
        with pytest.raises(we.WeatherProviderError):
            we.fetch_open_meteo_weather(
                model_id="dwd_icon_d2",
                loc=core.Location(name="x", latitude=50.8, longitude=4.3),
                tz="Europe/Brussels",
                target_date=dt.date(2026, 1, 10),
            )

    model_logs = [r.message for r in caplog.records if "model_fetch" in r.message]
    assert model_logs
    msg = model_logs[-1]
    assert "model=dwd_icon_d2" in msg
    assert "status=429" in msg
    assert "category=rate_limited" in msg
    assert "outcome=failed" in msg

def test_irradiance_anomaly_excludes_model_and_keeps_ensemble_running(
    monkeypatch: pytest.MonkeyPatch,
    hourly_index: pd.DatetimeIndex,
) -> None:
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
        out = weather_df.copy()
        if model_id == "dwd_icon_d2":
            out["ghi_wm2"] = 50000.0
            out["shortwave_radiation"] = 50000.0
        else:
            out["ghi_wm2"] = 50.0
            out["shortwave_radiation"] = 50.0
        return core.ForecastResult(df=out, sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()), [], False, {"source": "live", "live_failed_used_cached": False}

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
        weather_models=["knmi_harmonie_arome", "dwd_icon_d2"],
        ensemble_method="mean",
        pv_uncertainty=False,
        accuracy_mode=True,
        fast_mode=False,
    )

    assert "dwd_icon_d2" in out.failed_models
    assert out.failed_model_reasons["dwd_icon_d2"]["category"] == "irradiance_anomaly"
    assert "irradiance anomaly" in out.failed_model_reasons["dwd_icon_d2"]["message"]
    assert out.selected_models == ["knmi_harmonie_arome"]
    assert out.pv_ensemble_p50.sum() > 0


def test_build_ensemble_marks_model_failed_when_no_hourly_overlap(
    monkeypatch: pytest.MonkeyPatch,
    hourly_index: pd.DatetimeIndex,
) -> None:
    loc = core.Location(name="x", latitude=50.8, longitude=4.3)
    empty_df = pd.DataFrame(
        {
            "temp_air_c": [10.0] * len(hourly_index),
            "ghi_wm2": [20.0] * len(hourly_index),
            "dni_wm2": [10.0] * len(hourly_index),
            "dhi_wm2": [8.0] * len(hourly_index),
            "cloud_cover_pct": [30.0] * len(hourly_index),
            "wind_speed_ms": [1.0] * len(hourly_index),
        },
        index=hourly_index + dt.timedelta(days=5),
    )

    def fake_weather(model_id, *_args, **_kwargs):
        if model_id == "dwd_icon_d2":
            return (
                core.ForecastResult(df=empty_df.copy(), sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()),
                [],
                False,
                {"source": "live", "live_failed_used_cached": False, "horizon_days": 1},
            )
        out = pd.DataFrame(
            {
                "temp_air_c": [10.0] * len(hourly_index),
                "ghi_wm2": [20.0] * len(hourly_index),
                "dni_wm2": [10.0] * len(hourly_index),
                "dhi_wm2": [8.0] * len(hourly_index),
                "cloud_cover_pct": [30.0] * len(hourly_index),
                "wind_speed_ms": [1.0] * len(hourly_index),
            },
            index=hourly_index,
        )
        return (
            core.ForecastResult(df=out, sunrise=hourly_index[7].to_pydatetime(), sunset=hourly_index[17].to_pydatetime()),
            [],
            False,
            {"source": "live", "live_failed_used_cached": False, "horizon_days": 1},
        )

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
        weather_models=["knmi_harmonie_arome", "dwd_icon_d2"],
        ensemble_method="mean",
        pv_uncertainty=False,
        accuracy_mode=True,
        fast_mode=False,
    )

    assert "dwd_icon_d2" in out.failed_models
    assert out.failed_model_reasons["dwd_icon_d2"]["category"] == "invalid_payload"
    assert "no overlapping hourly coverage" in out.failed_model_reasons["dwd_icon_d2"]["message"]
    assert out.selected_models == ["knmi_harmonie_arome"]


def test_fetch_open_meteo_uses_last_good_cache_when_live_fetch_fails(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex, tmp_path) -> None:
    payload = {
        "hourly": {
            "time": [ts.isoformat() for ts in hourly_index],
            "temperature_2m": [11.0] * 24,
            "wind_speed_10m": [1.0] * 24,
            "shortwave_radiation": [101.0] * 24,
            "direct_normal_irradiance": [50.0] * 24,
            "diffuse_radiation": [20.0] * 24,
            "cloud_cover": [25.0] * 24,
        },
        "daily": {"sunrise": [hourly_index[7].isoformat()], "sunset": [hourly_index[17].isoformat()]},
    }

    monkeypatch.setattr(we, "PROVIDER_CACHE_DIR", tmp_path / "provider_cache")
    monkeypatch.setattr(we, "PROVIDER_CIRCUIT_STATE_PATH", we.PROVIDER_CACHE_DIR / "circuit_breaker_state.json")
    we._WEATHER_CACHE.clear()
    we._CIRCUIT_BREAKER_STATE.clear()
    we._CIRCUIT_BREAKER_LOADED = False

    monkeypatch.setattr(we, "_request_open_meteo", lambda *_args, **_kwargs: payload)
    forecast, _, _, first_meta = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )
    assert first_meta["source"] == "live"
    assert forecast.df["ghi_wm2"].iloc[0] == pytest.approx(101.0)

    we._WEATHER_CACHE.clear()

    def fail_request(*_args, **_kwargs):
        raise we.WeatherProviderError(category="provider_down", status=503, message="provider down")

    monkeypatch.setattr(we, "_request_open_meteo", fail_request)
    forecast_cached, _, _, second_meta = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )

    assert second_meta["source"] == "provider_cache"
    assert second_meta["live_failed_used_cached"] is True
    assert forecast_cached.df["ghi_wm2"].iloc[0] == pytest.approx(101.0)


def test_circuit_breaker_opens_and_skips_live_calls(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex, tmp_path) -> None:
    monkeypatch.setattr(we, "PROVIDER_CACHE_DIR", tmp_path / "provider_cache")
    monkeypatch.setattr(we, "PROVIDER_CIRCUIT_STATE_PATH", we.PROVIDER_CACHE_DIR / "circuit_breaker_state.json")
    we._WEATHER_CACHE.clear()
    we._CIRCUIT_BREAKER_STATE.clear()
    we._CIRCUIT_BREAKER_LOADED = False

    run_hour = 12
    monkeypatch.setattr(
        we.dt,
        "datetime",
        type(
            "FixedDateTime",
            (dt.datetime,),
            {
                "now": classmethod(lambda cls, tz=None: dt.datetime(2026, 1, 10, run_hour, 0, tzinfo=tz or dt.timezone.utc)),
            },
        ),
    )

    payload = {
        "hourly": {
            "time": [ts.isoformat() for ts in hourly_index],
            "temperature_2m": [10.0] * 24,
            "wind_speed_10m": [1.0] * 24,
            "shortwave_radiation": [90.0] * 24,
            "direct_normal_irradiance": [45.0] * 24,
            "diffuse_radiation": [18.0] * 24,
            "cloud_cover": [20.0] * 24,
        },
        "daily": {"sunrise": [hourly_index[7].isoformat()], "sunset": [hourly_index[17].isoformat()]},
    }

    we._store_provider_cache("ecmwf_ifs", dt.date(2026, 1, 10), run_hour, payload)

    calls = {"count": 0}

    def fail_request(*_args, **_kwargs):
        calls["count"] += 1
        raise we.WeatherProviderError(category="provider_down", status=503, message="provider down")

    monkeypatch.setattr(we, "_request_open_meteo", fail_request)

    for _ in range(we.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        we._WEATHER_CACHE.clear()
        _, _, _, meta = we.fetch_open_meteo_weather(
            model_id="ecmwf_ifs",
            loc=core.Location(name="x", latitude=50.8, longitude=4.3),
            tz="Europe/Brussels",
            target_date=dt.date(2026, 1, 10),
        )
        assert meta["source"] == "provider_cache"

    expected_live_calls = we.CIRCUIT_BREAKER_FAILURE_THRESHOLD
    assert calls["count"] == expected_live_calls

    we._WEATHER_CACHE.clear()
    _, _, _, meta = we.fetch_open_meteo_weather(
        model_id="ecmwf_ifs",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
    )
    assert meta["source"] == "provider_cache"
    assert meta["circuit_breaker_open"] is True
    assert calls["count"] == expected_live_calls


def test_fetch_open_meteo_in_memory_cache_meta_keeps_horizon_days(monkeypatch: pytest.MonkeyPatch, hourly_index: pd.DatetimeIndex) -> None:
    payload = {
        "hourly": {
            "time": [ts.isoformat() for ts in hourly_index],
            "temperature_2m": [11.0] * 24,
            "wind_speed_10m": [1.0] * 24,
            "shortwave_radiation": [101.0] * 24,
            "direct_normal_irradiance": [50.0] * 24,
            "diffuse_radiation": [20.0] * 24,
            "cloud_cover": [25.0] * 24,
        },
        "daily": {"sunrise": [hourly_index[7].isoformat()], "sunset": [hourly_index[17].isoformat()]},
    }

    monkeypatch.setattr(we, "_request_open_meteo", lambda *_args, **_kwargs: payload)

    _, _, _, first_meta = we.fetch_open_meteo_weather(
        model_id="dwd_icon_d2",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
        requested_days=7,
    )
    assert first_meta["source"] == "live"
    assert first_meta["horizon_days"] == 2

    _, _, _, second_meta = we.fetch_open_meteo_weather(
        model_id="dwd_icon_d2",
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        target_date=dt.date(2026, 1, 10),
        requested_days=7,
    )
    assert second_meta["source"] == "in_memory_cache"
    assert second_meta["cache_hit"] is True
    assert second_meta["requested_days"] == 7
    assert second_meta["horizon_days"] == 2
    assert second_meta["model_max_days"] == 2


def test_build_ensemble_classifies_overlap_and_tail_missing_hours(monkeypatch: pytest.MonkeyPatch) -> None:
    loc = core.Location(name="x", latitude=50.8, longitude=4.3)
    tz = "Europe/Brussels"
    target_date = dt.date(2026, 1, 10)
    canonical_index = pd.date_range(
        pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=tz),
        pd.Timestamp(dt.datetime.combine(target_date + dt.timedelta(days=7), dt.time(0, 0)), tz=tz),
        freq="h",
        inclusive="left",
    )

    icon_df = pd.DataFrame(
        {
            "temp_air_c": [10.0] * (2 * 24),
            "ghi_wm2": [20.0] * (2 * 24),
            "dni_wm2": [10.0] * (2 * 24),
            "dhi_wm2": [8.0] * (2 * 24),
            "cloud_cover_pct": [30.0] * (2 * 24),
            "wind_speed_ms": [1.0] * (2 * 24),
        },
        index=canonical_index[: 2 * 24],
    )
    seamless_df = pd.DataFrame(
        {
            "temp_air_c": [11.0] * (4 * 24),
            "ghi_wm2": [25.0] * (4 * 24),
            "dni_wm2": [12.0] * (4 * 24),
            "dhi_wm2": [9.0] * (4 * 24),
            "cloud_cover_pct": [35.0] * (4 * 24),
            "wind_speed_ms": [2.0] * (4 * 24),
        },
        index=canonical_index[: 4 * 24],
    )

    def fake_weather(model_id, *_args, **_kwargs):
        if model_id == "dwd_icon_d2":
            return core.ForecastResult(df=icon_df, sunrise=canonical_index[7].to_pydatetime(), sunset=canonical_index[17].to_pydatetime()), [], False, {
                "source": "in_memory_cache",
                "requested_days": 7,
                "horizon_days": 2,
                "model_max_days": 2,
                "cache_hit": True,
            }
        return core.ForecastResult(df=seamless_df, sunrise=canonical_index[7].to_pydatetime(), sunset=canonical_index[17].to_pydatetime()), [], False, {
            "source": "in_memory_cache",
            "requested_days": 7,
            "horizon_days": 4,
            "model_max_days": 4,
            "cache_hit": True,
        }

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
        target_date=target_date,
        tz=tz,
        weather_models=["dwd_icon_d2", "meteofrance_seamless"],
        ensemble_method="mean",
        pv_uncertainty=False,
        accuracy_mode=True,
        fast_mode=False,
        requested_days=7,
    )

    icon_meta = out.fetch_meta_by_model["dwd_icon_d2"]
    seamless_meta = out.fetch_meta_by_model["meteofrance_seamless"]

    assert icon_meta["missing_hours_overlap"] == 0
    assert icon_meta["missing_hours_tail"] == 120
    assert icon_meta["expected_tail_hours"] == 120

    assert seamless_meta["missing_hours_overlap"] == 0
    assert seamless_meta["missing_hours_tail"] == 72
    assert seamless_meta["expected_tail_hours"] == 72
