from types import SimpleNamespace

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api
import planner_core as core


def _new_state(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_api, "LOCAL_STATE_DIR", tmp_path / "local_state")
    monkeypatch.setattr(backend_api, "SETTINGS_PATH", backend_api.LOCAL_STATE_DIR / "settings.json")
    monkeypatch.setattr(backend_api, "INPUTS_PATH", backend_api.LOCAL_STATE_DIR / "last_inputs.json")
    monkeypatch.setattr(backend_api, "LATEST_RESULT_PATH", backend_api.LOCAL_STATE_DIR / "latest_result.json")
    monkeypatch.setattr(backend_api, "HISTORY_PATH", backend_api.LOCAL_STATE_DIR / "results_history.json")
    monkeypatch.setattr(backend_api, "SQLITE_PATH", backend_api.LOCAL_STATE_DIR / "planner_history.sqlite")
    monkeypatch.setattr(backend_api, "TOKEN_PATH", backend_api.LOCAL_STATE_DIR / "api_token.txt")
    monkeypatch.setattr(backend_api, "RUN_HISTORY_PATH", tmp_path / "run_history_log.json")
    monkeypatch.setattr(backend_api, "init_db", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_api, "fetch_latest_full_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend_api, "insert_forecast_run", lambda *_args, **_kwargs: None)
    return backend_api.BackendState()


def _fake_ensemble(idx, weather):
    p50 = pd.Series([1.0, 2.0], index=idx)
    return SimpleNamespace(
        weather_primary=weather,
        weather_primary_model_id="ecmwf_ifs",
        weather_ensemble_table=SimpleNamespace(df=weather.df.copy()),
        weather_by_model={"ecmwf_ifs": weather},
        pv_by_model={
            "ecmwf_ifs": pd.DataFrame(
                {
                    "pv_total_kwh": p50,
                    "pv_total_unclipped_kwh": p50,
                    "pv_east_kwh": p50 / 2,
                    "pv_south_kwh": p50 / 2,
                    "pv_clipped_kwh": pd.Series([0.0, 0.0], index=idx),
                },
                index=idx,
            ),
        },
        pv_ensemble_east_p50=p50 / 2,
        pv_ensemble_south_p50=p50 / 2,
        pv_ensemble_unclipped_p50=p50,
        pv_ensemble_p50=p50,
        pv_ensemble_p25=None,
        pv_ensemble_p10=None,
        pv_ensemble_p90=None,
        selected_models=["ecmwf_ifs"],
        weights_used={"ecmwf_ifs": 1.0},
        per_model_pv_totals_kwh={"ecmwf_ifs": float(p50.sum())},
        missing_vars_by_model={},
        derived_irradiance_by_model={"ecmwf_ifs": False},
        derived_weather_code_by_model={"ecmwf_ifs": False},
        derived_irradiance_hours_by_model={"ecmwf_ifs": 0},
        quality_weight_factors_by_model={"ecmwf_ifs": 1.0},
        failed_models=[],
        failed_model_reasons={},
        model_live_failed_used_cached={},
        provider_payloads_by_model={},
        fetch_meta_by_model={},
        pv_tomorrow_model_spread_kwh=None,
        pv_models_used_count_per_hour=None,
        deduped_models_dropped=None,
        model_selection_reason="auto",
        dynamic_daylight_method=None,
        forecast_quality_tier=None,
        satellite_nowcast_used=False,
        satellite_nowcast_hours=0,
        satellite_nowcast_weight_factor=None,
        satellite_nowcast_reason=None,
        day_type=None,
        weights_by_model={"ecmwf_ifs": 1.0},
        nowcast_available=False,
        nowcast_used_hours=0,
        nowcast_blend_hours=0,
    )


def test_run_now_exposes_stable_pv_temperature_metadata(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)

    idx = pd.date_range("2026-01-10", periods=2, freq="h", tz="Europe/Brussels")
    weather_df = pd.DataFrame({"temp_air_c": [10.0, 11.0], "wind_speed_ms": [1.0, 1.0], "cloud_cover_pct": [20.0, 30.0]}, index=idx)
    weather = core.ForecastResult(df=weather_df, sunrise=idx[0].to_pydatetime(), sunset=idx[-1].to_pydatetime())

    monkeypatch.setattr(backend_api, "build_ensemble_forecast", lambda **_kwargs: _fake_ensemble(idx, weather))

    result = state.run_now(backend_api.RunNowPayload(pv_uncertainty=False, weather_models=["ecmwf_ifs"]))["result"]

    for container in (result, result["weather_ensemble"]):
        meta = container["pv_temperature_metadata"]
        assert meta["temperature_model"] == core.PV_TEMPERATURE_MODEL
        assert meta["temperature_wind_input_source"] == core.PV_TEMPERATURE_WIND_INPUT_SOURCE
        assert meta["effective_module_wind_height_m"] == float(core.PV_EFFECTIVE_MODULE_WIND_HEIGHT_M)
        assert meta["forecast_wind_reference_height_m"] == float(core.PV_FORECAST_WIND_REFERENCE_HEIGHT_M)
        assert meta["faiman_u0"] == float(core.PV_TEMPERATURE_FAIMAN_U0)
        assert meta["faiman_u1"] == float(core.PV_TEMPERATURE_FAIMAN_U1)
