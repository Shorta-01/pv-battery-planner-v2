from types import SimpleNamespace

import pandas as pd
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api
import planner_core as core


def _patch_core_for_run(monkeypatch):
    monkeypatch.setattr(backend_api, "insert_forecast_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(core, "ensure_pv_columns", lambda df, split_ratio=None: df)
    monkeypatch.setattr(core, "apply_daylight_clamp", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(core, "add_sun_percent", lambda df, *_args, **_kwargs: df)

    def fake_add_load(df, yesterday_kwh):
        out = df.copy()
        out["load_kwh"] = float(yesterday_kwh) / max(len(out.index), 1)
        return out

    monkeypatch.setattr(core, "add_load_and_surplus_columns", fake_add_load)
    monkeypatch.setattr(core, "compute_soc_low_timing_aware", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(core, "compute_soc_high_headroom", lambda *_args, **_kwargs: (0.0, 0.8))
    monkeypatch.setattr(core, "choose_cutoff_soc", lambda *_args, **_kwargs: (0.5, "ok"))
    monkeypatch.setattr(core, "plan_charge_power", lambda *_args, **_kwargs: (None, 2.0, "ok", 0.4))
    monkeypatch.setattr(
        core,
        "simulate_expensive_hours_detailed",
        lambda pv, *_args, **_kwargs: (pd.DataFrame(index=pv.index), 1.0, 0.5, 0.0, 0.0),
    )
    monkeypatch.setattr(
        core,
        "simulate_full_day_soc",
        lambda pv, *_args, **_kwargs: (
            pd.Series([0.5] * len(pv.index), index=pv.index),
            pd.DataFrame(index=pv.index),
        ),
    )
    monkeypatch.setattr(
        core,
        "estimate_pv_with_pvlib",
        lambda clear_df, *_args, **_kwargs: (None, None, None, None, None, pd.Series([2.0] * len(clear_df.index), index=clear_df.index)),
    )
    monkeypatch.setattr(core, "compute_euro_savings_no_battery_vs_plan", lambda *_args, **_kwargs: {})




def _patch_core_for_run_without_insert(monkeypatch):
    monkeypatch.setattr(core, "ensure_pv_columns", lambda df, split_ratio=None: df)
    monkeypatch.setattr(core, "apply_daylight_clamp", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(core, "add_sun_percent", lambda df, *_args, **_kwargs: df)

    def fake_add_load(df, yesterday_kwh):
        out = df.copy()
        out["load_kwh"] = float(yesterday_kwh) / max(len(out.index), 1)
        return out

    monkeypatch.setattr(core, "add_load_and_surplus_columns", fake_add_load)
    monkeypatch.setattr(core, "compute_soc_low_timing_aware", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(core, "compute_soc_high_headroom", lambda *_args, **_kwargs: (0.0, 0.8))
    monkeypatch.setattr(core, "choose_cutoff_soc", lambda *_args, **_kwargs: (0.5, "ok"))
    monkeypatch.setattr(core, "plan_charge_power", lambda *_args, **_kwargs: (None, 2.0, "ok", 0.4))
    monkeypatch.setattr(
        core,
        "simulate_expensive_hours_detailed",
        lambda pv, *_args, **_kwargs: (pd.DataFrame(index=pv.index), 1.0, 0.5, 0.0, 0.0),
    )
    monkeypatch.setattr(
        core,
        "simulate_full_day_soc",
        lambda pv, *_args, **_kwargs: (
            pd.Series([0.5] * len(pv.index), index=pv.index),
            pd.DataFrame(index=pv.index),
        ),
    )
    monkeypatch.setattr(
        core,
        "estimate_pv_with_pvlib",
        lambda clear_df, *_args, **_kwargs: (None, None, None, None, None, pd.Series([2.0] * len(clear_df.index), index=clear_df.index)),
    )
    monkeypatch.setattr(core, "compute_euro_savings_no_battery_vs_plan", lambda *_args, **_kwargs: {})

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
    return backend_api.BackendState()


def _fake_ensemble(idx, weather, pv_uncertainty):
    p50 = pd.Series([1.0, 2.0], index=idx)
    p10 = pd.Series([0.5, 1.0], index=idx) if pv_uncertainty else None
    p90 = pd.Series([1.5, 3.0], index=idx) if pv_uncertainty else None
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
        pv_ensemble_p10=p10,
        pv_ensemble_p90=p90,
        selected_models=["ecmwf_ifs"],
        weights_used={"ecmwf_ifs": 1.0},
        per_model_pv_totals_kwh={"ecmwf_ifs": float(p50.sum())},
        missing_vars_by_model={},
        derived_irradiance_by_model={"ecmwf_ifs": False},
        failed_models=[],
        failed_model_reasons={},
        model_live_failed_used_cached={},
        provider_payloads_by_model={},
    )


def test_run_now_pv_uncertainty_false_omits_uncertainty_outputs(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    _patch_core_for_run(monkeypatch)

    idx = pd.date_range("2026-01-10", periods=2, freq="h", tz="Europe/Brussels")
    weather_df = pd.DataFrame({"temp_air_c": [10.0, 11.0], "wind_speed_ms": [1.0, 1.0], "cloud_cover_pct": [20.0, 30.0]}, index=idx)
    weather = core.ForecastResult(df=weather_df, sunrise=idx[0].to_pydatetime(), sunset=idx[-1].to_pydatetime())
    calls = []

    def fake_build_ensemble_forecast(*, pv_uncertainty, **_kwargs):
        calls.append(pv_uncertainty)
        return _fake_ensemble(idx, weather, pv_uncertainty)

    monkeypatch.setattr(backend_api, "build_ensemble_forecast", fake_build_ensemble_forecast)

    result = state.run_now(backend_api.RunNowPayload(pv_uncertainty=False, weather_models=["ecmwf_ifs"]))["result"]

    assert calls == [False]
    assert result["status"] == "ok"
    assert result["warnings_count"] == 0
    assert isinstance(result["run_duration_ms"], int)
    assert result["run_duration_ms"] >= 0
    assert result["pv_totals_kwh"] == {"p10": None, "p50": 3.0, "p90": None}
    assert result["inputs_used"]["buffer_percent"] == 0.0
    assert result["inputs_used"]["max_ac_charge_power_kw"] == 5.0
    assert result["inputs_used"]["weather_models_selected"] == ["ecmwf_ifs"]
    assert result["inputs_used"]["ensemble_method"] == "weighted"
    assert result["inputs_used"]["pv_uncertainty_enabled"] is False
    assert result["inputs_used"]["fast_mode"] is False
    assert result["weather_ensemble"]["pv_totals_kwh"] is None
    assert "pv_total_low_kwh" not in result["pv"]["columns"]
    assert "pv_total_high_kwh" not in result["pv"]["columns"]


def test_run_now_pv_uncertainty_true_returns_uncertainty_outputs(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    _patch_core_for_run(monkeypatch)

    idx = pd.date_range("2026-01-10", periods=2, freq="h", tz="Europe/Brussels")
    weather_df = pd.DataFrame({"temp_air_c": [10.0, 11.0], "wind_speed_ms": [1.0, 1.0], "cloud_cover_pct": [20.0, 30.0]}, index=idx)
    weather = core.ForecastResult(df=weather_df, sunrise=idx[0].to_pydatetime(), sunset=idx[-1].to_pydatetime())

    monkeypatch.setattr(
        backend_api,
        "build_ensemble_forecast",
        lambda *, pv_uncertainty, **_kwargs: _fake_ensemble(idx, weather, pv_uncertainty),
    )

    result = state.run_now(backend_api.RunNowPayload(pv_uncertainty=True, weather_models=["ecmwf_ifs"]))["result"]

    assert result["status"] == "ok"
    assert result["warnings_count"] == 0
    assert result["pv_totals_kwh"] == {"p10": 1.5, "p50": 3.0, "p90": 4.5}
    assert result["weather_ensemble"]["pv_totals_kwh"]["p10"] == 1.5
    assert result["weather_ensemble"]["pv_totals_kwh"]["p90"] == 4.5
    assert "pv_total_low_kwh" in result["pv"]["columns"]
    assert "pv_total_high_kwh" in result["pv"]["columns"]


def test_run_now_degraded_generates_health_warnings(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    _patch_core_for_run(monkeypatch)

    idx = pd.date_range("2026-01-10", periods=2, freq="h", tz="Europe/Brussels")
    weather_df = pd.DataFrame({"temp_air_c": [10.0, 11.0], "wind_speed_ms": [1.0, 1.0], "cloud_cover_pct": [20.0, 30.0]}, index=idx)
    weather = core.ForecastResult(df=weather_df, sunrise=idx[0].to_pydatetime(), sunset=idx[-1].to_pydatetime())

    ensemble = _fake_ensemble(idx, weather, pv_uncertainty=False)
    ensemble.failed_models = ["dwd_icon_d2"]
    ensemble.failed_model_reasons = {"dwd_icon_d2": {"message": "rate limited"}}
    ensemble.derived_irradiance_by_model = {"ecmwf_ifs": True}
    ensemble.model_live_failed_used_cached = {"ecmwf_ifs": True}
    ensemble.missing_vars_by_model = {"ecmwf_ifs": ["direct_normal_irradiance", "foo"]}

    monkeypatch.setattr(
        backend_api,
        "build_ensemble_forecast",
        lambda **_kwargs: ensemble,
    )

    result = state.run_now(backend_api.RunNowPayload(pv_uncertainty=False, weather_models=["ecmwf_ifs"]))["result"]

    assert result["status"] == "degraded"
    assert result["warnings_count"] == 4
    assert any("model failed: dwd_icon_d2 (rate limited)" == w for w in result["warnings"])
    assert any("derived irradiance used: ecmwf_ifs" == w for w in result["warnings"])
    assert any("model_live_failed_used_cached=true: ecmwf_ifs" == w for w in result["warnings"])
    assert any("important vars missing: ecmwf_ifs (direct_normal_irradiance)" == w for w in result["warnings"])


def test_run_now_all_weather_models_failed_persists_error_run(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    _patch_core_for_run(monkeypatch)

    inserted = {}

    def fake_insert(_db_path, payload):
        inserted["payload"] = payload

    monkeypatch.setattr(backend_api, "insert_forecast_run", fake_insert)

    exc = RuntimeError("All weather model requests failed.")
    setattr(exc, "failed_models", ["ecmwf_ifs", "dwd_icon_d2"])
    setattr(
        exc,
        "failed_model_reasons",
        {
            "ecmwf_ifs": {"category": "provider_down", "message": "service unavailable"},
            "dwd_icon_d2": {"category": "timeout", "message": "request timed out"},
        },
    )

    def fail_ensemble(**_kwargs):
        raise exc

    monkeypatch.setattr(backend_api, "build_ensemble_forecast", fail_ensemble)

    result = state.run_now(
        backend_api.RunNowPayload(pv_uncertainty=False, weather_models=["ecmwf_ifs", "dwd_icon_d2"])
    )["result"]

    assert result["status"] == "error"
    assert result["warnings_count"] == 3
    assert "all weather model requests failed" in result["warnings"]
    assert "model failed: ecmwf_ifs (service unavailable)" in result["warnings"]
    assert "model failed: dwd_icon_d2 (request timed out)" in result["warnings"]
    assert result["inputs_used"]["weather_models_selected"] == ["ecmwf_ifs", "dwd_icon_d2"]
    assert result["weather_ensemble"]["failed_models"] == ["ecmwf_ifs", "dwd_icon_d2"]
    assert result["weather_ensemble"]["failure_reasons_by_model"]["ecmwf_ifs"]["category"] == "provider_down"
    assert inserted["payload"]["status"] == "error"
    assert inserted["payload"]["warnings_count"] == 3
    assert len(inserted["payload"].get("run_events", [])) >= 1


def test_run_now_includes_provider_payloads_when_enabled(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    _patch_core_for_run(monkeypatch)
    state.settings["config"].setdefault("weather", {})["store_provider_payloads"] = True

    idx = pd.date_range("2026-01-10", periods=2, freq="h", tz="Europe/Brussels")
    weather_df = pd.DataFrame({"temp_air_c": [10.0, 11.0], "wind_speed_ms": [1.0, 1.0], "cloud_cover_pct": [20.0, 30.0]}, index=idx)
    weather = core.ForecastResult(df=weather_df, sunrise=idx[0].to_pydatetime(), sunset=idx[-1].to_pydatetime())

    ensemble = _fake_ensemble(idx, weather, pv_uncertainty=False)
    ensemble.provider_payloads_by_model = {
        "ecmwf_ifs": {
            "fetched_at_utc": "2026-01-09T22:00:00+00:00",
            "endpoint": "https://api.open-meteo.com/v1/ecmwf",
            "params": {"models": "ifs"},
            "response_headers": {"content-type": "application/json"},
            "response_json": "{\"hourly\":{}}",
            "http_status": 200,
            "latency_ms": 123,
        }
    }

    monkeypatch.setattr(backend_api, "build_ensemble_forecast", lambda **_kwargs: ensemble)

    result = state.run_now(backend_api.RunNowPayload(pv_uncertainty=False, weather_models=["ecmwf_ifs"]))["result"]

    assert "provider_payloads_by_model" in result
    assert result["provider_payloads_by_model"]["ecmwf_ifs"]["http_status"] == 200


def test_run_now_model_failure_persists_run_events_to_sqlite(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_api, "LOCAL_STATE_DIR", tmp_path / "local_state")
    monkeypatch.setattr(backend_api, "SETTINGS_PATH", backend_api.LOCAL_STATE_DIR / "settings.json")
    monkeypatch.setattr(backend_api, "INPUTS_PATH", backend_api.LOCAL_STATE_DIR / "last_inputs.json")
    monkeypatch.setattr(backend_api, "LATEST_RESULT_PATH", backend_api.LOCAL_STATE_DIR / "latest_result.json")
    monkeypatch.setattr(backend_api, "HISTORY_PATH", backend_api.LOCAL_STATE_DIR / "results_history.json")
    monkeypatch.setattr(backend_api, "SQLITE_PATH", backend_api.LOCAL_STATE_DIR / "planner_history.sqlite")
    monkeypatch.setattr(backend_api, "TOKEN_PATH", backend_api.LOCAL_STATE_DIR / "api_token.txt")
    monkeypatch.setattr(backend_api, "RUN_HISTORY_PATH", tmp_path / "run_history_log.json")
    monkeypatch.setattr(backend_api, "fetch_latest_full_run", lambda *_args, **_kwargs: None)

    state = backend_api.BackendState()
    _patch_core_for_run_without_insert(monkeypatch)

    exc = RuntimeError("All weather model requests failed.")
    setattr(exc, "failed_models", ["ecmwf_ifs"])
    setattr(exc, "failed_model_reasons", {"ecmwf_ifs": {"category": "http_error", "message": "HTTP 502"}})

    monkeypatch.setattr(backend_api, "build_ensemble_forecast", lambda **_kwargs: (_ for _ in ()).throw(exc))

    result = state.run_now(backend_api.RunNowPayload(pv_uncertainty=False, weather_models=["ecmwf_ifs"]))["result"]

    assert result["status"] == "error"
    with sqlite3.connect(backend_api.SQLITE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT COUNT(*) AS cnt FROM run_events WHERE run_id = ?", (result["run_id"],)).fetchone()

    assert row is not None
    assert int(row["cnt"]) >= 1
