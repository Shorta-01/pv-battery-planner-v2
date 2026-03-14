import datetime as dt
from types import SimpleNamespace

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

import backend_api
import planner_core as core


def _configure_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_api, "LOCAL_STATE_DIR", tmp_path / "local_state")
    monkeypatch.setattr(backend_api, "SETTINGS_PATH", backend_api.LOCAL_STATE_DIR / "settings.json")
    monkeypatch.setattr(backend_api, "INPUTS_PATH", backend_api.LOCAL_STATE_DIR / "last_inputs.json")
    monkeypatch.setattr(backend_api, "LATEST_RESULT_PATH", backend_api.LOCAL_STATE_DIR / "latest_result.json")
    monkeypatch.setattr(backend_api, "HISTORY_PATH", backend_api.LOCAL_STATE_DIR / "results_history.json")
    monkeypatch.setattr(backend_api, "SQLITE_PATH", backend_api.LOCAL_STATE_DIR / "planner_history.sqlite")
    monkeypatch.setattr(backend_api, "TOKEN_PATH", backend_api.LOCAL_STATE_DIR / "api_token.txt")
    monkeypatch.setattr(backend_api, "RUN_HISTORY_PATH", tmp_path / "run_history_log.json")


class _FakeBmwService:
    def __init__(self):
        self._vehicles = {"v1": {"soc_pct": 62.0, "data_status": "fresh", "freshness_seconds": 10}}

    def vehicles(self):
        return self._vehicles

    def manual_refresh(self, *, force_reprobe=False):
        _ = force_reprobe
        return {"ok": True, "refreshed": True}

    def provider_status(self):
        return {"provider": "bmw", "ok": True}


@pytest.fixture()
def deterministic_state(monkeypatch, tmp_path):
    _configure_paths(monkeypatch, tmp_path)

    monkeypatch.setattr(core, "ensure_pv_columns", lambda df, split_ratio=None: df)
    monkeypatch.setattr(core, "apply_daylight_clamp", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(core, "add_sun_percent", lambda df, *_args, **_kwargs: df)

    def _add_load(df, yesterday_kwh):
        out = df.copy()
        out["load_kwh"] = float(yesterday_kwh) / max(len(out.index), 1)
        return out

    monkeypatch.setattr(core, "add_load_and_surplus_columns", _add_load)
    monkeypatch.setattr(core, "compute_soc_low_timing_aware", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(core, "compute_soc_high_headroom", lambda *_args, **_kwargs: (0.0, 0.9))
    monkeypatch.setattr(core, "choose_cutoff_soc", lambda *_args, **_kwargs: (0.5, "ok"))
    monkeypatch.setattr(core, "simulate_expensive_hours_detailed", lambda pv, *_args, **_kwargs: (pd.DataFrame(index=pv.index), 1.0, 0.3, 0.0, 0.0))
    monkeypatch.setattr(core, "simulate_full_day_soc", lambda pv, *_args, **_kwargs: (pd.Series([0.5] * len(pv.index), index=pv.index), pd.DataFrame(index=pv.index)))
    monkeypatch.setattr(core, "estimate_pv_with_pvlib", lambda clear_df, *_args, **_kwargs: (None, None, None, None, None, pd.Series([2.0] * len(clear_df.index), index=clear_df.index)))
    monkeypatch.setattr(core, "compute_euro_savings_no_battery_vs_plan", lambda *_args, **_kwargs: {"savings_eur": 0.2})
    monkeypatch.setattr(core, "resolve_location_metadata", lambda **kwargs: {"timezone": kwargs.get("timezone") or "Europe/Brussels", "elevation_m": kwargs.get("elevation_m") or 30.0, "warnings": []})

    def _fake_run_detailed_plan(*_args, **_kwargs):
        idx = _kwargs["pv_df"].index
        flows = pd.DataFrame(index=idx)
        flows.attrs["charge_effective_cap_kw"] = 5.0
        flows.attrs["charge_limit_reason_raw"] = "none"
        flows.attrs["grid_import_cap_binding_events"] = 0
        flows.attrs["grid_import_cap_load_exceeds_events"] = 0
        flows.attrs["grid_import_cap_limited_charge_kwh_total"] = 0.0
        return pd.DataFrame(index=idx), flows, pd.Series([0.5] * len(idx), index=idx), 2.0, 0.55, "ok", "stable", True, ""

    monkeypatch.setattr(core, "run_detailed_plan", _fake_run_detailed_plan)

    idx = pd.date_range("2026-01-10", periods=4, freq="h", tz="Europe/Brussels")
    weather_df = pd.DataFrame(
        {
            "temp_air_c": [9.0, 10.0, 11.0, 12.0],
            "wind_speed_ms": [1.0, 1.0, 1.0, 1.0],
            "cloud_cover_pct": [10.0, 20.0, 30.0, 40.0],
        },
        index=idx,
    )
    weather = core.ForecastResult(df=weather_df, sunrise=idx[0].to_pydatetime(), sunset=idx[-1].to_pydatetime())

    def _ensemble(*, pv_uncertainty=False, **_kwargs):
        p50 = pd.Series([0.5, 0.8, 1.2, 1.5], index=idx)
        p10 = pd.Series([0.2, 0.5, 0.8, 1.0], index=idx) if pv_uncertainty else None
        p90 = pd.Series([0.8, 1.2, 1.7, 2.1], index=idx) if pv_uncertainty else None
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
                        "pv_clipped_kwh": pd.Series([0.0] * len(idx), index=idx),
                    },
                    index=idx,
                )
            },
            pv_ensemble_east_p50=p50 / 2,
            pv_ensemble_south_p50=p50 / 2,
            pv_ensemble_unclipped_p50=p50,
            pv_ensemble_p50=p50,
            pv_ensemble_p25=p10,
            pv_ensemble_p10=p10,
            pv_ensemble_p90=p90,
            selected_models=["ecmwf_ifs"],
            weights_used={"ecmwf_ifs": 1.0},
            per_model_pv_totals_kwh={"ecmwf_ifs": float(p50.sum())},
            missing_vars_by_model={},
            derived_irradiance_by_model={},
            derived_weather_code_by_model={},
            derived_irradiance_hours_by_model={},
            quality_weight_factors_by_model={},
            fetch_meta_by_model={},
            failed_models=[],
            failed_model_reasons={},
            model_live_failed_used_cached={},
            provider_payloads_by_model={},
        )

    monkeypatch.setattr(backend_api, "build_ensemble_forecast", _ensemble)
    state = backend_api.BackendState()
    state.bmw_service = _FakeBmwService()
    backend_api.state = state
    return state


def _normalize_run_payload(payload: dict) -> dict:
    return {
        "status": payload["status"],
        "warnings_count": payload["warnings_count"],
        "weather_primary_model_id": payload["weather_primary_model_id"],
        "pv_totals_kwh": payload["pv_totals_kwh"],
        "metrics": {
            "charge_kw": payload["metrics"]["charge_kw"],
            "cutoff_soc": payload["metrics"]["cutoff_soc"],
            "pv_forecast_kwh": payload["metrics"]["pv_forecast_kwh"],
            "cons_forecast_kwh": payload["metrics"]["cons_forecast_kwh"],
            "pv_decision_scenario": payload["metrics"]["pv_decision_scenario"],
        },
        "inputs_used": {
            "pv_uncertainty_enabled": payload["inputs_used"]["pv_uncertainty_enabled"],
            "weather_models_selected": payload["inputs_used"]["weather_models_selected"],
            "soc_now_percent": payload["inputs_used"]["soc_now_percent"],
        },
    }


@pytest.mark.parametrize(
    "payload_kwargs,expected_pv",
    [
        ({"pv_uncertainty": False, "weather_models": ["ecmwf_ifs"]}, {"p10": None, "p50": 4.0, "p90": None}),
        ({"pv_uncertainty": True, "weather_models": ["ecmwf_ifs"]}, {"p10": 2.5, "p50": 4.0, "p90": 5.8}),
        ({"pv_uncertainty": False, "weather_models": ["ecmwf_ifs"], "forecast_mode": "expert"}, {"p10": None, "p50": 4.0, "p90": None}),
    ],
)
def test_forecast_run_characterization_scenarios(deterministic_state, payload_kwargs, expected_pv):
    out = deterministic_state.run_now(backend_api.RunNowPayload(**payload_kwargs))["result"]
    normalized = _normalize_run_payload(out)
    assert normalized["status"] in {"ok", "degraded"}
    assert normalized["pv_totals_kwh"]["p10"] == expected_pv["p10"]
    assert normalized["pv_totals_kwh"]["p50"] == pytest.approx(expected_pv["p50"], abs=1e-9)
    if expected_pv["p90"] is None:
        assert normalized["pv_totals_kwh"]["p90"] is None
    else:
        assert normalized["pv_totals_kwh"]["p90"] == pytest.approx(expected_pv["p90"], abs=1e-9)
    assert normalized["weather_primary_model_id"] == "ecmwf_ifs"
    assert normalized["metrics"]["pv_forecast_kwh"] == pytest.approx(4.0, abs=1e-9)
    assert out["forecast_mode_effective"] in {"auto", "expert"}


def test_planner_characterization_low_pv_high_load(monkeypatch, deterministic_state):
    monkeypatch.setattr(core, "run_detailed_plan", lambda *_args, **kwargs: (pd.DataFrame(index=kwargs["pv_df"].index), pd.DataFrame(index=kwargs["pv_df"].index), pd.Series([0.35] * len(kwargs["pv_df"].index), index=kwargs["pv_df"].index), 4.6, 0.75, "load_dominant", "charge", True, ""))
    result = deterministic_state.run_now(backend_api.RunNowPayload(soc_now_percent=18.0, yesterday_consumption_kwh=34.0))["result"]
    assert result["metrics"]["charge_kw"] == pytest.approx(4.6, abs=1e-9)
    assert result["metrics"]["cutoff_soc"] == pytest.approx(0.75, abs=1e-9)
    assert result["inputs_used"]["soc_now_percent"] == pytest.approx(62.0, abs=1e-9)


def test_planner_characterization_strong_pv_high_soc(monkeypatch, deterministic_state):
    monkeypatch.setattr(core, "run_detailed_plan", lambda *_args, **kwargs: (pd.DataFrame(index=kwargs["pv_df"].index), pd.DataFrame(index=kwargs["pv_df"].index), pd.Series([0.85] * len(kwargs["pv_df"].index), index=kwargs["pv_df"].index), 0.0, 0.3, "pv_abundant", "no_charge", True, ""))
    result = deterministic_state.run_now(backend_api.RunNowPayload(soc_now_percent=92.0, yesterday_consumption_kwh=8.0))["result"]
    assert result["metrics"]["charge_kw"] == pytest.approx(0.0, abs=1e-9)
    assert result["metrics"]["cutoff_soc"] == pytest.approx(0.3, abs=1e-9)


def test_planner_characterization_grid_import_cap_warning(monkeypatch, deterministic_state):
    def _capped(*_args, **kwargs):
        idx = kwargs["pv_df"].index
        flows = pd.DataFrame(index=idx)
        flows.attrs["charge_effective_cap_kw"] = 2.0
        flows.attrs["charge_limit_reason_raw"] = "grid_import_cap"
        flows.attrs["grid_import_cap_binding_events"] = 2
        flows.attrs["grid_import_cap_load_exceeds_events"] = 1
        flows.attrs["grid_import_cap_limited_charge_kwh_total"] = 1.4
        return pd.DataFrame(index=idx), flows, pd.Series([0.45] * len(idx), index=idx), 2.0, 0.65, "capped", "limited", False, "cap limited"

    monkeypatch.setattr(core, "run_detailed_plan", _capped)
    result = deterministic_state.run_now(backend_api.RunNowPayload())["result"]
    assert result["metrics"]["grid_import_cap_binding"] is True
    assert result["metrics"]["grid_import_cap_limited_charge_kwh"] == pytest.approx(1.4, abs=1e-9)
    assert any("Grid import cap limited battery charging" in w for w in result["warnings"])


def test_results_history_and_latest_contracts(deterministic_state):
    deterministic_state.run_now(backend_api.RunNowPayload(weather_models=["ecmwf_ifs"]))
    deterministic_state.run_now(backend_api.RunNowPayload(weather_models=["ecmwf_ifs"]))

    latest = backend_api.latest_result(authorization=f"Bearer {deterministic_state.api_token}")
    history = backend_api.history(days=7, show_all_runs=True, authorization=f"Bearer {deterministic_state.api_token}")

    assert {"run_id", "target_date", "status"}.issubset(latest.keys())
    assert len(history["items"]) >= 2
    assert {"run_id", "target_date", "status"}.issubset(history["items"][0].keys())


def test_run_id_lookup_returns_full_payload(deterministic_state):
    run_payload = deterministic_state.run_now(backend_api.RunNowPayload(weather_models=["ecmwf_ifs"]))["result"]
    fetched = backend_api.result_by_run_id(run_payload["run_id"], authorization=f"Bearer {deterministic_state.api_token}")
    assert fetched["run_id"] == run_payload["run_id"]
    assert fetched["metrics"]["pv_forecast_kwh"] == pytest.approx(run_payload["metrics"]["pv_forecast_kwh"], abs=1e-9)


def test_error_event_persistence_dedupe_and_fixed_toggle(deterministic_state):
    auth = f"Bearer {deterministic_state.api_token}"
    payload = backend_api.ErrorEventPayload(
        source="frontend",
        severity="error",
        error_type="exception",
        where="ui:test",
        title="Crash",
        body="boom",
        context={"tab": "planner"},
    )
    first = backend_api.post_error(payload, authorization=auth)
    second = backend_api.post_error(payload, authorization=auth)
    assert first["error_id"] == second["error_id"]

    backend_api.post_error_fixed(first["error_id"], backend_api.ErrorFixedPayload(fixed=True), authorization=auth)
    details = backend_api.get_error_by_id(first["error_id"], authorization=auth)
    assert details["fixed"] == 1


def test_error_event_delete_paths(deterministic_state):
    auth = f"Bearer {deterministic_state.api_token}"
    p = backend_api.ErrorEventPayload(source="backend", severity="warning", error_type="validation", where="/v1/run/now", title="Bad", body="bad")
    error_id = backend_api.post_error(p, authorization=auth)["error_id"]
    backend_api.delete_one_error(error_id, authorization=auth)
    rows = backend_api.get_errors(limit=10, include_fixed=True, authorization=auth)
    assert error_id not in {row["error_id"] for row in rows["items"]}


def test_ev_bmw_endpoints_mocked_flow(deterministic_state):
    auth = f"Bearer {deterministic_state.api_token}"
    refresh = backend_api.ev_manual_refresh(False, authorization=auth)
    status = backend_api.get_ev_provider_status(authorization=auth)

    assert refresh["ok"] is True
    assert status["provider"] == "bmw"


def test_startup_runtime_split_validation_called(monkeypatch, deterministic_state):
    _ = deterministic_state
    calls = {"count": 0}

    def _validate(require_production_quality=True):
        assert require_production_quality is True
        calls["count"] += 1
        return True

    monkeypatch.setattr(backend_api, "validate_pvlib_runtime", _validate)
    backend_api._validate_forecast_runtime_dependencies()
    assert calls["count"] == 1


def test_startup_runtime_split_validation_failure_blocks_app(monkeypatch, deterministic_state):
    _ = deterministic_state
    monkeypatch.setattr(backend_api, "validate_pvlib_runtime", lambda require_production_quality=True: (_ for _ in ()).throw(RuntimeError("pvlib missing")))
    with pytest.raises(RuntimeError, match="pvlib missing"):
        backend_api._validate_forecast_runtime_dependencies()
