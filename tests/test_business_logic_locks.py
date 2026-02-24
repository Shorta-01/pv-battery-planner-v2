import datetime as dt
from types import SimpleNamespace
from pathlib import Path
import sys

import pandas as pd

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


def _fake_ensemble(idx, weather, pv_uncertainty):
    p50 = pd.Series([1.0, 2.0], index=idx)
    p25 = pd.Series([0.8, 1.5], index=idx) if pv_uncertainty else None
    p10 = pd.Series([0.5, 1.0], index=idx) if pv_uncertainty else None
    p90 = pd.Series([1.5, 3.0], index=idx) if pv_uncertainty else None
    return SimpleNamespace(
        weather_primary=weather,
        weather_primary_model_id="ecmwf_ifs",
        weather_ensemble_table=SimpleNamespace(df=weather.df.copy()),
        weather_by_model={"ecmwf_ifs": weather},
        pv_by_model={"ecmwf_ifs": pd.DataFrame({"pv_total_kwh": p50}, index=idx)},
        pv_ensemble_east_p50=p50 / 2,
        pv_ensemble_south_p50=p50 / 2,
        pv_ensemble_unclipped_p50=p50,
        pv_ensemble_p50=p50,
        pv_ensemble_p25=p25,
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


def _patch_minimal_core(monkeypatch, warning_tuple=None):
    monkeypatch.setattr(core, "ensure_pv_columns", lambda df, split_ratio=None: df)
    monkeypatch.setattr(core, "apply_daylight_clamp", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(core, "add_sun_percent", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(core, "add_load_and_surplus_columns", lambda df, *_args, **_kwargs: df.assign(load_kwh=0.5))
    monkeypatch.setattr(core, "compute_euro_savings_no_battery_vs_plan", lambda *_args, **_kwargs: {})

    def _run_plan(*_args, **_kwargs):
        idx = pd.date_range("2026-01-10", periods=2, freq="h", tz="Europe/Brussels")
        detail = pd.DataFrame(index=idx)
        flows = pd.DataFrame({"grid_import_kwh": [0.0, 0.0], "grid_export_kwh": [0.0, 0.0]}, index=idx)
        soc = pd.Series([0.4, 0.4], index=idx)
        if warning_tuple is not None:
            return detail, flows, soc, 1.0, 0.55, "bridge rationale", warning_tuple[0], warning_tuple[1], warning_tuple[2]
        return detail, flows, soc, 1.0, 0.55, "bridge rationale", "Automatically computed.", True, ""

    monkeypatch.setattr(core, "run_detailed_plan", _run_plan)


def test_confidence_does_not_change_decision_quantile():
    assert backend_api.pick_decision_quantile("Low", uncertainty_enabled=True) == ("p25", "fixed")
    assert backend_api.pick_decision_quantile("High", uncertainty_enabled=True) == ("p25", "fixed")
    assert backend_api.pick_decision_quantile("Low", uncertainty_enabled=False) == ("p50", "uncertainty_disabled")


def test_unreachable_target_warning_propagates(monkeypatch, tmp_path):
    state = _new_state(monkeypatch, tmp_path)
    _patch_minimal_core(monkeypatch, warning_tuple=("Warning: Cutoff may be unreachable.", False, "Warning: Cutoff may be unreachable."))

    idx = pd.date_range("2026-01-10", periods=2, freq="h", tz="Europe/Brussels")
    weather_df = pd.DataFrame({"temp_air_c": [10.0, 11.0], "wind_speed_ms": [1.0, 1.0], "cloud_cover_pct": [20.0, 30.0]}, index=idx)
    weather = core.ForecastResult(df=weather_df, sunrise=idx[0].to_pydatetime(), sunset=idx[-1].to_pydatetime())
    monkeypatch.setattr(backend_api, "build_ensemble_forecast", lambda *, pv_uncertainty, **_kwargs: _fake_ensemble(idx, weather, pv_uncertainty))

    result = state.run_now(backend_api.RunNowPayload(pv_uncertainty=True, weather_models=["ecmwf_ifs"]))["result"]
    metrics = result["metrics"]
    assert metrics["charge_target_reachable"] is False
    assert metrics["charge_warning_text"].startswith("Warning")
    assert metrics["charge_note"].startswith("Warning")


def test_flat_tariff_suppresses_arbitrage_charge():
    charge_date = dt.date(2026, 1, 11)
    tariff_cfg = {
        "peak_grid_price_eur_per_kwh": 0.30,
        "offpeak_grid_price_eur_per_kwh": 0.30,
        "offpeak_windows_by_dow": [[("00:00", "06:00")]] * 7,
    }
    required_grid_kwh, charge_kw, note, achieved_soc = core.plan_charge_power(
        soc_start=0.20,
        soc_cutoff=0.60,
        charge_date=charge_date,
        user_cap_kw=5.0,
        tariff_cfg=tariff_cfg,
    )

    assert required_grid_kwh == 0.0
    assert charge_kw == 0.0
    assert achieved_soc == 0.20
    assert "no meaningful spread" in note.lower()
