from __future__ import annotations

import copy
import datetime as dt
import gc
import json
import os
import secrets
import threading
import tempfile
import traceback
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import planner_core as core
from db_sqlite import (
    compute_config_hash,
    fetch_history_all_runs,
    fetch_history_latest_per_day,
    fetch_latest_full_run,
    init_db,
    insert_forecast_run,
)
from weather_ensemble import (
    DEFAULT_ACCURACY_MODELS,
    build_ensemble_forecast,
    weather_models_payload,
)

LOCAL_STATE_DIR = Path("local_state")
SETTINGS_PATH = LOCAL_STATE_DIR / "settings.json"
INPUTS_PATH = LOCAL_STATE_DIR / "last_inputs.json"
LATEST_RESULT_PATH = LOCAL_STATE_DIR / "latest_result.json"
HISTORY_PATH = LOCAL_STATE_DIR / "results_history.json"
SQLITE_PATH = LOCAL_STATE_DIR / "planner_history.sqlite"
RUN_HISTORY_PATH = Path("run_history_log.json")
TOKEN_PATH = LOCAL_STATE_DIR / "api_token.txt"
DEFAULT_NIGHTLY_TIME = "22:00"
DEFAULT_MAX_AC_CAP = 5.0
MAX_HISTORY = 30

PV_QUALITY_COLORS = {
    "Excellent": "#2a9d8f",
    "Good": "#52b788",
    "Mixed": "#f4a261",
    "Poor": "#e76f51",
    "Very low": "#d62828",
}

FULL_RESULT_HEAVY_KEYS = {"weather", "pv", "detail", "flows", "soc"}
DEBUG = os.getenv("DEBUG", "").strip() in ("1", "true", "True", "yes", "YES")


def _to_history_summary(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    pv_quality = payload.get("pv_quality") if isinstance(payload.get("pv_quality"), dict) else {}
    slim_pv_quality = {
        "score": pv_quality.get("score"),
        "label": pv_quality.get("label"),
        "ratio": pv_quality.get("ratio"),
        "color": pv_quality.get("color"),
        "is_fallback": pv_quality.get("is_fallback"),
    }
    return {
        "target_date": payload.get("target_date"),
        "metrics": payload.get("metrics", {}),
        "pv_quality": {k: v for k, v in slim_pv_quality.items() if v is not None},
        "warnings": payload.get("warnings", []),
        "run_at": payload.get("run_at"),
        "run_type": payload.get("run_type", "manual"),
    }


class SettingsPayload(BaseModel):
    config: dict
    nightly_run_time: str = DEFAULT_NIGHTLY_TIME
    timezone: str = "Europe/Brussels"
    max_ac_charge_power_kw_default: float = DEFAULT_MAX_AC_CAP


class InputsPayload(BaseModel):
    soc_at_22_percent: float = Field(..., ge=0.0, le=100.0)
    yesterday_consumption_kwh: float = Field(..., gt=0.0)


class RunNowPayload(BaseModel):
    soc_at_22_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    yesterday_consumption_kwh: float | None = Field(default=None, gt=0.0)
    buffer_percent: float = Field(default=0.0, ge=0.0, le=10.0)
    user_max_ac_kw: float | None = Field(default=None, ge=0.0)
    weather_models: list[str] | None = None
    ensemble_method: str = Field(default="weighted")
    pv_uncertainty: bool = False
    fast_mode: bool = False


class NightlyTickPayload(BaseModel):
    force: bool = False


class BackendState:
    def __init__(self) -> None:
        LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
        init_db(str(SQLITE_PATH))
        self._lock = threading.Lock()
        self.api_token = self._load_or_create_token()
        self.settings = self._load_settings()
        self.last_inputs = self._read_json(INPUTS_PATH, default={})
        self.latest_result = self._read_json(LATEST_RESULT_PATH, default={})
        self.history = self._load_history()
        self._apply_config(self.settings["config"])
        self._migrate_json_history_to_sqlite()

    def _migrate_json_history_to_sqlite(self) -> None:
        payloads: list[dict] = []
        raw_history = self._read_json(HISTORY_PATH, default=[])
        if isinstance(raw_history, list):
            payloads.extend([item for item in raw_history if isinstance(item, dict)])
        latest_payload = self._read_json(LATEST_RESULT_PATH, default={})
        if isinstance(latest_payload, dict) and latest_payload:
            payloads.append(latest_payload)

        run_history_log = self._read_json(RUN_HISTORY_PATH, default=[])
        fallback_by_date: dict[str, dict] = {}
        if isinstance(run_history_log, list):
            for row in run_history_log:
                if not isinstance(row, dict):
                    continue
                d = str(row.get("Date") or "")
                if not d:
                    continue
                fallback_by_date[d] = row

        for payload in payloads:
            if "metrics" not in payload or not isinstance(payload.get("metrics"), dict):
                payload["metrics"] = {}
            metrics = payload["metrics"]
            date_key = str(payload.get("target_date") or "")
            fb = fallback_by_date.get(date_key, {})
            if "charge_kw" not in metrics and fb:
                metrics["charge_kw"] = float(fb.get("Allowed AC charge power (kW)", 0.0) or 0.0)
            if "cutoff_soc" not in metrics and fb:
                cutoff_pct = float(fb.get("AC charge cutoff SOC (%)", 0.0) or 0.0)
                metrics["cutoff_soc"] = cutoff_pct / 100.0
            payload.setdefault("run_id", str(uuid.uuid4()))
            payload.setdefault("run_at_utc", dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat())
            payload.setdefault("config", self.settings.get("config", {}))
            if "inputs_used" not in payload and self.last_inputs:
                payload["inputs_used"] = {
                    "soc_at_22_percent": self.last_inputs.get("soc_at_22_percent"),
                    "yesterday_consumption_kwh": self.last_inputs.get("yesterday_consumption_kwh"),
                }
            insert_forecast_run(str(SQLITE_PATH), payload)

    def _is_heavy_history_item(self, item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        return any(key in item for key in FULL_RESULT_HEAVY_KEYS)

    def _load_history(self) -> list[dict]:
        raw_history = self._read_json(HISTORY_PATH, default=[])
        if not isinstance(raw_history, list):
            return []

        migrated = False
        summaries: list[dict] = []
        for item in raw_history:
            if not isinstance(item, dict):
                continue
            if self._is_heavy_history_item(item):
                migrated = True
            summaries.append(_to_history_summary(item))

        summaries = summaries[-MAX_HISTORY:]
        if migrated:
            self._write_json(HISTORY_PATH, summaries)
        return summaries

    def _load_or_create_token(self) -> str:
        if TOKEN_PATH.exists():
            return TOKEN_PATH.read_text(encoding="utf-8").strip()
        token = secrets.token_urlsafe(32)
        TOKEN_PATH.write_text(token, encoding="utf-8")
        return token

    def _read_json(self, path: Path, default):
        if not path.exists():
            return copy.deepcopy(default)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Warning: invalid JSON in {path.name}; using defaults")
            return copy.deepcopy(default)
        except OSError:
            return copy.deepcopy(default)

    def _write_json(self, path: Path, payload: dict | list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as tmp:
            json.dump(payload, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)

        os.replace(tmp_path, path)

    def _load_settings(self) -> dict:
        if SETTINGS_PATH.exists():
            loaded = self._read_json(SETTINGS_PATH, {})
            if isinstance(loaded, dict) and "config" in loaded:
                return loaded

        cfg = core.DEFAULT_CONFIG
        if core.CONFIG_PATH.exists():
            cfg = core.load_config_file(core.CONFIG_PATH)
        merged = core.set_user_config(cfg)
        settings = {
            "config": merged,
            "nightly_run_time": DEFAULT_NIGHTLY_TIME,
            "timezone": "Europe/Brussels",
            "max_ac_charge_power_kw_default": DEFAULT_MAX_AC_CAP,
        }
        self._write_json(SETTINGS_PATH, settings)
        return settings

    def _save_settings(self) -> None:
        self._write_json(SETTINGS_PATH, self.settings)

    def _save_inputs(self) -> None:
        self._write_json(INPUTS_PATH, self.last_inputs)

    def _save_results(self) -> None:
        self._write_json(LATEST_RESULT_PATH, self.latest_result)
        self.history = self.history[-MAX_HISTORY:]
        self._write_json(HISTORY_PATH, self.history)

    def _tzinfo(self) -> ZoneInfo:
        loc_cfg = self.settings.get("config", {}).get("location", {}) if isinstance(self.settings.get("config"), dict) else {}
        tz_name = str(loc_cfg.get("timezone") or self.settings.get("timezone") or "Europe/Brussels")
        try:
            return ZoneInfo(tz_name)
        except Exception:
            return ZoneInfo("Europe/Brussels")

    def _apply_config(self, config: dict) -> dict:
        merged = core.set_user_config(config)
        self.settings["config"] = merged
        return merged

    def update_settings(self, payload: SettingsPayload) -> dict:
        _ = ZoneInfo(payload.timezone)
        dt.datetime.strptime(payload.nightly_run_time, "%H:%M")
        merged = self._apply_config(payload.config)
        loc_cfg = merged.get("location", {}) if isinstance(merged, dict) else {}
        canonical_tz = str(loc_cfg.get("timezone") or payload.timezone)
        self.settings.update(
            {
                "config": merged,
                "nightly_run_time": payload.nightly_run_time,
                "timezone": canonical_tz,
                "max_ac_charge_power_kw_default": float(payload.max_ac_charge_power_kw_default),
            }
        )
        self._save_settings()
        return self.settings

    def update_inputs(self, payload: InputsPayload) -> dict:
        now_local = dt.datetime.now(self._tzinfo()).isoformat()
        self.last_inputs = {
            "soc_at_22_percent": float(payload.soc_at_22_percent),
            "yesterday_consumption_kwh": float(payload.yesterday_consumption_kwh),
            "last_inputs_updated_at": now_local,
        }
        self._save_inputs()
        return self.last_inputs

    def _serialize_df(self, df: pd.DataFrame) -> dict:
        return json.loads(df.to_json(date_format="iso", orient="split"))

    def _serialize_series(self, s: pd.Series) -> dict:
        frame = s.to_frame(name="value")
        return self._serialize_df(frame)

    def _run(
        self,
        target_date: dt.date,
        soc_percent: float,
        yesterday_kwh: float,
        buffer_percent: float,
        user_max_ac_kw: float,
        weather_models: list[str] | None,
        ensemble_method: str,
        pv_uncertainty: bool,
        fast_mode: bool = False,
    ) -> dict:
        cfg = self.settings["config"]
        loc_cfg = cfg.get("location", {})
        tz = str(loc_cfg.get("timezone", "Europe/Brussels"))
        loc = core.Location(
            name=str(loc_cfg.get("address_query") or loc_cfg.get("name") or "Configured"),
            latitude=float(loc_cfg["latitude"]),
            longitude=float(loc_cfg["longitude"]),
        )

        selected_models = weather_models if weather_models is not None else DEFAULT_ACCURACY_MODELS
        if not selected_models:
            raise HTTPException(status_code=400, detail="Select at least one weather model.")

        ensemble = build_ensemble_forecast(
            loc=loc,
            target_date=target_date,
            tz=tz,
            weather_models=selected_models,
            ensemble_method=str(ensemble_method).lower().strip(),
            pv_uncertainty=bool(pv_uncertainty),
            accuracy_mode=True,
            fast_mode=bool(fast_mode),
        )

        weather = ensemble.weather_primary
        pv = pd.DataFrame(index=ensemble.pv_ensemble_p50.index)
        pv["pv_east_kwh"] = ensemble.pv_ensemble_east_p50.reindex(pv.index).fillna(0.0)
        pv["pv_south_kwh"] = ensemble.pv_ensemble_south_p50.reindex(pv.index).fillna(0.0)
        pv["pv_total_unclipped_kwh"] = ensemble.pv_ensemble_unclipped_p50.reindex(pv.index).fillna(0.0)
        pv["pv_total_kwh"] = ensemble.pv_ensemble_p50.reindex(pv.index).fillna(0.0)
        pv["pv_clipped_kwh"] = (pv["pv_total_unclipped_kwh"] - pv["pv_total_kwh"]).clip(lower=0.0)
        pv["pv_dc_available_kwh"] = pv["pv_total_unclipped_kwh"]
        pv["pv_ac_limited_kwh"] = pv["pv_total_kwh"]

        total_e = float(pd.to_numeric(pv["pv_east_kwh"], errors="coerce").fillna(0.0).sum())
        total_s = float(pd.to_numeric(pv["pv_south_kwh"], errors="coerce").fillna(0.0).sum())
        split_total = total_e + total_s
        ratio_e, ratio_s = (total_e / split_total, total_s / split_total) if split_total > 0 else (0.0, 1.0)

        pv = core.ensure_pv_columns(pv, split_ratio=(ratio_e, ratio_s))
        pv = core.apply_daylight_clamp(pv, weather.sunrise, weather.sunset).sort_index()
        pv = core.ensure_pv_columns(pv, split_ratio=(ratio_e, ratio_s))
        pv = core.add_sun_percent(pv, weather.sunrise, weather.sunset)
        pv = core.add_load_and_surplus_columns(pv, yesterday_kwh)

        tariff_cfg = cfg.get("tariff", core.DEFAULT_CONFIG["tariff"])
        soc_low = core.compute_soc_low_timing_aware(pv, yesterday_kwh, target_date, tariff_cfg=tariff_cfg)
        _, soc_high = core.compute_soc_high_headroom(pv, yesterday_kwh, target_date)
        cutoff_soc_raw, cutoff_reason = core.choose_cutoff_soc(target_date, soc_low, soc_high)
        cutoff_soc = min(max(cutoff_soc_raw + (float(buffer_percent) / 100.0), core.MIN_SOC), core.MAX_CUTOFF_SOC)
        charge_date = target_date - dt.timedelta(days=1)
        _, charge_kw, charge_note, achieved_soc_start = core.plan_charge_power(
            soc_percent / 100.0,
            cutoff_soc,
            charge_date,
            user_cap_kw=user_max_ac_kw,
        )
        detail_df, grid_import, grid_export, _, _ = core.simulate_expensive_hours_detailed(
            pv, yesterday_kwh, achieved_soc_start, target_date
        )
        soc_series, flows_df = core.simulate_full_day_soc(
            pv,
            yesterday_kwh,
            soc_percent / 100.0,
            charge_kw,
            cutoff_soc,
            target_date,
        )

        charge_date = target_date - dt.timedelta(days=1)

        try:
            clear_df = pd.DataFrame(index=weather.df.index)
            clear_df["temp_air_c"] = pd.to_numeric(weather.df.get("temp_air_c"), errors="coerce").fillna(10.0)
            clear_df["wind_speed_ms"] = pd.to_numeric(weather.df.get("wind_speed_ms"), errors="coerce").fillna(1.0).clip(lower=0.0)
            clear_df["cloud_cover_pct"] = 0.0
            _, _, _, _, _, pv_ac_limited_kwh = core.estimate_pv_with_pvlib(clear_df, loc, tz=tz)
            clear_kwh = float(pv_ac_limited_kwh.sum())
            pv_total_kwh = float(pd.to_numeric(pv.get("pv_total_kwh", 0.0), errors="coerce").fillna(0.0).sum())
            score = int(min(max(round(100 * pv_total_kwh / max(clear_kwh, 0.1)), 0), 100))
            pv_quality = {"score": score, "pv_total_kwh": pv_total_kwh, "ratio": score / 100.0, "is_fallback": False}
        except Exception:
            pv_quality = {"score": 0, "pv_total_kwh": float(pv["pv_total_kwh"].sum()), "ratio": 0.0, "is_fallback": True}

        for label, threshold in {
            "Excellent": 75,
            "Good": 55,
            "Mixed": 35,
            "Poor": 15,
            "Very low": 0,
        }.items():
            if pv_quality["score"] >= threshold:
                pv_quality["label"] = label
                break

        pv_quality["color"] = PV_QUALITY_COLORS.get(pv_quality.get("label", "Very low"), "#d62828")

        savings = core.compute_euro_savings_no_battery_vs_plan(
            pv_df=pv,
            flows_df=flows_df,
            soc_at_22=soc_percent / 100.0,
            charge_kw=float(charge_kw),
            cutoff_soc=float(cutoff_soc),
            today_date=charge_date,
            tomorrow_date=target_date,
            total_consumption_kwh=yesterday_kwh,
            tariff_cfg=tariff_cfg,
        )
        pv_quality.update(savings)

        pv_forecast_kwh = float(pv["pv_total_kwh"].fillna(0.0).sum()) if "pv_total_kwh" in pv.columns else 0.0
        cons_forecast_kwh = (
            float(pv["load_kwh"].fillna(0.0).sum())
            if "load_kwh" in pv.columns
            else float(yesterday_kwh)
        )

        run_at_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        run_id = str(uuid.uuid4())
        system_snapshot = {
            "lat": loc_cfg.get("latitude"),
            "lon": loc_cfg.get("longitude"),
            "timezone": tz,
            "tilt": cfg.get("arrays", {}).get("south", {}).get("tilt_deg"),
            "azimuth": cfg.get("arrays", {}).get("south", {}).get("azimuth_deg"),
            "dc_kwp": cfg.get("arrays", {}).get("south", {}).get("dc_capacity_kwp"),
            "battery_kwh": cfg.get("battery", {}).get("capacity_kwh"),
            "inverter_ac_limit_kw": cfg.get("inverter", {}).get("ac_limit_kw"),
            "loss_factor": cfg.get("system", {}).get("loss_factor"),
        }
        config_hash = compute_config_hash(cfg)

        payload = {
            "run_id": run_id,
            "target_date": target_date.isoformat(),
            "weather": self._serialize_df(weather.df),
            "pv": self._serialize_df(pv),
            "detail": self._serialize_df(detail_df),
            "flows": self._serialize_df(flows_df),
            "soc": self._serialize_series(soc_series),
            "sunrise": pd.Timestamp(weather.sunrise).isoformat(),
            "sunset": pd.Timestamp(weather.sunset).isoformat(),
            "metrics": {
                "charge_kw": float(charge_kw),
                "cutoff_soc": float(cutoff_soc),
                "cutoff_reason": cutoff_reason,
                "charge_note": charge_note,
                "grid_import": float(grid_import),
                "grid_export": float(grid_export),
                "pv_forecast_kwh": pv_forecast_kwh,
                "cons_forecast_kwh": cons_forecast_kwh,
            },
            "pv_quality": pv_quality,
            "warnings": [],
            "run_at": dt.datetime.now(self._tzinfo()).isoformat(),
            "run_at_utc": run_at_utc,
            "run_type": "manual",
            "timezone": tz,
            "inputs_used": {
                "soc_at_22_percent": float(soc_percent),
                "yesterday_consumption_kwh": float(yesterday_kwh),
            },
            "system_snapshot": {k: v for k, v in system_snapshot.items() if v is not None},
            "planner_version": "v2",
            "config_hash": config_hash,
            "config_json": json.dumps(
                {
                    **cfg,
                    "weather_models_selected": ensemble.selected_models,
                    "ensemble_method": ensemble_method,
                    "pv_uncertainty_enabled": bool(pv_uncertainty),
                    "per_model_pv_totals_kwh": ensemble.per_model_pv_totals_kwh,
                    "pv_totals_kwh": {
                        "p10": float(ensemble.pv_ensemble_p10.sum()) if ensemble.pv_ensemble_p10 is not None else None,
                        "p50": float(ensemble.pv_ensemble_p50.sum()),
                        "p90": float(ensemble.pv_ensemble_p90.sum()) if ensemble.pv_ensemble_p90 is not None else None,
                    },
                },
                sort_keys=True,
            ),
            "config": cfg,
            "created_at_utc": run_at_utc,
            "weather_ensemble": {
                "selected_models": ensemble.selected_models,
                "ensemble_method": str(ensemble_method).lower().strip(),
                "weights_used": ensemble.weights_used,
                "per_model_pv_totals_kwh": ensemble.per_model_pv_totals_kwh,
                "pv_totals_kwh": {
                    "p10": float(ensemble.pv_ensemble_p10.sum()) if ensemble.pv_ensemble_p10 is not None else None,
                    "p50": float(ensemble.pv_ensemble_p50.sum()),
                    "p90": float(ensemble.pv_ensemble_p90.sum()) if ensemble.pv_ensemble_p90 is not None else None,
                }
                if pv_uncertainty
                else None,
                "missing_vars_by_model": ensemble.missing_vars_by_model,
                "derived_irradiance_by_model": ensemble.derived_irradiance_by_model,
                "failed_models": ensemble.failed_models,
                "failed_model_reasons": ensemble.failed_model_reasons,
                "fast_mode": bool(fast_mode),
            },
        }
        insert_forecast_run(str(SQLITE_PATH), payload)
        self.latest_result = payload
        self.history.append(_to_history_summary(payload))
        self.history = self.history[-MAX_HISTORY:]
        self.settings["last_successful_for_target_date"] = target_date.isoformat()
        self._save_results()
        self._save_settings()
        del detail_df, flows_df, pv, weather
        gc.collect()
        return payload

    def run_now(self, payload: RunNowPayload) -> dict:
        if not self._lock.acquire(blocking=False):
            raise HTTPException(status_code=423, detail="Run already in progress")
        try:
            source = self.last_inputs
            soc = float(payload.soc_at_22_percent if payload.soc_at_22_percent is not None else source.get("soc_at_22_percent", 45.0))
            ykwh = float(
                payload.yesterday_consumption_kwh
                if payload.yesterday_consumption_kwh is not None
                else source.get("yesterday_consumption_kwh", 18.0)
            )
            cap = float(payload.user_max_ac_kw if payload.user_max_ac_kw is not None else self.settings["max_ac_charge_power_kw_default"])
            local_today = dt.datetime.now(self._tzinfo()).date()
            target_date = local_today + dt.timedelta(days=1)
            result = self._run(
                target_date,
                soc,
                ykwh,
                float(payload.buffer_percent),
                cap,
                payload.weather_models,
                payload.ensemble_method,
                payload.pv_uncertainty,
                payload.fast_mode,
            )
            result["run_type"] = "manual"
            self.latest_result = result
            if self.history:
                self.history[-1]["run_type"] = "manual"
            self._save_results()
            return {"ran": True, "result": result}
        finally:
            self._lock.release()

    def run_nightly_tick(self, payload: NightlyTickPayload) -> dict:
        if not self._lock.acquire(blocking=False):
            raise HTTPException(status_code=423, detail="Run already in progress")
        try:
            local_now = dt.datetime.now(self._tzinfo())
            target_date = local_now.date() + dt.timedelta(days=1)
            trigger_time = dt.datetime.strptime(self.settings.get("nightly_run_time", DEFAULT_NIGHTLY_TIME), "%H:%M").time()
            if not payload.force and local_now.time() < trigger_time:
                return {"ran": False, "reason": "before_window", "target_date": target_date.isoformat(), "local_now": local_now.isoformat()}
            if not payload.force and self.settings.get("last_successful_for_target_date") == target_date.isoformat():
                return {"ran": False, "reason": "already_ran", "target_date": target_date.isoformat()}

            warnings: list[str] = []
            soc = float(self.last_inputs.get("soc_at_22_percent", 45.0))
            ykwh = float(self.last_inputs.get("yesterday_consumption_kwh", 18.0))
            if not self.last_inputs:
                warnings.append("missing inputs")
            updated_at_raw = self.last_inputs.get("last_inputs_updated_at")
            if updated_at_raw:
                updated_at = dt.datetime.fromisoformat(updated_at_raw)
                if (local_now - updated_at) > dt.timedelta(hours=24):
                    warnings.append("stale inputs")

            result = self._run(
                target_date,
                soc,
                ykwh,
                0.0,
                float(self.settings["max_ac_charge_power_kw_default"]),
                DEFAULT_ACCURACY_MODELS,
                "weighted",
                False,
            )
            result["warnings"] = warnings
            result["run_type"] = "nightly"
            self.latest_result = result
            if self.history:
                self.history[-1]["warnings"] = warnings
                self.history[-1]["run_type"] = "nightly"
            self._save_results()
            return {"ran": True, "reason": "ran", "target_date": target_date.isoformat(), "warnings": warnings, "result": result}
        finally:
            self._lock.release()


app = FastAPI(title="PV Battery Planner Backend")


@app.exception_handler(core.ExternalServiceError)
def external_service_error_handler(request, exc: core.ExternalServiceError):
    return JSONResponse(
        status_code=502,
        content={
            "error": "external_service_error",
            "service": getattr(exc, "service", None),
            "category": getattr(exc, "category", None),
            "detail": str(exc),
            "hint": getattr(exc, "hint", None),
        },
    )


@app.exception_handler(Exception)
def unhandled_exception_handler(request, exc: Exception):
    payload = {
        "error": "internal_server_error",
        "detail": str(exc),
    }
    if DEBUG:
        payload["traceback"] = traceback.format_exc()
    return JSONResponse(status_code=500, content=payload)


state = BackendState()


def _require_token(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != state.api_token:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


@app.get("/v1/health")
def health(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return {"status": "ok", "time": dt.datetime.now(state._tzinfo()).isoformat()}


@app.get("/v1/settings")
def get_settings(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.settings


@app.put("/v1/settings")
def put_settings(payload: SettingsPayload, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.update_settings(payload)


@app.get("/v1/inputs/last")
def get_last_inputs(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.last_inputs


@app.put("/v1/inputs/last")
def put_last_inputs(payload: InputsPayload, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.update_inputs(payload)


@app.post("/v1/run/now")
def run_now(payload: RunNowPayload, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.run_now(payload)


@app.post("/v1/run/nightly")
def run_nightly(payload: NightlyTickPayload, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.run_nightly_tick(payload)


@app.get("/v1/weather/models")
def weather_models(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return {"items": weather_models_payload()}


@app.get("/v1/results/latest")
def latest_result(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    db_payload = fetch_latest_full_run(str(SQLITE_PATH))
    if db_payload is not None:
        return db_payload
    return state.latest_result


@app.get("/v1/results/history")
def history(days: int = 30, show_all_runs: bool = False, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    limit_days = max(1, days)
    if show_all_runs:
        items = fetch_history_all_runs(str(SQLITE_PATH), limit_days=limit_days)
    else:
        items = fetch_history_latest_per_day(str(SQLITE_PATH), limit_days=limit_days)
    if items:
        return {"items": items}
    return {"items": state.history[-limit_days:]}
