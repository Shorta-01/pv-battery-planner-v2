from __future__ import annotations

import copy
import csv
import datetime as dt
import gc
import json
import os
import time
import secrets
import threading
import tempfile
import traceback
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import planner_core as core
import scoring
from db_sqlite import (
    compute_config_hash,
    fetch_history_all_runs,
    fetch_history_latest_per_day,
    fetch_latest_full_run,
    fetch_full_run_by_id,
    init_db,
    insert_actual_hourly_rows,
    insert_forecast_run,
)
from weather_ensemble import (
    DEFAULT_ACCURACY_MODELS,
    WEATHER_DISPLAY_VARS,
    WEATHER_MODELS,
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
SETTINGS_SOURCE_SETTINGS_JSON = "settings.json"
SETTINGS_SOURCE_CONFIG_DEFAULTS = "config.json(defaults)"
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


def _clamp_score_0_100(value: float) -> int:
    return int(min(100, max(0, round(float(value)))))


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
    forecast_mode: str | None = None
    ensemble_method: str = Field(default="weighted")
    pv_uncertainty: bool = False
    fast_mode: bool = False


class NightlyTickPayload(BaseModel):
    force: bool = False


class ActualsHourlyPayload(BaseModel):
    rows: list[dict]
    source: str = "manual_csv"




def _wmo_severity(code: int) -> int:
    # Higher = more severe / more “weather impact”
    if code in (95, 96, 99):
        return 6  # thunderstorm
    if 61 <= code <= 67 or 80 <= code <= 82:
        return 5  # rain / showers
    if 71 <= code <= 77 or code in (85, 86):
        return 4  # snow
    if 51 <= code <= 57:
        return 3  # drizzle
    if code in (45, 48):
        return 2  # fog
    if code in (3,):
        return 2  # overcast
    if code in (2,):
        return 1  # partly cloudy
    if code in (1, 0):
        return 0  # clear-ish
    return 1


def _best_of_day_weather_code(day_df: pd.DataFrame) -> int | None:
    # Expect columns: 'weather_code' and datetime index or 'time'
    if day_df is None or day_df.empty or "weather_code" not in day_df.columns:
        return None

    # Daytime filter: 08:00–18:00
    df = day_df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour
    elif "time" in df.columns and isinstance(df["time"], pd.Series):
        hours = pd.to_datetime(df["time"]).dt.hour
    else:
        return None

    df = df[(hours >= 8) & (hours <= 18)]
    if df.empty:
        return None

    # Build weighted counts
    counts: dict[int, float] = {}
    for v in df["weather_code"].dropna().tolist():
        try:
            c = int(v)
        except Exception:
            continue
        sev = _wmo_severity(c)
        # Weight scheme: base frequency + severity boost
        w = 1.0 + (sev * 0.35)
        counts[c] = counts.get(c, 0.0) + w

    if not counts:
        return None

    # Choose max weighted, tie-break by highest severity
    best_score = max(counts.values())
    candidates = [c for c, score in counts.items() if score == best_score]
    if len(candidates) == 1:
        return candidates[0]

    candidates.sort(key=lambda c: _wmo_severity(c), reverse=True)
    return candidates[0]


def _model_max_days(model_id: str) -> int:
    try:
        return int((WEATHER_MODELS.get(model_id) or {}).get("max_days") or 0)
    except Exception:
        return 0


def auto_select_models_for_location(loc: object, requested_days: int) -> list[str]:
    models_all = list(WEATHER_MODELS.keys())
    if requested_days >= 7:
        return models_all

    preferred: list[str] = []
    for mid in [
        "ecmwf_ifs",
        "dwd_icon_eu",
        "knmi_harmonie_arome",
        "dwd_icon_d2",
        "meteo_france_seamless",
        "meteofrance_seamless",
    ]:
        if mid in WEATHER_MODELS and mid in models_all and mid not in preferred:
            preferred.append(mid)
    if preferred:
        return preferred[:4]
    return models_all[:3] if len(models_all) >= 3 else models_all


def _best_of_day_from_model(
    fr: object,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    tz: str,
) -> int | None:
    df = getattr(fr, "df", None)
    if not isinstance(df, pd.DataFrame) or "weather_code" not in df.columns:
        return None

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize(tz)
        else:
            df = df.tz_convert(tz)

    day_df = df.loc[(df.index >= day_start) & (df.index < day_end), ["weather_code"]]
    return _best_of_day_weather_code(day_df)


def _pick_week_ahead_weather_code(
    day_offset: int,
    *,
    target_date: dt.date,
    tz: str,
    weather_by_model: dict[str, object],
    weights_used: dict[str, float] | None,
    primary_id: str | None,
) -> tuple[int | None, str | None, int | None]:
    """
    Returns (best_code, source_model_id, source_model_max_days)

    Policy:
      - choose model with highest non-NaN weather_code coverage for the day
      - tie-break: higher ensemble weight, then primary model
      - representative code: 12:00 local if available, otherwise mode of day
    """
    day = target_date + dt.timedelta(days=day_offset)
    day_start = pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz)
    day_end = day_start + pd.Timedelta(days=1)

    candidates: list[tuple[int, float, int, str, int, int]] = []
    # (-coverage, -weight, primary_penalty, model_id, code, max_days)

    for model_id, fr in (weather_by_model or {}).items():
        df = getattr(fr, "df", None)
        if not isinstance(df, pd.DataFrame) or "weather_code" not in df.columns:
            continue
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            if idx.tz is None:
                df = df.copy()
                df.index = df.index.tz_localize(tz)
            else:
                df = df.tz_convert(tz)
        day_df = df.loc[(df.index >= day_start) & (df.index < day_end), ["weather_code"]].copy()
        if day_df.empty:
            continue
        wc = pd.to_numeric(day_df["weather_code"], errors="coerce")
        coverage = int(wc.notna().sum())
        if coverage <= 0:
            continue
        midday = day_start + pd.Timedelta(hours=12)
        midday_code = pd.to_numeric(pd.Series([day_df["weather_code"].get(midday)]), errors="coerce").dropna()
        if not midday_code.empty:
            code = int(midday_code.iloc[0])
        else:
            mode_vals = wc.dropna().mode()
            if mode_vals.empty:
                continue
            code = int(mode_vals.iloc[0])
        max_days = _model_max_days(model_id)
        w = float((weights_used or {}).get(model_id, 0.0))
        primary_penalty = 0 if (primary_id and model_id == primary_id) else 1

        candidates.append((-coverage, -w, primary_penalty, model_id, int(code), int(max_days)))

    if not candidates:
        return None, None, None

    candidates.sort()
    _, _, _, best_model_id, best_code, best_max_days = candidates[0]
    return best_code, best_model_id, best_max_days


def _build_pv_week_ahead(
    *,
    target_date: dt.date,
    tz: str,
    pv_totals_p50: list[float | None],
    pv_totals_p10: list[float | None] | None,
    pv_totals_p90: list[float | None] | None,
    weather_by_model: dict[str, object] | None,
    weights_used: dict[str, float] | None,
    weather_primary_model_id: str | None,
) -> list[dict[str, object]]:
    days = min(7, len(pv_totals_p50))
    out: list[dict[str, object]] = []

    for i in range(days):
        day = target_date + dt.timedelta(days=i)

        code, source_model_id, source_max_days = _pick_week_ahead_weather_code(
            i,
            target_date=target_date,
            tz=tz,
            weather_by_model=weather_by_model or {},
            weights_used=weights_used,
            primary_id=weather_primary_model_id,
        )

        out.append(
            {
                "date": day.isoformat(),
                "p50_kwh": float(pv_totals_p50[i]) if pv_totals_p50[i] is not None else None,
                "p10_kwh": float(pv_totals_p10[i]) if pv_totals_p10 and i < len(pv_totals_p10) and pv_totals_p10[i] is not None else None,
                "p90_kwh": float(pv_totals_p90[i]) if pv_totals_p90 and i < len(pv_totals_p90) and pv_totals_p90[i] is not None else None,
                "weather_code": int(code) if code is not None else None,
                "weather_best_of_day": code is not None,
                "weather_code_source_model_id": source_model_id,
                "weather_code_source_model_label": (
                    (WEATHER_MODELS.get(source_model_id) or {}).get("label") if source_model_id else None
                ),
                "weather_code_source_max_days": int(source_max_days) if source_max_days is not None else None,
                "icon_key": "unknown" if code is None else None,
            }
        )

    return out


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
                merged = core.set_user_config(loaded["config"])
                loaded["config"] = merged
                loaded.setdefault("nightly_run_time", DEFAULT_NIGHTLY_TIME)
                loaded.setdefault("timezone", str(merged.get("location", {}).get("timezone") or "Europe/Brussels"))
                loaded.setdefault("max_ac_charge_power_kw_default", DEFAULT_MAX_AC_CAP)
                loaded.setdefault("settings_source", SETTINGS_SOURCE_SETTINGS_JSON)
                return loaded

        settings = self._build_settings_from_repo_defaults()
        self._write_json(SETTINGS_PATH, settings)
        return settings

    def _build_settings_from_repo_defaults(self) -> dict:
        cfg = core.DEFAULT_CONFIG
        if core.CONFIG_PATH.exists():
            cfg = core.load_config_file(core.CONFIG_PATH)
        merged = core.set_user_config(cfg)
        return {
            "config": merged,
            "nightly_run_time": DEFAULT_NIGHTLY_TIME,
            "timezone": str(merged.get("location", {}).get("timezone") or "Europe/Brussels"),
            "max_ac_charge_power_kw_default": DEFAULT_MAX_AC_CAP,
            "settings_source": SETTINGS_SOURCE_CONFIG_DEFAULTS,
        }

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
                "settings_source": SETTINGS_SOURCE_SETTINGS_JSON,
            }
        )
        self._save_settings()
        return self.settings

    def reset_settings_to_repo_defaults(self) -> dict:
        self.settings = self._build_settings_from_repo_defaults()
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
        forecast_mode: str | None,
        ensemble_method: str,
        pv_uncertainty: bool,
        fast_mode: bool = False,
    ) -> dict:
        run_started = time.perf_counter()
        cfg = self.settings["config"]
        loc_cfg = cfg.get("location", {})
        tz = str(loc_cfg.get("timezone", "Europe/Brussels"))
        loc = core.Location(
            name=str(loc_cfg.get("address_query") or loc_cfg.get("name") or "Configured"),
            latitude=float(loc_cfg["latitude"]),
            longitude=float(loc_cfg["longitude"]),
        )

        mode = str(forecast_mode or "auto").lower().strip()
        if mode not in ("auto", "expert"):
            mode = "auto"

        if mode == "expert":
            tomorrow_models = list(weather_models or [])
            if not tomorrow_models:
                tomorrow_models = auto_select_models_for_location(loc, requested_days=1)
        else:
            tomorrow_models = auto_select_models_for_location(loc, requested_days=1)
        week_models = auto_select_models_for_location(loc, requested_days=7)
        if not tomorrow_models:
            raise HTTPException(status_code=400, detail="Select at least one weather model.")

        normalized_ensemble_method = str(ensemble_method).lower().strip()
        weather_cfg = cfg.get("weather", {}) if isinstance(cfg, dict) else {}
        store_provider_payloads = bool(weather_cfg.get("store_provider_payloads", False)) if isinstance(weather_cfg, dict) else False

        run_at_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        run_id = str(uuid.uuid4())
        config_hash = compute_config_hash(cfg)
        inputs_used = {
            "soc_at_22_percent": float(soc_percent),
            "yesterday_consumption_kwh": float(yesterday_kwh),
            "buffer_percent": float(buffer_percent),
            "max_ac_charge_power_kw": float(user_max_ac_kw),
            "weather_models_selected": tomorrow_models,
            "forecast_mode": mode,
            "ensemble_method": normalized_ensemble_method,
            "pv_uncertainty_enabled": bool(pv_uncertainty),
            "fast_mode": bool(fast_mode),
        }

        try:
            ensemble_tomorrow = build_ensemble_forecast(
                loc=loc,
                target_date=target_date,
                tz=tz,
                weather_models=tomorrow_models,
                ensemble_method=normalized_ensemble_method,
                pv_uncertainty=bool(pv_uncertainty),
                accuracy_mode=True,
                fast_mode=bool(fast_mode),
                requested_days=1,
            )
        except RuntimeError as exc:
            if "All weather model requests failed" not in str(exc):
                raise
            failed_models = list(getattr(exc, "failed_models", tomorrow_models) or tomorrow_models)
            failed_reasons = getattr(exc, "failed_model_reasons", {})
            if not isinstance(failed_reasons, dict):
                failed_reasons = {}
            warnings: list[str] = ["all weather model requests failed"]
            for model_id in failed_models:
                reason = failed_reasons.get(model_id)
                if isinstance(reason, dict):
                    reason_msg = str(reason.get("message") or reason.get("category") or "unknown")
                else:
                    reason_msg = "unknown"
                warnings.append(f"model failed: {model_id} ({reason_msg})")
            warnings = list(dict.fromkeys(warnings))
            run_duration_ms = int((time.perf_counter() - run_started) * 1000)
            error_payload = {
                "run_id": run_id,
                "target_date": target_date.isoformat(),
                "run_at": dt.datetime.now(self._tzinfo()).isoformat(),
                "run_at_utc": run_at_utc,
                "run_type": "manual",
                "timezone": tz,
                "status": "error",
                "run_duration_ms": run_duration_ms,
                "warnings": warnings,
                "warnings_count": len(warnings),
                "inputs_used": inputs_used,
                "planner_version": "v2",
                "config_hash": config_hash,
                "config_json": json.dumps(
                    {
                        **cfg,
                        "weather_models_selected": tomorrow_models,
                        "forecast_mode": mode,
                        "ensemble_method": normalized_ensemble_method,
                        "pv_uncertainty_enabled": bool(pv_uncertainty),
                    },
                    sort_keys=True,
                ),
                "config": cfg,
                "created_at_utc": run_at_utc,
                "metrics": {
                    "pv_forecast_kwh": 0.0,
                    "cons_forecast_kwh": float(yesterday_kwh),
                },
                "weather_ensemble": {
                    "selected_models": tomorrow_models,
                    "ensemble_method": normalized_ensemble_method,
                    "weights_used": getattr(exc, "weights_used", None),
                    "primary_model_id": None,
                    "failed_models": failed_models,
                    "failure_reasons_by_model": failed_reasons,
                    "failed_model_reasons": failed_reasons,
                    "model_live_failed_used_cached": {},
                    "fast_mode": bool(fast_mode),
                },
                "forecast_mode_effective": mode,
                "tomorrow_models_used": list(tomorrow_models),
                "week_ahead_models_considered": list(week_models),
                "weather_by_model": {},
            }
            insert_forecast_run(str(SQLITE_PATH), error_payload)
            self.latest_result = error_payload
            self.history.append(_to_history_summary(error_payload))
            self.history = self.history[-MAX_HISTORY:]
            self._save_results()
            return error_payload

        warnings: list[str] = []
        try:
            ensemble_week = build_ensemble_forecast(
                loc=loc,
                target_date=target_date,
                tz=tz,
                weather_models=week_models,
                ensemble_method=normalized_ensemble_method,
                pv_uncertainty=bool(pv_uncertainty),
                accuracy_mode=True,
                fast_mode=False,
                requested_days=7,
            )
        except Exception as exc:
            ensemble_week = None
            warnings.append(f"pv_week_ahead_ensemble_failed={type(exc).__name__}:{exc}")

        important_weather_vars = set(WEATHER_DISPLAY_VARS)
        for model_id in ensemble_tomorrow.failed_models:
            reason = ensemble_tomorrow.failed_model_reasons.get(model_id) if isinstance(ensemble_tomorrow.failed_model_reasons, dict) else None
            reason_msg = str(reason.get("message") or reason.get("category") or "unknown") if isinstance(reason, dict) else "unknown"
            warnings.append(f"model failed: {model_id} ({reason_msg})")

        derived_hours_by_model = getattr(ensemble_tomorrow, "derived_irradiance_hours_by_model", {})
        for model_id, used_derived in ensemble_tomorrow.derived_irradiance_by_model.items():
            derived_hours = int(derived_hours_by_model.get(model_id, 0)) if isinstance(derived_hours_by_model, dict) else 0
            if used_derived and derived_hours > 0:
                warnings.append(f"derived irradiance used: {model_id}")

        for model_id, used_cached in getattr(ensemble_tomorrow, "model_live_failed_used_cached", {}).items():
            if used_cached:
                warnings.append(f"model_live_failed_used_cached=true: {model_id}")

        for model_id, missing_vars in ensemble_tomorrow.missing_vars_by_model.items():
            if not missing_vars:
                continue
            missing_important = sorted(var for var in set(missing_vars) if var in important_weather_vars)
            if missing_important:
                warnings.append(f"important vars missing: {model_id} ({', '.join(missing_important)})")

        def _daily_totals_nullable(series: pd.Series | None, days: int = 7) -> list[float | None]:
            if series is None or len(series) == 0:
                return [None] * days
            s = pd.to_numeric(series, errors="coerce")
            if not isinstance(s.index, pd.DatetimeIndex):
                return [None] * days
            if s.index.tz is None:
                s = s.copy()
                s.index = s.index.tz_localize(tz)
            else:
                s = s.tz_convert(tz)
            daily = s.resample("D").sum(min_count=1)
            out: list[float | None] = []
            for i in range(days):
                if i >= len(daily):
                    out.append(None)
                else:
                    v = daily.iloc[i]
                    out.append(None if pd.isna(v) else float(v))
            return out

        if ensemble_week is not None:
            pv_totals_p50 = _daily_totals_nullable(ensemble_week.pv_ensemble_p50)
            pv_totals_p10 = _daily_totals_nullable(ensemble_week.pv_ensemble_p10)
            pv_totals_p90 = _daily_totals_nullable(ensemble_week.pv_ensemble_p90)
            primary_id = getattr(ensemble_week, "weather_primary_model_id", None)
            weather_by_model = getattr(ensemble_week, "weather_by_model", {}) or {}
            weights_used_week = getattr(ensemble_week, "weights_used", None)
        else:
            pv_totals_p50 = [None] * 7
            pv_totals_p10 = [None] * 7
            pv_totals_p90 = [None] * 7
            primary_id = None
            weather_by_model = {}
            weights_used_week = None

        pv_week_ahead = _build_pv_week_ahead(
            target_date=target_date,
            tz=tz,
            pv_totals_p50=pv_totals_p50,
            pv_totals_p10=pv_totals_p10,
            pv_totals_p90=pv_totals_p90,
            weather_by_model=weather_by_model,
            weights_used=weights_used_week,
            weather_primary_model_id=primary_id,
        )[1:7]

        warnings = list(dict.fromkeys(warnings))
        status = "degraded" if warnings else "ok"

        weather = ensemble_tomorrow.weather_primary
        tomorrow_start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=tz)
        tomorrow_end = pd.Timestamp(dt.datetime.combine(target_date + dt.timedelta(days=1), dt.time(0, 0)), tz=tz)
        tomorrow_index = pd.date_range(tomorrow_start, tomorrow_end, freq="h", inclusive="left")

        pv = pd.DataFrame(index=tomorrow_index)
        pv["pv_east_kwh"] = ensemble_tomorrow.pv_ensemble_east_p50.reindex(pv.index).fillna(0.0)
        pv["pv_south_kwh"] = ensemble_tomorrow.pv_ensemble_south_p50.reindex(pv.index).fillna(0.0)
        pv["pv_total_unclipped_kwh"] = ensemble_tomorrow.pv_ensemble_unclipped_p50.reindex(pv.index).fillna(0.0)
        pv["pv_total_kwh"] = ensemble_tomorrow.pv_ensemble_p50.reindex(pv.index).fillna(0.0)
        if ensemble_tomorrow.pv_ensemble_p10 is not None:
            pv["pv_total_low_kwh"] = ensemble_tomorrow.pv_ensemble_p10.reindex(pv.index).fillna(0.0)
        if ensemble_tomorrow.pv_ensemble_p90 is not None:
            pv["pv_total_high_kwh"] = ensemble_tomorrow.pv_ensemble_p90.reindex(pv.index).fillna(0.0)
        pv["pv_clipped_kwh"] = (pv["pv_total_unclipped_kwh"] - pv["pv_total_kwh"]).clip(lower=0.0)
        pv["pv_dc_available_kwh"] = pv["pv_total_unclipped_kwh"]
        pv["pv_ac_limited_kwh"] = pv["pv_total_kwh"]

        total_e = float(pd.to_numeric(pv["pv_east_kwh"], errors="coerce").fillna(0.0).sum())
        total_s = float(pd.to_numeric(pv["pv_south_kwh"], errors="coerce").fillna(0.0).sum())
        split_total = total_e + total_s
        ratio_e, ratio_s = (total_e / split_total, total_s / split_total) if split_total > 0 else (0.5, 0.5)
        pv = core.ensure_pv_columns(pv, split_ratio=(ratio_e, ratio_s))

        cons_profile = core.load_consumption_profile_kwh_per_hour()
        cons = core.build_consumption_forecast(cons_profile, yesterday_kwh, target_date, tz)
        detail_df, flows_df, soc_series, charge_kw, cutoff_soc, cutoff_reason = core.run_detailed_plan(
            target_date=target_date,
            weather=weather,
            pv_df=pv,
            consumption_kwh=cons,
            soc_at_22_percent=soc_percent,
            buffer_percent=buffer_percent,
            max_ac_charge_power_kw=user_max_ac_kw,
        )
        charge_note = f"{cutoff_reason}."
        grid_import = float(flows_df.get("grid_import_kwh", pd.Series(dtype=float)).sum())
        grid_export = float(flows_df.get("grid_export_kwh", pd.Series(dtype=float)).sum())
        pv_forecast_kwh = float(pv.get("pv_total_kwh", pd.Series(dtype=float)).sum())
        cons_forecast_kwh = float(cons.sum())
        pv_quality = scoring.compute_pv_quality_score(
            pv_df=pv,
            weather_df=weather.df,
            target_date=target_date,
            tz=tz,
            fallback_score=55,
        )

        pv_totals_kwh = {
            "p50": float(ensemble_tomorrow.pv_ensemble_p50.sum()) if ensemble_tomorrow.pv_ensemble_p50 is not None else None,
            "p10": float(ensemble_tomorrow.pv_ensemble_p10.sum()) if ensemble_tomorrow.pv_ensemble_p10 is not None else None,
            "p90": float(ensemble_tomorrow.pv_ensemble_p90.sum()) if ensemble_tomorrow.pv_ensemble_p90 is not None else None,
        }
        pv_tomorrow_low_high_kwh = dict(ensemble_tomorrow.pv_tomorrow_low_high_kwh) if isinstance(ensemble_tomorrow.pv_tomorrow_low_high_kwh, dict) else {"low": None, "high": None, "valid_models": 0}

        run_duration_ms = int((time.perf_counter() - run_started) * 1000)
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
        payload = {
            "run_id": run_id,
            "target_date": target_date.isoformat(),
            "weather": self._serialize_df(weather.df),
            "weather_primary_model_id": ensemble_tomorrow.weather_primary_model_id,
            "weather_ensemble_table": self._serialize_df(ensemble_tomorrow.weather_ensemble_table.df),
            "weather_by_model": {model_id: self._serialize_df(fr.df) for model_id, fr in ensemble_tomorrow.weather_by_model.items()},
            "pv_by_model": {model_id: self._serialize_df(model_pv) for model_id, model_pv in ensemble_tomorrow.pv_by_model.items()},
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
            "warnings": warnings,
            "warnings_count": len(warnings),
            "status": status,
            "run_duration_ms": run_duration_ms,
            "run_at": dt.datetime.now(self._tzinfo()).isoformat(),
            "run_at_utc": run_at_utc,
            "run_type": "manual",
            "timezone": tz,
            "inputs_used": inputs_used,
            "pv_totals_kwh": pv_totals_kwh,
            "pv_tomorrow_low_high_kwh": pv_tomorrow_low_high_kwh,
            "pv_week_ahead": pv_week_ahead,
            "forecast_mode_effective": mode,
            "tomorrow_models_used": list(tomorrow_models),
            "week_ahead_models_considered": list(week_models),
            "system_snapshot": {k: v for k, v in system_snapshot.items() if v is not None},
            "planner_version": "v2",
            "config_hash": config_hash,
            "config_json": json.dumps(
                {
                    **cfg,
                    "weather_models_selected": ensemble_tomorrow.selected_models,
                    "forecast_mode": mode,
                    "ensemble_method": normalized_ensemble_method,
                    "pv_uncertainty_enabled": bool(pv_uncertainty),
                    "per_model_pv_totals_kwh": ensemble_tomorrow.per_model_pv_totals_kwh,
                    "pv_totals_kwh": pv_totals_kwh,
                },
                sort_keys=True,
            ),
            "config": cfg,
            "created_at_utc": run_at_utc,
            "weather_ensemble": {
                "selected_models": ensemble_tomorrow.selected_models,
                "ensemble_method": normalized_ensemble_method,
                "weights_used": ensemble_tomorrow.weights_used,
                "primary_model_id": ensemble_tomorrow.weather_primary_model_id,
                "per_model_pv_totals_kwh": ensemble_tomorrow.per_model_pv_totals_kwh,
                "pv_totals_kwh": pv_totals_kwh if pv_uncertainty else None,
                "pv_tomorrow_low_high_kwh": pv_tomorrow_low_high_kwh,
                "pv_week_ahead": pv_week_ahead,
                "missing_vars_by_model": ensemble_tomorrow.missing_vars_by_model,
                "derived_irradiance_by_model": ensemble_tomorrow.derived_irradiance_by_model,
                "derived_irradiance_hours_by_model": getattr(ensemble_tomorrow, "derived_irradiance_hours_by_model", {}),
                "fetch_meta_by_model": getattr(ensemble_tomorrow, "fetch_meta_by_model", {}),
                "failed_models": ensemble_tomorrow.failed_models,
                "failed_model_reasons": ensemble_tomorrow.failed_model_reasons,
                "model_live_failed_used_cached": getattr(ensemble_tomorrow, "model_live_failed_used_cached", {}),
                "fast_mode": bool(fast_mode),
            },
            "provider_payloads_by_model": (ensemble_tomorrow.provider_payloads_by_model if store_provider_payloads else {}),
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
                payload.forecast_mode,
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
                "auto",
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


def _parse_actual_rows_csv_text(csv_text: str) -> list[dict]:
    reader = csv.DictReader(csv_text.splitlines())
    expected = ["ts_local", "pv_kwh", "load_kwh", "grid_import_kwh", "grid_export_kwh", "soc_pct"]
    if reader.fieldnames != expected:
        raise HTTPException(status_code=400, detail=f"CSV headers must be exactly: {','.join(expected)}")

    rows: list[dict] = []
    for row in reader:
        rows.append(dict(row))
    return rows


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


@app.post("/v1/settings/reset_to_repo_defaults")
def reset_settings_to_repo_defaults(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.reset_settings_to_repo_defaults()


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


@app.post("/v1/actuals/hourly")
async def ingest_actuals_hourly(request: Request, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    content_type = (request.headers.get("content-type") or "").lower()

    source = "manual_csv"
    rows: list[dict]

    if "text/csv" in content_type:
        csv_text = (await request.body()).decode("utf-8")
        rows = _parse_actual_rows_csv_text(csv_text)
    else:
        payload = await request.json()
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            source = str(payload.get("source") or source)
            payload_rows = payload.get("rows")
            if not isinstance(payload_rows, list):
                raise HTTPException(status_code=400, detail="JSON payload must include rows[]")
            rows = payload_rows
        else:
            raise HTTPException(status_code=400, detail="Unsupported payload")

    try:
        inserted = insert_actual_hourly_rows(str(SQLITE_PATH), rows, source=source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"inserted": inserted, "source": source}


@app.post("/v1/score/day")
def score_day(date: str, source: str = "manual_csv", authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    try:
        result = scoring.score_day(str(SQLITE_PATH), date, source=source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "run_id": result["run_id"],
        "score_date": result["score_date"],
        "source": result["source"],
        "pv_mae_kwh": result["pv_mae_kwh"],
        "pv_rmse_kwh": result["pv_rmse_kwh"],
        "pv_bias_kwh": result["pv_bias_kwh"],
        "pv_daily_forecast_kwh": result["pv_daily_forecast_kwh"],
        "pv_daily_actual_kwh": result["pv_daily_actual_kwh"],
        "pv_daily_error_kwh": result["pv_daily_error_kwh"],
        "pv_hourly_points": result["pv_hourly_points"],
        "models_scored": sorted(result.get("model_scores", {}).keys()),
    }


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



@app.get("/v1/results/run/{run_id}")
def result_by_run_id(run_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    payload = fetch_full_run_by_id(str(SQLITE_PATH), run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return payload


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
