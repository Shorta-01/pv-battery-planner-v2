from __future__ import annotations

import copy
import csv
import datetime as dt
import gc
import json
import inspect
import logging
import os
import time
import secrets
import threading
import tempfile
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import planner_core as core
import ocpp_evse
import scoring
from bmw_service import BmwService
from ev_status import build_unified_ev_status

from error_logging import compute_dedupe_key, format_exception_body
from db_sqlite import (
    compute_config_hash,
    delete_all_error_events,
    delete_error_event,
    fetch_error_event_by_id,
    fetch_error_events,
    fetch_daily_score_for_date,
    fetch_effective_daily_kwh,
    fetch_history_all_runs,
    fetch_history_latest_per_day,
    fetch_latest_full_run,
    fetch_full_run_by_id,
    init_db,
    insert_actual_hourly_rows,
    insert_error_event,
    insert_forecast_run,
    set_error_fixed,
)
from weather_ensemble import (
    DEFAULT_ACCURACY_MODELS,
    WEATHER_DISPLAY_VARS,
    WEATHER_MODELS,
    auto_select_models_for_location,
    build_ensemble_forecast,
    should_use_satellite_nowcast_auto,
    get_model_caps,
    weather_models_payload,
    select_week_ahead_models,
    validate_pvlib_runtime,
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
logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _json_payload_size_bytes(payload: object) -> int:
    try:
        raw = json.dumps(payload, separators=(",", ":"), default=str)
    except Exception:
        raw = json.dumps({"unserializable_payload": True}, separators=(",", ":"))
    return len(raw.encode("utf-8"))


class _RunDiagnostics:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.stage_timings_ms: dict[str, float] = {}
        self.payload_sizes_bytes: dict[str, int] = {}
        self.data: dict[str, object] = {
            "timestamp_utc": _utc_now_iso(),
            "success": False,
            "status": "running",
            "error_summary": None,
            "stage_timings_ms": self.stage_timings_ms,
            "payload_sizes_bytes": self.payload_sizes_bytes,
            "total_run_ms": 0.0,
        }

    @contextmanager
    def stage(self, name: str):
        stage_started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - stage_started) * 1000.0
            self.stage_timings_ms[name] = max(0.0, float(elapsed_ms))

    def mark_payload(self, key: str, payload: object) -> int:
        size_bytes = _json_payload_size_bytes(payload)
        self.payload_sizes_bytes[key] = int(size_bytes)
        return int(size_bytes)

    def finalize(self, *, success: bool, status: str, error_summary: str | None = None) -> dict[str, object]:
        total_run_ms = (time.perf_counter() - self.started_at) * 1000.0
        self.stage_timings_ms["total_run"] = max(0.0, float(total_run_ms))
        self.data["total_run_ms"] = max(0.0, float(total_run_ms))
        self.data["success"] = bool(success)
        self.data["status"] = str(status)
        self.data["error_summary"] = error_summary
        return self.data


def _clamp_score_0_100(value: float) -> int:
    return int(min(100, max(0, round(float(value)))))


def _float_or_default(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_float(value, default, *, field_name: str, warnings: list[str] | None = None) -> float:
    def _warn(reason: str) -> None:
        if warnings is not None:
            warnings.append(f"{field_name}: {reason} -> default {default}")

    if value is None:
        _warn("missing")
        return float(default)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"null", "none"}:
            _warn("empty/null string")
            return float(default)
        value = stripped

    if pd.isna(value):
        _warn("NaN")
        return float(default)

    try:
        return float(value)
    except (TypeError, ValueError):
        _warn("invalid numeric value")
        return float(default)


def _valid_hhmm(value: str) -> bool:
    try:
        dt.datetime.strptime(value, "%H:%M")
        return True
    except Exception:
        return False


def _parse_elevation_m(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("elevation")
    if isinstance(raw, list):
        if not raw:
            return None
        raw = raw[0]
    try:
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fetch_elevation_m(lat: float, lon: float) -> float | None:
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": float(lat), "longitude": float(lon)},
            timeout=8,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    try:
        return _parse_elevation_m(resp.json())
    except ValueError:
        return None


def pick_decision_quantile(soc_offpeak_confidence: str, uncertainty_enabled: bool = True) -> tuple[str, str]:
    _ = str(soc_offpeak_confidence or "").strip()
    if not bool(uncertainty_enabled):
        return "p50", "uncertainty_disabled"
    return "p25", "fixed"


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
        "baseline_cost_eur_cycle": pv_quality.get("baseline_cost_eur_cycle"),
        "plan_cost_eur_cycle_cash": pv_quality.get("plan_cost_eur_cycle_cash"),
        "terminal_battery_value_eur_cycle": pv_quality.get("terminal_battery_value_eur_cycle"),
        "plan_cost_eur_cycle": pv_quality.get("plan_cost_eur_cycle"),
        "savings_eur_cycle": pv_quality.get("savings_eur_cycle"),
        "grid_only_cost_eur_cycle": pv_quality.get("grid_only_cost_eur_cycle"),
        "isystem_cost_eur_cycle_cash": pv_quality.get("isystem_cost_eur_cycle_cash"),
        "isystem_cost_eur_cycle": pv_quality.get("isystem_cost_eur_cycle"),
        "benefit_vs_grid_only_eur_cycle": pv_quality.get("benefit_vs_grid_only_eur_cycle"),
        "cycle_start_soc_pct": pv_quality.get("cycle_start_soc_pct"),
        "cycle_end_soc_pct": pv_quality.get("cycle_end_soc_pct"),
        "cycle_soc_delta_pct": pv_quality.get("cycle_soc_delta_pct"),
        "cycle_stored_energy_delta_kwh": pv_quality.get("cycle_stored_energy_delta_kwh"),
        "savings_cycle_terminal_value_applied": pv_quality.get("savings_cycle_terminal_value_applied"),
        "savings_horizon_label": pv_quality.get("savings_horizon_label"),
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
    soc_now_percent: float | None = None
    soc_at_22_percent: float | None = None
    yesterday_consumption_kwh: float = Field(..., gt=0.0)


class RunNowPayload(BaseModel):
    soc_now_percent: float | None = None
    soc_at_22_percent: float | None = None
    yesterday_consumption_kwh: float | None = Field(default=None, gt=0.0)
    buffer_percent: float = Field(default=0.0, ge=0.0, le=10.0)
    user_max_ac_kw: float | None = Field(default=None, ge=0.0)
    weather_models: list[str] | None = None
    forecast_mode: str | None = None
    ensemble_method: str = Field(default="weighted")
    pv_uncertainty: bool = False
    fast_mode: bool = False
    use_satellite_nowcast_0_6h: bool | None = None


class NightlyTickPayload(BaseModel):
    force: bool = False


def _resolve_soc_percent(
    *,
    payload_soc_now: float | None,
    payload_soc_legacy: float | None,
    source: dict,
    warnings: list[str],
) -> float:
    source_soc_now = source.get("soc_now_percent") if isinstance(source, dict) else None
    source_soc_legacy = source.get("soc_at_22_percent") if isinstance(source, dict) else None
    raw_soc = payload_soc_now
    field_name = "soc_now_percent"
    if raw_soc is None:
        raw_soc = payload_soc_legacy
        if raw_soc is not None:
            field_name = "soc_at_22_percent"
    if raw_soc is None:
        raw_soc = source_soc_now
    if raw_soc is None:
        raw_soc = source_soc_legacy
        if raw_soc is not None:
            field_name = "soc_at_22_percent"

    soc = _coerce_float(raw_soc, 45.0, field_name=field_name, warnings=warnings)
    if soc < 0.0 or soc > 100.0:
        clamped = min(100.0, max(0.0, soc))
        warnings.append(f"{field_name}: clamped to {clamped}")
        soc = clamped
    return soc


class ActualsHourlyPayload(BaseModel):
    rows: list[dict]
    source: str = "manual_csv"




class ErrorEventPayload(BaseModel):
    source: str
    severity: str
    error_type: str
    where: str
    title: str
    body: str
    context: dict | None = None


class ErrorFixedPayload(BaseModel):
    fixed: bool


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
    return int(get_model_caps(model_id).get("max_days", 0) or 0)


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


def _best_of_day_from_ensemble_weather_table(
    weather_ensemble_df: pd.DataFrame | None,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    tz: str,
) -> int | None:
    if not isinstance(weather_ensemble_df, pd.DataFrame) or "weather_code" not in weather_ensemble_df.columns:
        return None
    df = weather_ensemble_df
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
    derived_weather_code_by_model: dict[str, bool] | None = None,
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
        derived_penalty = 1 if bool((derived_weather_code_by_model or {}).get(model_id, False)) else 0

        candidates.append((-coverage, derived_penalty, -w, primary_penalty, model_id, int(code), int(max_days)))

    if not candidates:
        return None, None, None

    candidates.sort()
    _, _, _, _, best_model_id, best_code, best_max_days = candidates[0]
    return best_code, best_model_id, best_max_days


def _pick_week_ahead_weather_code_vote(
    day_offset: int,
    *,
    target_date: dt.date,
    tz: str,
    weather_by_model: dict[str, object],
    weights_used: dict[str, float] | None,
    primary_id: str | None,
    derived_weather_code_by_model: dict[str, bool] | None = None,
) -> tuple[int | None, dict[str, object] | None]:
    day = target_date + dt.timedelta(days=day_offset)
    day_start = pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz)
    day_end = day_start + pd.Timedelta(days=1)

    code_votes: dict[int, float] = {}
    code_model_counts: dict[int, int] = {}
    code_derived_counts: dict[int, int] = {}
    vote_breakdown: dict[str, dict[str, float | int | bool]] = {}
    models_considered = 0
    models_used = 0

    for model_id in sorted((weather_by_model or {}).keys()):
        models_considered += 1
        fr = (weather_by_model or {}).get(model_id)
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

        base_weight = float((weights_used or {}).get(model_id, 1.0) or 1.0)
        coverage_factor = max(0.0, min(1.0, coverage / 24.0))
        derived = bool((derived_weather_code_by_model or {}).get(model_id, False))
        derived_penalty_factor = 0.75 if derived else 1.0
        primary_bonus_factor = 1.05 if (primary_id and model_id == primary_id) else 1.0
        final_weight = base_weight * coverage_factor * derived_penalty_factor * primary_bonus_factor
        if final_weight <= 0:
            continue

        models_used += 1
        code_votes[code] = code_votes.get(code, 0.0) + final_weight
        code_model_counts[code] = code_model_counts.get(code, 0) + 1
        if derived:
            code_derived_counts[code] = code_derived_counts.get(code, 0) + 1

        vote_breakdown[model_id] = {
            "code": int(code),
            "vote": float(final_weight),
            "coverage": int(coverage),
            "derived": bool(derived),
        }

    if not code_votes:
        return None, None

    def _code_sort_key(code: int) -> tuple[float, int, float, int, int]:
        total_vote = float(code_votes.get(code, 0.0))
        model_count = int(code_model_counts.get(code, 0))
        derived_count = int(code_derived_counts.get(code, 0))
        derived_ratio = (derived_count / model_count) if model_count > 0 else 1.0
        severity = _wmo_severity(int(code))
        return (-total_vote, -model_count, derived_ratio, -severity, int(code))

    best_code = sorted(code_votes.keys(), key=_code_sort_key)[0]
    diagnostics: dict[str, object] = {
        "selection_policy": "multi_model_weighted_vote",
        "models_considered_count": int(models_considered),
        "models_used_count": int(models_used),
        "top_code_vote": float(code_votes.get(best_code, 0.0)),
        "vote_breakdown": vote_breakdown,
        "code_votes": {str(k): float(v) for k, v in sorted(code_votes.items(), key=lambda kv: int(kv[0]))},
    }
    return int(best_code), diagnostics


def _build_pv_week_ahead(
    *,
    target_date: dt.date,
    tz: str,
    pv_totals_p50: list[float | None] | None = None,
    pv_totals_p10: list[float | None] | None = None,
    pv_totals_p90: list[float | None] | None = None,
    weather_by_model: dict[str, object] | None = None,
    weights_used: dict[str, float] | None = None,
    weather_primary_model_id: str | None = None,
    derived_weather_code_by_model: dict[str, bool] | None = None,
    # Backward-compatible args used by isolated function tests.
    hourly_pv_p50: pd.Series | None = None,
    hourly_pv_p10: pd.Series | None = None,
    hourly_pv_p90: pd.Series | None = None,
    weather_code_series: pd.Series | None = None,
    weather_ensemble_df: pd.DataFrame | None = None,
    week_weather_code_policy: str | None = None,
) -> list[dict[str, object]]:
    if pv_totals_p50 is None:
        def _daily_totals(series: pd.Series | None) -> list[float | None]:
            if series is None or len(series) == 0:
                return [None] * 7
            s = pd.to_numeric(series, errors="coerce")
            if isinstance(s.index, pd.DatetimeIndex):
                if s.index.tz is None:
                    s.index = s.index.tz_localize(tz)
                else:
                    s = s.tz_convert(tz)
            daily = s.resample("D").sum(min_count=1)
            out_vals: list[float | None] = []
            for i in range(7):
                v = daily.iloc[i] if i < len(daily) else None
                out_vals.append(None if v is None or pd.isna(v) else float(v))
            return out_vals
        pv_totals_p50 = _daily_totals(hourly_pv_p50)
        pv_totals_p10 = _daily_totals(hourly_pv_p10)
        pv_totals_p90 = _daily_totals(hourly_pv_p90)

    days = min(7, len(pv_totals_p50))
    out: list[dict[str, object]] = []

    for i in range(days):
        day = target_date + dt.timedelta(days=i)
        code: int | None = None
        source_model_id: str | None = None
        source_max_days: int | None = None
        weather_icon_models_considered: int | None = None
        weather_icon_models_used: int | None = None
        weather_icon_vote_weight: float | None = None
        weather_icon_vote_breakdown: dict[str, object] | None = None

        day_start = pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz)
        day_end = day_start + pd.Timedelta(days=1)
        selection_policy = week_weather_code_policy

        ensemble_picker = globals().get("_best_of_day_from_ensemble_weather_table")
        if callable(ensemble_picker):
            ensemble_code = ensemble_picker(weather_ensemble_df, day_start, day_end, tz)
        else:
            ensemble_code = _best_of_day_weather_code(
                weather_ensemble_df.loc[(weather_ensemble_df.index >= day_start) & (weather_ensemble_df.index < day_end), ["weather_code"]]
            ) if isinstance(weather_ensemble_df, pd.DataFrame) and "weather_code" in weather_ensemble_df.columns and isinstance(weather_ensemble_df.index, pd.DatetimeIndex) else None
        if ensemble_code is not None:
            code = int(ensemble_code)
            source_model_id = "ensemble_weather"
            source_max_days = None
            selection_policy = selection_policy or "ensemble_best_of_day"
        else:
            vote_picker = globals().get("_pick_week_ahead_weather_code_vote")
            if callable(vote_picker):
                vote_code, vote_diag = vote_picker(
                    i,
                    target_date=target_date,
                    tz=tz,
                    weather_by_model=weather_by_model or {},
                    weights_used=weights_used,
                    primary_id=weather_primary_model_id,
                    derived_weather_code_by_model=derived_weather_code_by_model,
                )
                if vote_code is not None:
                    code = int(vote_code)
                    source_model_id = "multi_model_vote"
                    source_max_days = None
                    if isinstance(vote_diag, dict):
                        weather_icon_models_considered = int(vote_diag.get("models_considered_count", 0) or 0)
                        weather_icon_models_used = int(vote_diag.get("models_used_count", 0) or 0)
                        vote_weight_raw = vote_diag.get("top_code_vote")
                        weather_icon_vote_weight = float(vote_weight_raw) if vote_weight_raw is not None else None
                        vb = vote_diag.get("vote_breakdown")
                        weather_icon_vote_breakdown = vb if isinstance(vb, dict) else None
                    selection_policy = selection_policy or "multi_model_weighted_vote"

            picker = globals().get("_pick_week_ahead_weather_code")
            if callable(picker):
                if code is None:
                    code, source_model_id, source_max_days = picker(
                        i,
                        target_date=target_date,
                        tz=tz,
                        weather_by_model=weather_by_model or {},
                        weights_used=weights_used,
                        primary_id=weather_primary_model_id,
                        derived_weather_code_by_model=derived_weather_code_by_model,
                    )
                if code is not None and selection_policy is None:
                    selection_policy = "source_model_best_of_day"

        if code is None and isinstance(weather_code_series, pd.Series):
            s = pd.to_numeric(weather_code_series, errors="coerce")
            if isinstance(s.index, pd.DatetimeIndex):
                if s.index.tz is None:
                    s.index = s.index.tz_localize(tz)
                else:
                    s = s.tz_convert(tz)
                day_start = pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz)
                day_end = day_start + pd.Timedelta(days=1)
                day_series = s[(s.index >= day_start) & (s.index < day_end)]
                noon = day_start + pd.Timedelta(hours=12)
                noon_val = pd.to_numeric(pd.Series([day_series.get(noon)]), errors="coerce").dropna()
                if not noon_val.empty:
                    code = int(noon_val.iloc[0])
                else:
                    mode_vals = day_series.dropna().mode()
                    if not mode_vals.empty:
                        code = int(mode_vals.iloc[0])
                        selection_policy = selection_policy or "legacy_series"

        fallback_used = code is None
        if fallback_used:
            code = 3
            source_model_id = None
            source_max_days = None
            selection_policy = selection_policy or "hard_fallback"

        model_labels = globals().get("WEATHER_MODELS", {}) if isinstance(globals().get("WEATHER_MODELS", {}), dict) else {}
        out.append(
            {
                "date": day.isoformat(),
                "p50_kwh": float(pv_totals_p50[i]) if pv_totals_p50[i] is not None else None,
                "p10_kwh": float(pv_totals_p10[i]) if pv_totals_p10 and i < len(pv_totals_p10) and pv_totals_p10[i] is not None else None,
                "p90_kwh": float(pv_totals_p90[i]) if pv_totals_p90 and i < len(pv_totals_p90) and pv_totals_p90[i] is not None else None,
                "weather_code": int(code),
                "weather_best_of_day": not fallback_used,
                "weather_code_source_model_id": source_model_id,
                "weather_code_source_model_label": (
                    "Ensemble weather"
                    if source_model_id == "ensemble_weather"
                    else ((model_labels.get(source_model_id) or {}).get("label") if source_model_id else None)
                ),
                "weather_code_source_max_days": int(source_max_days) if source_max_days is not None else None,
                "icon_key": None,
                "weather_code_fallback_used": bool(fallback_used),
                "weather_code_selection_policy": selection_policy,
                "weather_icon_models_considered": weather_icon_models_considered,
                "weather_icon_models_used": weather_icon_models_used,
                "weather_icon_vote_weight": weather_icon_vote_weight,
                "weather_icon_vote_breakdown": weather_icon_vote_breakdown,
            }
        )

    return out


class BackendState:
    def __init__(self) -> None:
        LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
        init_db(str(SQLITE_PATH))
        self._lock = threading.Lock()
        self.latest_diagnostics: dict[str, object] = {}
        self.api_token = self._load_or_create_token()
        self.settings = self._load_settings()
        self.last_inputs = self._read_json(INPUTS_PATH, default={})
        self.last_inputs_sanitized_warnings: list[str] = []
        self.settings_sanitized_warnings: list[str] = []
        self._sanitize_last_inputs()
        self._sanitize_settings()
        self.bmw_service = BmwService(self._resolved_ev_runtime_config(core.DEFAULT_CONFIG))
        self.latest_result = self._read_json(LATEST_RESULT_PATH, default={})
        self.history = self._load_history()
        self._apply_config(self.settings["config"])
        self.bmw_service.update_config(self._resolved_ev_runtime_config(self.settings.get("config", {})))
        if DEBUG:
            bmw_debug = self.bmw_service.device_flow_debug_info()
            logger.info(
                "BMW runtime debug: module=%s client_id_masked=%s auth_base=%s start_url=%s poll_url=%s rest_base=%s rest_token_mode=%s stream_enabled=%s rebuilt=%s",
                bmw_debug.get("bmw_auth_module_path"),
                bmw_debug.get("active_client_id_masked"),
                bmw_debug.get("active_auth_base_url"),
                bmw_debug.get("device_flow_start_url"),
                bmw_debug.get("device_flow_poll_url"),
                bmw_debug.get("rest_api_base_url"),
                bmw_debug.get("rest_token_mode"),
                bmw_debug.get("stream_enabled"),
                bmw_debug.get("provider_rebuilt_after_config_update"),
            )
        self._migrate_json_history_to_sqlite()

    def record_endpoint_payload_size(self, endpoint_key: str, payload: object) -> int:
        size_bytes = _json_payload_size_bytes(payload)
        if not isinstance(self.latest_diagnostics, dict):
            self.latest_diagnostics = {
                "timestamp_utc": _utc_now_iso(),
                "status": "n/a",
                "success": False,
                "stage_timings_ms": {},
                "payload_sizes_bytes": {},
                "total_run_ms": 0.0,
            }
        payload_sizes = self.latest_diagnostics.setdefault("payload_sizes_bytes", {})
        if isinstance(payload_sizes, dict):
            payload_sizes[endpoint_key] = int(size_bytes)
        return int(size_bytes)

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
                soc_last = self.last_inputs.get("soc_now_percent", self.last_inputs.get("soc_at_22_percent"))
                payload["inputs_used"] = {
                    "soc_now_percent": soc_last,
                    "soc_at_22_percent": soc_last,
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
                try:
                    merged = core.set_user_config(loaded["config"])
                except ValueError as exc:
                    if "location.elevation_m is required" not in str(exc):
                        raise
                    repaired_cfg = copy.deepcopy(loaded["config"]) if isinstance(loaded.get("config"), dict) else {}
                    loc_cfg = repaired_cfg.get("location")
                    if not isinstance(loc_cfg, dict):
                        loc_cfg = {}
                        repaired_cfg["location"] = loc_cfg
                    resolved_meta = core.resolve_location_metadata(
                        latitude=loc_cfg.get("latitude"),
                        longitude=loc_cfg.get("longitude"),
                        fallback_timezone=loc_cfg.get("timezone"),
                        fallback_elevation_m=loc_cfg.get("elevation_m"),
                        force_refresh=False,
                    )
                    loc_cfg["timezone"] = resolved_meta["timezone"]
                    loc_cfg["elevation_m"] = resolved_meta["elevation_m"]
                    loc_cfg["auto_resolve_metadata"] = True
                    merged = core.set_user_config(repaired_cfg)
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

    def _sanitize_last_inputs(self) -> None:
        changed = False
        if not isinstance(self.last_inputs, dict):
            self.last_inputs = {}
            changed = True

        warnings: list[str] = []

        raw_soc_now = self.last_inputs.get("soc_now_percent")
        raw_soc_legacy = self.last_inputs.get("soc_at_22_percent")
        soc = _resolve_soc_percent(
            payload_soc_now=raw_soc_now,
            payload_soc_legacy=raw_soc_legacy,
            source=self.last_inputs,
            warnings=warnings,
        )
        if raw_soc_now != soc:
            changed = True
        self.last_inputs["soc_now_percent"] = soc
        if "soc_at_22_percent" in self.last_inputs:
            self.last_inputs.pop("soc_at_22_percent", None)
            changed = True

        raw_ykwh = self.last_inputs.get("yesterday_consumption_kwh")
        ykwh = _coerce_float(raw_ykwh, 18.0, field_name="yesterday_consumption_kwh", warnings=warnings)
        if ykwh <= 0:
            warnings.append(f"yesterday_consumption_kwh: must be > 0 -> default 18.0")
            ykwh = 18.0
        if raw_ykwh != ykwh:
            changed = True
        self.last_inputs["yesterday_consumption_kwh"] = ykwh

        self.last_inputs_sanitized_warnings = warnings
        if changed:
            self._save_inputs()

    def _sanitize_settings(self) -> None:
        changed = False
        if not isinstance(self.settings, dict):
            self.settings = self._build_settings_from_repo_defaults()
            changed = True

        warnings: list[str] = []

        raw_cap = self.settings.get("max_ac_charge_power_kw_default")
        cap = _coerce_float(raw_cap, DEFAULT_MAX_AC_CAP, field_name="max_ac_charge_power_kw_default", warnings=warnings)
        if cap <= 0:
            warnings.append(f"max_ac_charge_power_kw_default: must be > 0 -> default {DEFAULT_MAX_AC_CAP}")
            cap = DEFAULT_MAX_AC_CAP
        if raw_cap != cap:
            changed = True
        self.settings["max_ac_charge_power_kw_default"] = cap

        raw_nightly = str(self.settings.get("nightly_run_time") or "").strip()
        if not _valid_hhmm(raw_nightly):
            warnings.append(f"nightly_run_time: invalid -> default {DEFAULT_NIGHTLY_TIME}")
            self.settings["nightly_run_time"] = DEFAULT_NIGHTLY_TIME
            changed = True

        # Do not auto-fetch elevation here. Settings sanitization should be deterministic and
        # free of network side effects during BackendState initialization.

        self.settings_sanitized_warnings = warnings
        if changed:
            self._save_settings()

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
        config_in = copy.deepcopy(config) if isinstance(config, dict) else {}
        loc_cfg = config_in.get("location", {}) if isinstance(config_in.get("location"), dict) else {}
        resolved_meta = core.resolve_location_metadata(
            latitude=loc_cfg.get("latitude"),
            longitude=loc_cfg.get("longitude"),
            fallback_timezone=loc_cfg.get("timezone"),
            fallback_elevation_m=loc_cfg.get("elevation_m"),
            force_refresh=True,
        )
        if isinstance(loc_cfg, dict):
            loc_cfg["timezone"] = resolved_meta["timezone"]
            loc_cfg["elevation_m"] = resolved_meta["elevation_m"]
            loc_cfg["auto_resolve_metadata"] = True
        merged = core.set_user_config(config_in)
        self.settings["config"] = merged
        if resolved_meta.get("warnings"):
            logger.warning("Location metadata auto-derive warnings: %s", "; ".join(str(w) for w in resolved_meta.get("warnings", [])))
        return merged


    def _resolved_ev_runtime_config(self, cfg: dict | None = None) -> dict:
        effective_cfg = cfg if isinstance(cfg, dict) else (self.settings.get("config", {}) if isinstance(self.settings, dict) else {})
        ev_cfg = effective_cfg.get("ev_vehicle_data", {}) if isinstance(effective_cfg.get("ev_vehicle_data"), dict) else {}
        cap_default = _float_or_default(self.settings.get("max_ac_charge_power_kw_default"), DEFAULT_MAX_AC_CAP)
        runtime = {
            "ev_vehicle_data": dict(ev_cfg),
            "charger_max_power_kw": cap_default,
        }
        runtime.update(dict(ev_cfg))
        if runtime.get("charger_max_power_kw") in (None, ""):
            runtime["charger_max_power_kw"] = cap_default
        return runtime

    def get_unified_ev_status(self) -> dict:
        cfg = self.settings.get("config", {}) if isinstance(self.settings, dict) else {}
        ev_cfg = cfg.get("ev_vehicle_data", {}) if isinstance(cfg.get("ev_vehicle_data"), dict) else {}
        provider_status = self.bmw_service.provider_status()
        vehicles = self.bmw_service.vehicles()
        evse = evse_mgr.status_dict()
        evse["enabled"] = bool((cfg.get("car_charger", {}) if isinstance(cfg.get("car_charger"), dict) else {}).get("enabled", False))
        return build_unified_ev_status(
            ev_cfg=ev_cfg,
            runtime_ev_cfg=self._resolved_ev_runtime_config(cfg),
            bmw_provider_status=provider_status,
            bmw_vehicles=vehicles,
            evse_status=evse,
        )

    def update_settings(self, payload: SettingsPayload) -> dict:
        dt.datetime.strptime(payload.nightly_run_time, "%H:%M")
        merged = self._apply_config(payload.config)
        loc_cfg = merged.get("location", {}) if isinstance(merged, dict) else {}
        canonical_tz = str(loc_cfg.get("timezone") or payload.timezone or "Europe/Brussels")
        try:
            _ = ZoneInfo(canonical_tz)
        except Exception:
            canonical_tz = "Europe/Brussels"
        self.bmw_service.update_config(self._resolved_ev_runtime_config(merged))
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

    def reset_settings_to_factory_defaults(self) -> dict:
        preserved = copy.deepcopy(self.settings.get("config", {}).get("tariff", {}).get("offpeak_windows_by_dow"))
        self.settings = self._build_settings_from_repo_defaults()
        if isinstance(preserved, list):
            self.settings["config"].setdefault("tariff", {})["offpeak_windows_by_dow"] = preserved
        self._apply_config(self.settings["config"])
        self.settings["timezone"] = str(
            self.settings["config"].get("location", {}).get("timezone")
            or self.settings.get("timezone")
            or "Europe/Brussels"
        )
        self._save_settings()
        return self.settings

    def update_inputs(self, payload: InputsPayload) -> dict:
        now_local = dt.datetime.now(self._tzinfo()).isoformat()
        soc = _resolve_soc_percent(
            payload_soc_now=payload.soc_now_percent,
            payload_soc_legacy=payload.soc_at_22_percent,
            source=self.last_inputs,
            warnings=[],
        )
        self.last_inputs = {
            "soc_now_percent": float(soc),
            "yesterday_consumption_kwh": float(payload.yesterday_consumption_kwh),
            "last_inputs_updated_at": now_local,
        }
        if payload.soc_now_percent is not None or payload.soc_at_22_percent is not None:
            self.last_inputs["soc_at_22_percent"] = float(soc)
        self._save_inputs()
        return self.last_inputs

    def _serialize_df(self, df: pd.DataFrame) -> dict:
        return json.loads(df.to_json(date_format="iso", orient="split"))

    def _serialize_series(self, s: pd.Series) -> dict:
        frame = s.to_frame(name="value")
        return self._serialize_df(frame)

    def _bmw_planning_freshness_threshold_seconds(self, ev_cfg: dict) -> int:
        for key in ("bmw_healthcheck_seconds",):
            value = ev_cfg.get(key)
            try:
                parsed = int(value)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                continue
        return 300

    def _bmw_vehicle_freshness_seconds(self, vehicle: dict) -> int | None:
        freshness = vehicle.get("freshness_seconds")
        try:
            if freshness is not None:
                return max(0, int(float(freshness)))
        except (TypeError, ValueError):
            pass

        last_update_raw = vehicle.get("last_update_ts")
        if isinstance(last_update_raw, str):
            try:
                parsed = dt.datetime.fromisoformat(last_update_raw.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=dt.timezone.utc)
                return max(0, int((dt.datetime.now(dt.timezone.utc) - parsed.astimezone(dt.timezone.utc)).total_seconds()))
            except ValueError:
                return None
        return None

    def _get_planning_ready_ev_state(self) -> dict:
        ev_cfg = self.settings.get("config", {}).get("ev_vehicle_data", {}) if isinstance(self.settings.get("config", {}), dict) else {}
        enabled = bool(ev_cfg.get("enabled", False))
        threshold_seconds = self._bmw_planning_freshness_threshold_seconds(ev_cfg)
        planning_state = {
            "vehicles": {},
            "refresh_attempted": False,
            "refresh_succeeded": False,
            "refresh_reason": "ev_disabled" if not enabled else "cached_fresh",
            "warning": None,
            "threshold_seconds": threshold_seconds,
            "fallback_to_cached_state": False,
            "fallback_reason": None,
        }

        try:
            vehicles = self.bmw_service.vehicles()
        except Exception:
            vehicles = {}
        planning_state["vehicles"] = vehicles

        if not enabled:
            return planning_state

        refresh_reason = "run_forecast"

        planning_state["refresh_attempted"] = True
        planning_state["refresh_reason"] = refresh_reason
        try:
            refresh_result = self.bmw_service.manual_refresh(force_reprobe=False)
        except Exception as exc:
            refresh_result = {"ok": False, "reason": "poll_failed", "error": str(exc)}
        refresh_ok = bool(refresh_result.get("ok")) if isinstance(refresh_result, dict) else False
        planning_state["refresh_succeeded"] = refresh_ok

        try:
            refreshed_vehicles = self.bmw_service.vehicles()
        except Exception:
            refreshed_vehicles = vehicles
        planning_state["vehicles"] = refreshed_vehicles

        if refresh_ok:
            planning_state["warning"] = "BMW live refresh succeeded before planning."
            planning_state["fallback_to_cached_state"] = False
            planning_state["fallback_reason"] = None
        else:
            reason = refresh_result.get("reason") if isinstance(refresh_result, dict) else "unknown"
            planning_state["fallback_to_cached_state"] = bool(vehicles)
            planning_state["fallback_reason"] = str(reason or "unknown")
            if vehicles:
                first_vehicle = next(iter(vehicles.values())) if isinstance(vehicles, dict) and vehicles else {}
                freshness_seconds = self._bmw_vehicle_freshness_seconds(first_vehicle) if isinstance(first_vehicle, dict) else None
                if freshness_seconds is None:
                    stale_hint = "unknown freshness"
                elif freshness_seconds > threshold_seconds:
                    stale_hint = f"stale (>{threshold_seconds}s)"
                else:
                    stale_hint = f"{freshness_seconds}s old"
                planning_state["warning"] = (
                    f"BMW live refresh failed ({reason}); using last known EV state ({stale_hint})."
                )
            else:
                planning_state["warning"] = (
                    f"BMW live refresh failed ({reason}) and no BMW vehicle data is available."
                )

        if not planning_state["vehicles"] and refresh_ok:
            planning_state["warning"] = "BMW EV integration enabled, but no BMW vehicle data available after live refresh."

        return planning_state

    def _normalize_run_config(
        self,
        *,
        target_date: dt.date,
        soc_percent: float,
        weather_models: list[str] | None,
        forecast_mode: str | None,
        ensemble_method: str,
        use_satellite_nowcast_0_6h_override: bool | None,
    ) -> dict:
        cfg = self.settings["config"]
        loc_cfg = cfg.get("location", {})
        tz = str(loc_cfg.get("timezone", "Europe/Brussels"))
        loc = core.Location(
            name=str(loc_cfg.get("address_query") or loc_cfg.get("name") or "Configured"),
            latitude=float(loc_cfg["latitude"]),
            longitude=float(loc_cfg["longitude"]),
            elevation_m=float(loc_cfg["elevation_m"]) if loc_cfg.get("elevation_m") is not None else None,
        )

        mode = str(forecast_mode or "auto").lower().strip()
        if mode not in ("auto", "expert"):
            mode = "auto"

        if mode == "expert":
            tomorrow_models = list(weather_models or [])
            if not tomorrow_models:
                tomorrow_models = auto_select_models_for_location(loc, requested_days=1)
        else:
            tomorrow_models = auto_select_models_for_location(loc.latitude, loc.longitude, requested_days=1)
        week_models = select_week_ahead_models(requested_days=7, lat=loc.latitude, lon=loc.longitude)
        if not tomorrow_models:
            raise HTTPException(status_code=400, detail="Select at least one weather model.")

        normalized_ensemble_method = str(ensemble_method).lower().strip()
        ensemble_method_tomorrow = "weighted"
        ensemble_method_week = "median"
        weather_cfg = cfg.get("weather", {}) if isinstance(cfg, dict) else {}
        store_provider_payloads = bool(weather_cfg.get("store_provider_payloads", False)) if isinstance(weather_cfg, dict) else False
        requested_use_sat = bool(weather_cfg.get("use_satellite_nowcast_0_6h", False)) if isinstance(weather_cfg, dict) else False
        now_utc = dt.datetime.now(dt.timezone.utc)
        requested_days = max(1, (target_date - now_utc.astimezone(ZoneInfo(tz)).date()).days)
        if mode == "auto":
            if requested_days > 1:
                effective_use_sat = False
            else:
                effective_use_sat = should_use_satellite_nowcast_auto(
                    latitude=loc.latitude,
                    longitude=loc.longitude,
                    timezone_name=tz,
                    requested_days=1,
                    now_utc=now_utc,
                )
        elif use_satellite_nowcast_0_6h_override is not None:
            effective_use_sat = bool(use_satellite_nowcast_0_6h_override)
        else:
            effective_use_sat = requested_use_sat

        ev_state = self._get_planning_ready_ev_state()
        ev_warning = ev_state.get("warning")
        vehicles = ev_state.get("vehicles") if isinstance(ev_state.get("vehicles"), dict) else {}
        fallback_to_cached_state = bool(ev_state.get("fallback_to_cached_state", bool(vehicles)))
        fallback_reason = ev_state.get("fallback_reason")
        if vehicles:
            first = next(iter(vehicles.values()))
            ev_soc = first.get("soc_pct")
            ev_status = str(first.get("data_status") or "")
            if ev_soc is not None and ev_status in {"fresh", "aging", "fallback"}:
                soc_percent = float(ev_soc)

        return {
            "cfg": cfg,
            "loc_cfg": loc_cfg,
            "tz": tz,
            "loc": loc,
            "mode": mode,
            "tomorrow_models": tomorrow_models,
            "week_models": week_models,
            "normalized_ensemble_method": normalized_ensemble_method,
            "ensemble_method_tomorrow": ensemble_method_tomorrow,
            "ensemble_method_week": ensemble_method_week,
            "store_provider_payloads": store_provider_payloads,
            "effective_use_sat": effective_use_sat,
            "ev_warning": ev_warning,
            "fallback_to_cached_state": fallback_to_cached_state,
            "fallback_reason": fallback_reason,
            "soc_percent": soc_percent,
        }

    def _daily_totals_nullable(self, series: pd.Series | None, *, tz: str, days: int = 7) -> list[float | None]:
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

    def _daily_counts_nullable(self, series: pd.Series | None, *, tz: str, days: int = 7) -> list[int | None]:
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
        daily = s.resample("D").median()
        out: list[int | None] = []
        for i in range(days):
            if i >= len(daily) or pd.isna(daily.iloc[i]):
                out.append(None)
            else:
                out.append(int(round(float(daily.iloc[i]))))
        return out

    def _daily_coverage(self, series: pd.Series | None, *, tz: str, days: int = 7) -> list[float | None]:
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
        daily_valid = s.notna().resample("D").sum()
        daily_total = s.resample("D").size()
        out: list[float | None] = []
        for i in range(days):
            if i >= len(daily_total) or daily_total.iloc[i] <= 0:
                out.append(None)
            else:
                out.append(float(daily_valid.iloc[i] / daily_total.iloc[i]))
        return out

    def _build_error_payload_all_models_failed(
        self,
        *,
        run_id: str,
        target_date: dt.date,
        run_at_utc: str,
        tz: str,
        run_duration_ms: int,
        run_diagnostics: dict[str, object],
        warnings: list[str],
        inputs_used: dict[str, object],
        cfg: dict,
        tomorrow_models: list[str],
        mode: str,
        ensemble_method_tomorrow: str,
        pv_uncertainty: bool,
        config_hash: str,
        failed_models: list[str],
        failed_reasons: dict,
        week_models: list[str],
        effective_fast_mode: bool,
        fast_mode: bool,
        yesterday_kwh: float,
        weights_used: object = None,
    ) -> dict:
        return {
            "run_id": run_id,
            "target_date": target_date.isoformat(),
            "run_at": dt.datetime.now(self._tzinfo()).isoformat(),
            "run_at_utc": run_at_utc,
            "run_type": "manual",
            "timezone": tz,
            "status": "error",
            "run_duration_ms": run_duration_ms,
            "run_diagnostics": run_diagnostics,
            "warnings": warnings,
            "warnings_count": len(warnings),
            "inputs_used": inputs_used,
            "planner_version": "v2",
            "data_source": {"soc": "manual", "load": "manual", "pv": "forecast"},
            "config_hash": config_hash,
            "config_json": json.dumps(
                {
                    **cfg,
                    "weather_models_selected": tomorrow_models,
                    "forecast_mode": mode,
                    "ensemble_method": ensemble_method_tomorrow,
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
                "ensemble_method": ensemble_method_tomorrow,
                "weights_used": weights_used,
                "primary_model_id": None,
                "failed_models": failed_models,
                "failure_reasons_by_model": failed_reasons,
                "failed_model_reasons": failed_reasons,
                "model_live_failed_used_cached": {},
                "fast_mode": bool(effective_fast_mode),
                "fast_mode_requested": bool(fast_mode),
                "model_selection_reason": None,
                "dynamic_daylight_method": None,
                "forecast_quality_tier": None,
            },
            "forecast_mode_effective": mode,
            "tomorrow_models_used": [],
            "week_ahead_models_considered": list(week_models),
            "weather_by_model": {},
        }

    def _build_run_metrics_payload(
        self,
        *,
        charge_kw: float,
        cutoff_soc: float,
        cutoff_reason: str,
        charge_note: str,
        charge_target_reachable: bool,
        charge_warning_text: str,
        charge_effective_cap_kw: float,
        charge_limit_reason: str,
        grid_import: float,
        grid_export: float,
        pv_forecast_kwh: float | None,
        cons_forecast_kwh: float,
        tomorrow_coverage_hours: int,
        offpeak_start: dt.datetime,
        offpeak_end: dt.datetime,
        estimated_soc_percent: float,
        hours_until_offpeak_start: float,
        soc_offpeak_confidence: str,
        soc_offpeak_method: str,
        soc_offpeak_debug: dict,
        effective_daily_kwh: float,
        used_history: bool,
        pv_credit_available: bool,
        decision_quantile: str,
        decision_reason: str,
        cutoff_adjusted_for_pv_headroom: bool,
        cutoff_adjustment_reason: str,
        pv_surplus_store_econ_enabled: bool,
        pv_surplus_export_preferred_kwh: float,
        pv_surplus_store_preferred_kwh: float,
        pv_store_vs_export_decisions_count: int,
        allow_injection_to_grid: bool,
        export_blocked_by_policy: bool,
        blocked_export_kwh_total: float,
        max_grid_import_kw: float,
        grid_import_cap_active: bool,
        grid_import_cap_binding_events: int,
        grid_import_cap_limited_charge_kwh_total: float,
        grid_import_cap_load_exceeds_events: int,
        week_models: list[str],
        ensemble_tomorrow,
        week_models_used_count_per_hour: pd.Series | None,
        week_models_count_per_day: list[int | None],
        week_coverage_per_day: list[float | None],
    ) -> dict:
        return {
            "charge_kw": float(charge_kw),
            "cutoff_soc": float(cutoff_soc),
            "cutoff_reason": cutoff_reason,
            "charge_note": charge_note,
            "charge_target_reachable": bool(charge_target_reachable),
            "charge_warning_text": str(charge_warning_text or ""),
            "charge_effective_cap_kw": float(charge_effective_cap_kw),
            "charge_limit_reason": str(charge_limit_reason),
            "grid_import": float(grid_import),
            "grid_export": float(grid_export),
            "pv_forecast_kwh": pv_forecast_kwh,
            "cons_forecast_kwh": cons_forecast_kwh,
            "tomorrow_coverage_hours": tomorrow_coverage_hours,
            "offpeak_start_local": offpeak_start.isoformat(),
            "offpeak_end_local": offpeak_end.isoformat(),
            "soc_offpeak_start_estimated_percent": float(estimated_soc_percent),
            "soc_offpeak_start_hours_until": float(hours_until_offpeak_start),
            "soc_offpeak_start_confidence": str(soc_offpeak_confidence),
            "soc_offpeak_start_method": str(soc_offpeak_method),
            "soc_offpeak_start_load_kwh_window": float(soc_offpeak_debug.get("load_kwh_window", 0.0)),
            "soc_offpeak_start_pv_credit_kwh": float(soc_offpeak_debug.get("pv_credit_kwh", 0.0)),
            "soc_offpeak_start_effective_daily_kwh_used": float(soc_offpeak_debug.get("effective_daily_kwh_used", effective_daily_kwh)),
            "soc_offpeak_start_used_history": bool(soc_offpeak_debug.get("used_history", used_history)),
            "soc_offpeak_start_pv_credit_available": bool(soc_offpeak_debug.get("pv_credit_available", pv_credit_available)),
            "soc_offpeak_start_peak_overlap": bool(soc_offpeak_debug.get("peak_overlap", False)),
            "pv_decision_scenario": decision_quantile,
            "pv_decision_reason": decision_reason,
            "decision_quantile_used": decision_quantile,
            "decision_quantile_policy": decision_reason,
            "confidence_influences_planning": False,
            "cutoff_adjusted_for_pv_headroom": bool(cutoff_adjusted_for_pv_headroom),
            "cutoff_adjustment_reason": str(cutoff_adjustment_reason),
            "pv_surplus_store_econ_enabled": bool(pv_surplus_store_econ_enabled),
            "pv_surplus_export_preferred_kwh": float(pv_surplus_export_preferred_kwh),
            "pv_surplus_store_preferred_kwh": float(pv_surplus_store_preferred_kwh),
            "pv_store_vs_export_decisions_count": int(pv_store_vs_export_decisions_count),
            "allow_injection_to_grid": bool(allow_injection_to_grid),
            "export_blocked_by_policy": bool(export_blocked_by_policy),
            "blocked_export_kwh_total": float(blocked_export_kwh_total),
            "max_grid_import_kw": float(max_grid_import_kw),
            "grid_import_cap_active": bool(grid_import_cap_active),
            "grid_import_cap_binding": bool(grid_import_cap_binding_events > 0),
            "grid_import_cap_limited_charge_kwh": float(grid_import_cap_limited_charge_kwh_total),
            "grid_import_cap_hours_binding": int(grid_import_cap_binding_events),
            "grid_import_cap_load_exceeds_hours": int(grid_import_cap_load_exceeds_events),
            "week_models_used": list(week_models),
            "week_models_count": int(len(week_models)),
            "forecast_quality_tier": getattr(ensemble_tomorrow, "forecast_quality_tier", None),
            "pv_week_models_used_count_per_hour": self._serialize_series(week_models_used_count_per_hour) if isinstance(week_models_used_count_per_hour, pd.Series) else None,
            "pv_week_valid_model_count_per_day": week_models_count_per_day,
            "pv_week_coverage_per_day": week_coverage_per_day,
        }

    def _build_weather_ensemble_payload(
        self,
        *,
        ensemble_tomorrow,
        tomorrow_models: list[str],
        ensemble_method_tomorrow: str,
        pv_uncertainty: bool,
        pv_totals_kwh: dict,
        pv_tomorrow_low_high_kwh: dict,
        pv_model_spread: dict | None,
        pv_week_ahead: list[dict],
        pv_temperature_metadata: dict,
        ensemble_method_week: str,
        effective_fast_mode: bool,
        fast_mode: bool,
        tomorrow_weather_code: int | None,
        tomorrow_source_model_id: str | None,
        tomorrow_source_label: str | None,
        tomorrow_source_max_days: int | None,
    ) -> dict:
        return {
            "selected_models": getattr(ensemble_tomorrow, "selected_models", tomorrow_models),
            "ensemble_method": ensemble_method_tomorrow,
            "weights_used": getattr(ensemble_tomorrow, "weights_used", None),
            "primary_model_id": ensemble_tomorrow.weather_primary_model_id,
            "per_model_pv_totals_kwh": getattr(ensemble_tomorrow, "per_model_pv_totals_kwh", {}),
            "pv_totals_kwh": pv_totals_kwh if pv_uncertainty else None,
            "pv_tomorrow_low_high_kwh": pv_tomorrow_low_high_kwh,
            "pv_tomorrow_model_spread_kwh": pv_model_spread,
            "pv_week_ahead": pv_week_ahead,
            "pv_temperature_metadata": pv_temperature_metadata,
            "ensemble_method_week_ahead": ensemble_method_week,
            "missing_vars_by_model": getattr(ensemble_tomorrow, "missing_vars_by_model", {}),
            "derived_irradiance_by_model": getattr(ensemble_tomorrow, "derived_irradiance_by_model", {}),
            "derived_weather_code_by_model": getattr(ensemble_tomorrow, "derived_weather_code_by_model", {}),
            "derived_irradiance_hours_by_model": getattr(ensemble_tomorrow, "derived_irradiance_hours_by_model", {}),
            "quality_weight_factors_by_model": getattr(ensemble_tomorrow, "quality_weight_factors_by_model", {}),
            "fetch_meta_by_model": getattr(ensemble_tomorrow, "fetch_meta_by_model", {}),
            "failed_models": getattr(ensemble_tomorrow, "failed_models", []),
            "failed_model_reasons": getattr(ensemble_tomorrow, "failed_model_reasons", {}),
            "model_live_failed_used_cached": getattr(ensemble_tomorrow, "model_live_failed_used_cached", {}),
            "deduped_models_dropped": getattr(ensemble_tomorrow, "deduped_models_dropped", None),
            "fast_mode": bool(effective_fast_mode),
            "fast_mode_requested": bool(fast_mode),
            "model_selection_reason": getattr(ensemble_tomorrow, "model_selection_reason", None),
            "dynamic_daylight_method": getattr(ensemble_tomorrow, "dynamic_daylight_method", None),
            "forecast_quality_tier": getattr(ensemble_tomorrow, "forecast_quality_tier", None),
            "tomorrow_weather_code": int(tomorrow_weather_code) if tomorrow_weather_code is not None else None,
            "tomorrow_weather_code_source_model_id": tomorrow_source_model_id,
            "tomorrow_weather_code_source_model_label": tomorrow_source_label,
            "tomorrow_weather_code_source_max_days": tomorrow_source_max_days,
            "satellite_nowcast_used": bool(getattr(ensemble_tomorrow, "satellite_nowcast_used", False)),
            "satellite_nowcast_hours": int(getattr(ensemble_tomorrow, "satellite_nowcast_hours", 0) or 0),
            "satellite_nowcast_weight_factor": getattr(ensemble_tomorrow, "satellite_nowcast_weight_factor", None),
            "satellite_nowcast_reason": getattr(ensemble_tomorrow, "satellite_nowcast_reason", None),
            "day_type": getattr(ensemble_tomorrow, "day_type", None),
            "weights_by_model": getattr(ensemble_tomorrow, "weights_by_model", None),
            "nowcast_available": bool(getattr(ensemble_tomorrow, "nowcast_available", False)),
            "nowcast_used_hours": int(getattr(ensemble_tomorrow, "nowcast_used_hours", 0) or 0),
            "nowcast_blend_hours": int(getattr(ensemble_tomorrow, "nowcast_blend_hours", 0) or 0),
        }

    def _build_success_payload(
        self,
        *,
        run_id: str,
        target_date: dt.date,
        weather,
        ensemble_tomorrow,
        pv: pd.DataFrame,
        detail_df: pd.DataFrame,
        flows_df: pd.DataFrame,
        soc_series: pd.Series,
        metrics_payload: dict,
        pv_quality: dict,
        warnings: list[str],
        status: str,
        run_duration_ms: int,
        run_at_utc: str,
        tz: str,
        inputs_used: dict,
        pv_totals_kwh: dict,
        pv_tomorrow_low_high_kwh: dict,
        pv_week_ahead: list[dict],
        pv_temperature_metadata: dict,
        tomorrow_weather_code: int | None,
        tomorrow_source_model_id: str | None,
        tomorrow_source_label: str | None,
        tomorrow_source_max_days: int | None,
        mode: str,
        tomorrow_models_used: list[str],
        week_models: list[str],
        week_models_used_count_per_hour: pd.Series | None,
        system_snapshot: dict,
        config_hash: str,
        cfg: dict,
        ensemble_method_tomorrow: str,
        pv_uncertainty: bool,
        ensemble_method_week: str,
        pv_model_spread: dict | None,
        effective_fast_mode: bool,
        fast_mode: bool,
        store_provider_payloads: bool,
    ) -> dict:
        weather_ensemble_payload = self._build_weather_ensemble_payload(
            ensemble_tomorrow=ensemble_tomorrow,
            tomorrow_models=tomorrow_models_used,
            ensemble_method_tomorrow=ensemble_method_tomorrow,
            pv_uncertainty=pv_uncertainty,
            pv_totals_kwh=pv_totals_kwh,
            pv_tomorrow_low_high_kwh=pv_tomorrow_low_high_kwh,
            pv_model_spread=pv_model_spread,
            pv_week_ahead=pv_week_ahead,
            pv_temperature_metadata=pv_temperature_metadata,
            ensemble_method_week=ensemble_method_week,
            effective_fast_mode=effective_fast_mode,
            fast_mode=fast_mode,
            tomorrow_weather_code=tomorrow_weather_code,
            tomorrow_source_model_id=tomorrow_source_model_id,
            tomorrow_source_label=tomorrow_source_label,
            tomorrow_source_max_days=tomorrow_source_max_days,
        )
        return {
            "run_id": run_id,
            "target_date": target_date.isoformat(),
            "weather": self._serialize_df(weather.df),
            "weather_primary_model_id": ensemble_tomorrow.weather_primary_model_id,
            "weather_ensemble_table": self._serialize_df(getattr(getattr(ensemble_tomorrow, "weather_ensemble_table", None), "df", pd.DataFrame(index=pv.index))),
            "weather_by_model": {model_id: self._serialize_df(fr.df) for model_id, fr in (getattr(ensemble_tomorrow, "weather_by_model", {}) or {}).items()},
            "pv_by_model": {model_id: self._serialize_df(model_pv) for model_id, model_pv in (getattr(ensemble_tomorrow, "pv_by_model", {}) or {}).items()},
            "derived_irradiance_by_model": getattr(ensemble_tomorrow, "derived_irradiance_by_model", {}),
            "derived_weather_code_by_model": getattr(ensemble_tomorrow, "derived_weather_code_by_model", {}),
            "quality_weight_factors_by_model": getattr(ensemble_tomorrow, "quality_weight_factors_by_model", {}),
            "pv": self._serialize_df(pv),
            "detail": self._serialize_df(detail_df),
            "flows": self._serialize_df(flows_df),
            "soc": self._serialize_series(soc_series),
            "sunrise": pd.Timestamp(weather.sunrise).isoformat(),
            "sunset": pd.Timestamp(weather.sunset).isoformat(),
            "metrics": metrics_payload,
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
            "pv_temperature_metadata": pv_temperature_metadata,
            "tomorrow_weather_code": int(tomorrow_weather_code) if tomorrow_weather_code is not None else None,
            "tomorrow_weather_code_source_model_id": tomorrow_source_model_id,
            "tomorrow_weather_code_source_model_label": tomorrow_source_label,
            "tomorrow_weather_code_source_max_days": tomorrow_source_max_days,
            "forecast_mode_effective": mode,
            "tomorrow_models_used": tomorrow_models_used,
            "selected_weather_models": tomorrow_models_used,
            "model_selection_reason": getattr(ensemble_tomorrow, "model_selection_reason", None),
            "dynamic_daylight_method": getattr(ensemble_tomorrow, "dynamic_daylight_method", None),
            "forecast_quality_tier": getattr(ensemble_tomorrow, "forecast_quality_tier", None),
            "requires_pvlib": True,
            "pvlib_validated": True,
            "week_ahead_models_considered": list(week_models),
            "week_models_used": list(week_models),
            "pv_week_models_used_count_per_hour": self._serialize_series(week_models_used_count_per_hour) if isinstance(week_models_used_count_per_hour, pd.Series) else None,
            "system_snapshot": {k: v for k, v in system_snapshot.items() if v is not None},
            "planner_version": "v2",
            "data_source": {"soc": "manual", "load": "manual", "pv": "forecast"},
            "config_hash": config_hash,
            "config_json": json.dumps(
                {
                    **cfg,
                    "weather_models_selected": getattr(ensemble_tomorrow, "selected_models", tomorrow_models_used),
                    "forecast_mode": mode,
                    "ensemble_method": ensemble_method_tomorrow,
                    "pv_uncertainty_enabled": bool(pv_uncertainty),
                    "per_model_pv_totals_kwh": getattr(ensemble_tomorrow, "per_model_pv_totals_kwh", {}),
                    "pv_totals_kwh": pv_totals_kwh,
                },
                sort_keys=True,
            ),
            "config": cfg,
            "created_at_utc": run_at_utc,
            "weather_ensemble": weather_ensemble_payload,
            "provider_payloads_by_model": ((getattr(ensemble_tomorrow, "provider_payloads_by_model", {}) or {}) if store_provider_payloads else {}),
        }

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
        use_satellite_nowcast_0_6h_override: bool | None = None,
    ) -> dict:
        diagnostics = _RunDiagnostics()
        with diagnostics.stage("config_normalization"):
            normalized = self._normalize_run_config(
                target_date=target_date,
                soc_percent=soc_percent,
                weather_models=weather_models,
                forecast_mode=forecast_mode,
                ensemble_method=ensemble_method,
                use_satellite_nowcast_0_6h_override=use_satellite_nowcast_0_6h_override,
            )
            cfg = normalized["cfg"]
            loc_cfg = normalized["loc_cfg"]
            tz = normalized["tz"]
            loc = normalized["loc"]
            mode = normalized["mode"]
            tomorrow_models = normalized["tomorrow_models"]
            week_models = normalized["week_models"]
            normalized_ensemble_method = normalized["normalized_ensemble_method"]
            ensemble_method_tomorrow = normalized["ensemble_method_tomorrow"]
            ensemble_method_week = normalized["ensemble_method_week"]
            store_provider_payloads = normalized["store_provider_payloads"]
            effective_use_sat = normalized["effective_use_sat"]
            ev_warning = normalized["ev_warning"]
            fallback_to_cached_state = normalized["fallback_to_cached_state"]
            fallback_reason = normalized["fallback_reason"]
            soc_percent = float(normalized["soc_percent"])

        run_at_utc = _utc_now_iso()
        run_id = str(uuid.uuid4())
        config_hash = compute_config_hash(cfg)

        # Production planning quality: fast_mode is developer-only and explicitly opt-in via env flag.
        debug_fast_mode_enabled = str(os.getenv("PVBP_ENABLE_DEBUG_FAST_MODE", "")).strip() == "1"
        effective_fast_mode = bool(fast_mode) and bool(debug_fast_mode_enabled) and mode == "expert"

        inputs_used = {
            "soc_now_percent": float(soc_percent),
            "soc_at_22_percent": None,
            "soc_offpeak_start_estimated_percent": None,
            "soc_offpeak_start_hours_until": None,
            "soc_offpeak_start_confidence": None,
            "soc_offpeak_start_method": None,
            "pv_decision_scenario": None,
            "pv_decision_reason": None,
            "yesterday_consumption_kwh": float(yesterday_kwh),
            "buffer_percent": float(buffer_percent),
            "max_ac_charge_power_kw": float(user_max_ac_kw),
            "weather_models_selected": tomorrow_models,
            "forecast_mode": mode,
            "ensemble_method": ensemble_method_tomorrow,
            "pv_uncertainty_enabled": bool(pv_uncertainty),
            "fast_mode": bool(effective_fast_mode),
            "fast_mode_requested": bool(fast_mode),
            "use_satellite_nowcast_0_6h": bool(effective_use_sat),
        }

        try:
            with diagnostics.stage("weather_fetch"):
                ensemble_tomorrow = build_ensemble_forecast(
                    loc=loc,
                    target_date=target_date,
                    tz=tz,
                    weather_models=tomorrow_models,
                    ensemble_method=ensemble_method_tomorrow,
                    pv_uncertainty=bool(pv_uncertainty),
                    accuracy_mode=True,
                    fast_mode=bool(effective_fast_mode),
                    requested_days=1,
                    use_satellite_nowcast_0_6h=effective_use_sat,
                    expert_mode=(mode == "expert"),
                    selection_mode=("expert" if mode == "expert" else "auto"),
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
            diag_error = diagnostics.finalize(success=False, status="error", error_summary=f"{type(exc).__name__}: {exc}")
            run_duration_ms = int(float(diag_error.get("total_run_ms", 0.0)))
            error_payload = self._build_error_payload_all_models_failed(
                run_id=run_id,
                target_date=target_date,
                run_at_utc=run_at_utc,
                tz=tz,
                run_duration_ms=run_duration_ms,
                run_diagnostics=diag_error,
                warnings=warnings,
                inputs_used=inputs_used,
                cfg=cfg,
                tomorrow_models=tomorrow_models,
                mode=mode,
                ensemble_method_tomorrow=ensemble_method_tomorrow,
                pv_uncertainty=bool(pv_uncertainty),
                config_hash=config_hash,
                failed_models=failed_models,
                failed_reasons=failed_reasons,
                week_models=week_models,
                effective_fast_mode=effective_fast_mode,
                fast_mode=fast_mode,
                yesterday_kwh=float(yesterday_kwh),
                weights_used=getattr(exc, "weights_used", None),
            )
            insert_forecast_run(str(SQLITE_PATH), error_payload)
            self.latest_diagnostics = diag_error
            self.latest_result = error_payload
            self.history.append(_to_history_summary(error_payload))
            self.history = self.history[-MAX_HISTORY:]
            self._save_results()
            return error_payload

        warnings: list[str] = [ev_warning] if isinstance(ev_warning, str) and ev_warning.strip() else []
        try:
            with diagnostics.stage("weather_ensemble"):
                ensemble_week = build_ensemble_forecast(
                    loc=loc,
                    target_date=target_date,
                    tz=tz,
                    weather_models=week_models,
                    ensemble_method=ensemble_method_week,
                    pv_uncertainty=bool(pv_uncertainty),
                    accuracy_mode=True,
                    fast_mode=False,
                    requested_days=7,
                    use_satellite_nowcast_0_6h=False,
                    expert_mode=False,
                    selection_mode="auto",
                )
        except Exception as exc:
            ensemble_week = None
            warnings.append(f"pv_week_ahead_ensemble_failed={type(exc).__name__}:{exc}")

        important_weather_vars = set(WEATHER_DISPLAY_VARS)
        for model_id in getattr(ensemble_tomorrow, "failed_models", []) or []:
            failed_reasons = getattr(ensemble_tomorrow, "failed_model_reasons", {})
            reason = failed_reasons.get(model_id) if isinstance(failed_reasons, dict) else None
            reason_msg = str(reason.get("message") or reason.get("category") or "unknown") if isinstance(reason, dict) else "unknown"
            warnings.append(f"model failed: {model_id} ({reason_msg})")

        derived_hours_by_model = getattr(ensemble_tomorrow, "derived_irradiance_hours_by_model", {})
        for model_id, used_derived in (getattr(ensemble_tomorrow, "derived_irradiance_by_model", {}) or {}).items():
            derived_hours = int(derived_hours_by_model.get(model_id, 0)) if isinstance(derived_hours_by_model, dict) else 0
            if used_derived and derived_hours > 0:
                warnings.append(f"derived irradiance used: {model_id}")

        for model_id, used_cached in getattr(ensemble_tomorrow, "model_live_failed_used_cached", {}).items():
            if used_cached:
                warnings.append(f"model_live_failed_used_cached=true: {model_id}")

        for model_id, missing_vars in (getattr(ensemble_tomorrow, "missing_vars_by_model", {}) or {}).items():
            if not missing_vars:
                continue
            missing_important = sorted(var for var in set(missing_vars) if var in important_weather_vars)
            if missing_important:
                warnings.append(f"important vars missing: {model_id} ({', '.join(missing_important)})")

        if ensemble_week is not None:
            pv_totals_p50 = self._daily_totals_nullable(ensemble_week.pv_ensemble_p50, tz=tz)
            pv_totals_p10 = self._daily_totals_nullable(ensemble_week.pv_ensemble_p10, tz=tz)
            pv_totals_p90 = self._daily_totals_nullable(ensemble_week.pv_ensemble_p90, tz=tz)
            primary_id = getattr(ensemble_week, "weather_primary_model_id", None)
            weather_by_model = getattr(ensemble_week, "weather_by_model", {}) or {}
            weights_used_week = getattr(ensemble_week, "weights_used", None)
            derived_weather_code_by_model_week = getattr(ensemble_week, "derived_weather_code_by_model", {}) or {}
            week_models_used_count_per_hour = getattr(ensemble_week, "pv_models_used_count_per_hour", None)
        else:
            pv_totals_p50 = [None] * 7
            pv_totals_p10 = [None] * 7
            pv_totals_p90 = [None] * 7
            primary_id = None
            weather_by_model = {}
            weights_used_week = None
            derived_weather_code_by_model_week = {}
            week_models_used_count_per_hour = None

        week_models_count_per_day = self._daily_counts_nullable(week_models_used_count_per_hour, tz=tz)
        week_coverage_per_day = self._daily_coverage(
            getattr(ensemble_week, "pv_ensemble_p50", None) if ensemble_week is not None else None,
            tz=tz,
        )

        pv_week_ahead_all = _build_pv_week_ahead(
            target_date=target_date,
            tz=tz,
            pv_totals_p50=pv_totals_p50,
            pv_totals_p10=pv_totals_p10,
            pv_totals_p90=pv_totals_p90,
            weather_by_model=weather_by_model,
            weights_used=weights_used_week,
            weather_primary_model_id=primary_id,
            derived_weather_code_by_model=derived_weather_code_by_model_week,
            weather_ensemble_df=getattr(getattr(ensemble_week, "weather_ensemble_table", None), "df", None),
        )
        pv_week_ahead = pv_week_ahead_all[1:7]

        warnings = list(dict.fromkeys(warnings))
        status = "degraded" if warnings else "ok"

        with diagnostics.stage("pv_estimate"):
            weather = ensemble_tomorrow.weather_primary
            tomorrow_index, tomorrow_index_dst_adjusted = core.build_local_day_hour_index(target_date, tz)
        if len(tomorrow_index) != 24:
            raise RuntimeError(f"INV-T4 violation: expected 24 hourly slots, got {len(tomorrow_index)}")

        pv = pd.DataFrame(index=tomorrow_index)
        pv_temperature_metadata = core.pv_temperature_metadata()
        pv["pv_east_kwh"] = pd.to_numeric(ensemble_tomorrow.pv_ensemble_east_p50.reindex(pv.index), errors="coerce")
        pv["pv_south_kwh"] = pd.to_numeric(ensemble_tomorrow.pv_ensemble_south_p50.reindex(pv.index), errors="coerce")
        pv["pv_total_unclipped_kwh"] = pd.to_numeric(ensemble_tomorrow.pv_ensemble_unclipped_p50.reindex(pv.index), errors="coerce")
        pv["pv_total_kwh"] = pd.to_numeric(ensemble_tomorrow.pv_ensemble_p50.reindex(pv.index), errors="coerce")
        if ensemble_tomorrow.pv_ensemble_p10 is not None:
            pv["pv_total_low_kwh"] = pd.to_numeric(ensemble_tomorrow.pv_ensemble_p10.reindex(pv.index), errors="coerce")
        if ensemble_tomorrow.pv_ensemble_p90 is not None:
            pv["pv_total_high_kwh"] = pd.to_numeric(ensemble_tomorrow.pv_ensemble_p90.reindex(pv.index), errors="coerce")
        clipped_raw = (pv["pv_total_unclipped_kwh"] - pv["pv_total_kwh"]).clip(lower=0.0)
        pv["pv_clipped_kwh"] = clipped_raw.where(pv["pv_total_unclipped_kwh"].notna() & pv["pv_total_kwh"].notna())
        pv["pv_dc_available_kwh"] = pv["pv_total_unclipped_kwh"]
        pv["pv_ac_limited_kwh"] = pv["pv_total_kwh"]

        total_e = float(pd.to_numeric(pv["pv_east_kwh"], errors="coerce").fillna(0.0).sum())
        total_s = float(pd.to_numeric(pv["pv_south_kwh"], errors="coerce").fillna(0.0).sum())
        split_total = total_e + total_s
        ratio_e, ratio_s = (total_e / split_total, total_s / split_total) if split_total > 0 else (0.5, 0.5)
        pv = core.ensure_pv_columns(pv, split_ratio=(ratio_e, ratio_s))
        pv.attrs.update(pv_temperature_metadata)

        cons_profile = core.load_consumption_profile_kwh_per_hour()
        cons = core.build_consumption_forecast(cons_profile, yesterday_kwh, target_date, tz)
        offpeak_start, offpeak_end = core.compute_charging_window_for_target_date(target_date, cfg.get("tariff", {}))
        now_local = dt.datetime.now(ZoneInfo(tz))
        battery_kwh = float(cfg.get("battery", {}).get("battery_kwh", core.BATTERY_KWH))
        min_soc_percent = float(cfg.get("battery", {}).get("min_soc_percent", core.MIN_SOC_PERCENT))

        effective_daily_kwh, _effective_meta = fetch_effective_daily_kwh(SQLITE_PATH, lookback_runs=14, prefer_same_day_type=True)
        used_history = effective_daily_kwh is not None
        if effective_daily_kwh is None:
            effective_daily_kwh = float(yesterday_kwh)

        hours_until_offpeak_start = max(0.0, (offpeak_start - now_local).total_seconds() / 3600.0)
        daytime_pv_window = (
            offpeak_start.date() == now_local.date()
            and 8 <= now_local.hour < 18
            and hours_until_offpeak_start > 0
        )
        prelim_load_kwh = core.estimate_window_consumption_kwh(
            start_local=now_local,
            end_local=offpeak_start.to_pydatetime(),
            effective_daily_kwh=float(effective_daily_kwh),
        )
        pv_credit_kwh = 0.0
        pv_credit_available = not daytime_pv_window
        if daytime_pv_window:
            try:
                loc_cfg_today = cfg.get("location", {})
                today_date = now_local.date()
                selected_today_models = auto_select_models_for_location(
                    float(loc_cfg_today.get("latitude")),
                    float(loc_cfg_today.get("longitude")),
                    requested_days=1,
                )
                use_nowcast = should_use_satellite_nowcast_auto(
                    float(loc_cfg_today.get("latitude")),
                    float(loc_cfg_today.get("longitude")),
                    requested_days=1,
                ) and hours_until_offpeak_start <= 6
                ensemble_today = build_ensemble_forecast(
                    loc=loc,
                    target_date=today_date,
                    tz=tz,
                    weather_models=selected_today_models,
                    ensemble_method="weighted",
                    pv_uncertainty=False,
                    requested_days=1,
                    use_satellite_nowcast_0_6h=use_nowcast,
                    expert_mode=False,
                    selection_mode="auto",
                )
                start_h = core._to_local_ts(now_local, tz).ceil("h")
                end_h = core._to_local_ts(offpeak_start, tz).floor("h")
                today_remaining_pv_kwh = 0.0
                if end_h > start_h:
                    pv_window = pd.to_numeric(
                        ensemble_today.pv_ensemble_p50.reindex(pd.date_range(start_h, end_h, freq="h", inclusive="left")),
                        errors="coerce",
                    ).fillna(0.0)
                    today_remaining_pv_kwh = float(pv_window.sum())
                pv_credit_kwh = min(today_remaining_pv_kwh * 0.5, prelim_load_kwh)
                pv_credit_available = True
            except Exception as exc:
                warnings.append(f"PV credit unavailable for SOC off-peak estimate: {exc}")
                pv_credit_kwh = 0.0
                pv_credit_available = False

        estimated_soc_percent, hours_until_offpeak_start, soc_offpeak_confidence, soc_offpeak_method, soc_offpeak_debug = core.estimate_soc_at_offpeak_start(
            soc_now_percent=float(soc_percent),
            now_local=now_local,
            offpeak_start=offpeak_start,
            effective_daily_kwh=float(effective_daily_kwh),
            pv_credit_kwh=float(pv_credit_kwh),
            battery_kwh=battery_kwh,
            min_soc_percent=min_soc_percent,
            used_history=bool(used_history),
            pv_credit_available=bool(pv_credit_available),
        )
        inputs_used["soc_at_22_percent"] = float(estimated_soc_percent)
        inputs_used["soc_offpeak_start_estimated_percent"] = float(estimated_soc_percent)
        decision_quantile, decision_reason = pick_decision_quantile(soc_offpeak_confidence, uncertainty_enabled=bool(pv_uncertainty))
        if decision_quantile == "p50":
            decision_series = ensemble_tomorrow.pv_ensemble_p50
        else:
            decision_series = getattr(ensemble_tomorrow, "pv_ensemble_p25", None)
        if decision_series is None:
            decision_series = ensemble_tomorrow.pv_ensemble_p50
        if decision_series is None:
            decision_series = ensemble_tomorrow.pv_ensemble_p50
            pv["pv_total_decision_kwh"] = pd.to_numeric(decision_series.reindex(pv.index), errors="coerce")

        with diagnostics.stage("planner_simulation"):
            detail_df, flows_df, soc_series, charge_kw, cutoff_soc, cutoff_reason, charge_note, charge_target_reachable, charge_warning_text = core.run_detailed_plan(
                target_date=target_date,
                weather=weather,
                pv_df=pv,
                consumption_kwh=cons,
                soc_at_22_percent=estimated_soc_percent,
                buffer_percent=buffer_percent,
                max_ac_charge_power_kw=user_max_ac_kw,
            )
        cutoff_adjusted_for_pv_headroom = str(cutoff_reason).startswith("CONFLICT")
        cutoff_adjustment_reason = (
            "Cutoff lowered to preserve PV headroom for daytime surplus handling."
            if cutoff_adjusted_for_pv_headroom
            else ""
        )
        pv_surplus_store_econ_enabled = bool(flows_df.attrs.get("pv_surplus_store_econ_enabled", False))
        pv_surplus_export_preferred_kwh = float(flows_df.attrs.get("pv_surplus_export_preferred_kwh", 0.0))
        pv_surplus_store_preferred_kwh = float(flows_df.attrs.get("pv_surplus_store_preferred_kwh", 0.0))
        pv_store_vs_export_decisions_count = int(flows_df.attrs.get("pv_store_vs_export_decisions_count", 0))
        allow_injection_to_grid = bool(flows_df.attrs.get("allow_injection_to_grid", cfg.get("tariff", {}).get("allow_injection_to_grid", True)))
        export_blocked_by_policy = bool(flows_df.attrs.get("export_blocked_by_policy", not allow_injection_to_grid))
        blocked_export_kwh_total = float(flows_df.attrs.get("blocked_export_kwh_total", 0.0))
        max_grid_import_kw = float(flows_df.attrs.get("max_grid_import_kw", cfg.get("tariff", {}).get("max_grid_import_kw", 0.0)))
        grid_import_cap_active = bool(flows_df.attrs.get("grid_import_cap_active", max_grid_import_kw > 0.0))
        grid_import_cap_binding_events = int(flows_df.attrs.get("grid_import_cap_binding_events", 0))
        charge_effective_cap_kw = float(flows_df.attrs.get("charge_effective_cap_kw", 0.0))
        charge_limit_reason_raw = str(flows_df.attrs.get("charge_limit_reason_raw", "none") or "none")
        grid_import_cap_load_exceeds_events = int(flows_df.attrs.get("grid_import_cap_load_exceeds_events", 0))
        grid_import_cap_limited_charge_kwh_total = float(flows_df.attrs.get("grid_import_cap_limited_charge_kwh_total", 0.0))
        if not allow_injection_to_grid:
            warnings.append(
                "Grid injection is disabled by settings. Excess PV will be curtailed when battery storage is not available."
            )
        if grid_import_cap_active:
            warnings.append(
                f"Grid import cap is enabled: {max_grid_import_kw:g} kW. Battery grid charging may be limited to respect the cap."
            )
        if grid_import_cap_load_exceeds_events > 0:
            warnings.append("Grid import cap exceeded by household load in some hours (cannot be enforced).")
        if grid_import_cap_binding_events > 0:
            warnings.append("Grid import cap limited battery charging; target may be unreachable.")
        charge_limit_reason = "grid_import_cap" if grid_import_cap_binding_events > 0 else charge_limit_reason_raw
        grid_import = float(flows_df.get("grid_import_kwh", pd.Series(dtype=float)).sum())
        grid_export = float(flows_df.get("grid_export_kwh", pd.Series(dtype=float)).sum())
        canonical_tomorrow_total_kwh = (float(pd.to_numeric(pv["pv_total_kwh"], errors="coerce").sum(min_count=1)) if "pv_total_kwh" in pv.columns else None)
        if canonical_tomorrow_total_kwh is not None and pd.isna(canonical_tomorrow_total_kwh):
            canonical_tomorrow_total_kwh = float(ensemble_tomorrow.pv_ensemble_p50.sum(min_count=1)) if ensemble_tomorrow.pv_ensemble_p50 is not None else None
        tomorrow_coverage_hours = int(pd.to_numeric(pv.get("pv_total_kwh", pd.Series(dtype=float)), errors="coerce").notna().sum())
        pv_forecast_kwh = canonical_tomorrow_total_kwh
        cons_forecast_kwh = float(cons.sum())
        if "load_kwh" not in pv.columns:
            pv = core.add_load_and_surplus_columns(pv, cons_forecast_kwh)
        with diagnostics.stage("savings_evaluation"):
            quality_sig = inspect.signature(scoring.compute_pv_quality_score)
            quality_kwargs = {
                "pv_df": pv,
                "weather_df": weather.df,
                "target_date": target_date,
                "tz": tz,
                "fallback_score": 55,
            }
            if "loc" in quality_sig.parameters:
                quality_kwargs["loc"] = loc
            pv_quality = scoring.compute_pv_quality_score(**quality_kwargs)

            today_date = target_date - dt.timedelta(days=1)
            # Savings cycle must start from estimated off-peak SOC for continuity with run_detailed_plan().
            savings = core.compute_euro_savings_no_battery_vs_plan(
                pv_df=pv,
                flows_df=flows_df,
                soc_at_22=float(estimated_soc_percent) / 100.0,
                charge_kw=charge_kw,
                cutoff_soc=cutoff_soc,
                today_date=today_date,
                tomorrow_date=target_date,
                total_consumption_kwh=cons_forecast_kwh,
                tariff_cfg=(cfg.get("tariff", {}) if isinstance(cfg, dict) else {}),
            )
            pv_quality.update(savings)

        pv_totals_kwh = {
            "p50": canonical_tomorrow_total_kwh,
            "p10": (
                float(pd.to_numeric(pv["pv_total_low_kwh"], errors="coerce").sum(min_count=1))
                if "pv_total_low_kwh" in pv.columns and not pd.isna(pd.to_numeric(pv["pv_total_low_kwh"], errors="coerce").sum(min_count=1))
                else (float(ensemble_tomorrow.pv_ensemble_p10.sum(min_count=1)) if ensemble_tomorrow.pv_ensemble_p10 is not None else None)
            ),
            "p90": (
                float(pd.to_numeric(pv["pv_total_high_kwh"], errors="coerce").sum(min_count=1))
                if "pv_total_high_kwh" in pv.columns and not pd.isna(pd.to_numeric(pv["pv_total_high_kwh"], errors="coerce").sum(min_count=1))
                else (float(ensemble_tomorrow.pv_ensemble_p90.sum(min_count=1)) if ensemble_tomorrow.pv_ensemble_p90 is not None else None)
            ),
        }
        pv_model_spread = getattr(ensemble_tomorrow, "pv_tomorrow_model_spread_kwh", None)
        if not isinstance(pv_model_spread, dict):
            legacy_pv_low_high = getattr(ensemble_tomorrow, "pv_tomorrow_low_high_kwh", None)
            pv_model_spread = legacy_pv_low_high if isinstance(legacy_pv_low_high, dict) else None

        selected_models = [str(m) for m in getattr(ensemble_tomorrow, "selected_models", []) if isinstance(m, str)]
        failed_models = {str(m) for m in getattr(ensemble_tomorrow, "failed_models", []) if isinstance(m, str)}
        per_model_pv_totals = getattr(ensemble_tomorrow, "per_model_pv_totals_kwh", {})
        quantile_contributors = [
            model_id
            for model_id in selected_models
            if model_id not in failed_models and pd.notna(pd.to_numeric(pd.Series([per_model_pv_totals.get(model_id)]), errors="coerce").iloc[0])
        ]
        fallback_valid_models = int(len(quantile_contributors))
        spread_valid_models_raw = (pv_model_spread or {}).get("valid_models") if isinstance(pv_model_spread, dict) else None
        spread_valid_models_num = pd.to_numeric(pd.Series([spread_valid_models_raw]), errors="coerce").iloc[0]
        spread_valid_models = int(spread_valid_models_num) if not pd.isna(spread_valid_models_num) else 0
        valid_models_for_range = spread_valid_models if spread_valid_models > 0 else fallback_valid_models

        p10_num = pd.to_numeric(pd.Series([pv_totals_kwh.get("p10")]), errors="coerce").iloc[0]
        p90_num = pd.to_numeric(pd.Series([pv_totals_kwh.get("p90")]), errors="coerce").iloc[0]
        if not pd.isna(p10_num) and not pd.isna(p90_num):
            pv_tomorrow_low_high_kwh = {
                "low": float(p10_num),
                "high": float(p90_num),
                "valid_models": int(valid_models_for_range),
            }
        else:
            pv_tomorrow_low_high_kwh = {
                "low": None,
                "high": None,
                "valid_models": int(valid_models_for_range),
            }

        tomorrow_weather_code = _best_of_day_weather_code(weather.df.reindex(pv.index)[["weather_code"]]) if "weather_code" in weather.df.columns else None
        if tomorrow_weather_code is None and "weather_code" in weather.df.columns:
            wc_first = pd.to_numeric(weather.df.reindex(pv.index)["weather_code"], errors="coerce").dropna()
            tomorrow_weather_code = int(wc_first.iloc[0]) if not wc_first.empty else None
        tomorrow_source_model_id = getattr(ensemble_tomorrow, "weather_primary_model_id", None)
        tomorrow_source_label = (WEATHER_MODELS.get(tomorrow_source_model_id) or {}).get("label") if tomorrow_source_model_id else None
        tomorrow_source_max_days = _model_max_days(tomorrow_source_model_id) if tomorrow_source_model_id else None

        inv_warnings: list[str] = []
        hourly_sum = float(pd.to_numeric(pv["pv_total_kwh"], errors="coerce").sum(min_count=1)) if "pv_total_kwh" in pv.columns else float("nan")
        if (canonical_tomorrow_total_kwh is not None and not pd.isna(canonical_tomorrow_total_kwh) and not pd.isna(hourly_sum) and abs(canonical_tomorrow_total_kwh - hourly_sum) > 0.01):
            inv_warnings.append("INV-T1 failed: forecast total PV != PV Outlook hourly sum")
        total_series = pd.to_numeric(pv["pv_total_kwh"], errors="coerce")
        if ((total_series < 0) & total_series.notna()).any():
            inv_warnings.append("INV-T2 failed: negative hourly PV detected")
        east_south = pd.to_numeric(pv["pv_east_kwh"], errors="coerce") + pd.to_numeric(pv["pv_south_kwh"], errors="coerce")
        mismatch = (total_series - east_south).abs()
        if ((mismatch > 0.01) & total_series.notna() & east_south.notna()).any():
            inv_warnings.append("INV-T3 failed: pv_total_kwh != pv_east_kwh + pv_south_kwh")
        if len(pv.index) != 24 or pv.index.has_duplicates:
            inv_warnings.append("INV-T4 failed: tomorrow index is not exactly 24 unique hourly points")
        if tomorrow_index_dst_adjusted:
            inv_warnings.append("INV-T4 note: DST normalization applied to preserve 24 hourly points")
        warnings.extend(inv_warnings)
        warnings = list(dict.fromkeys(warnings))
        status = "degraded" if warnings else "ok"

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
        tomorrow_models_used = list(getattr(ensemble_tomorrow, "selected_models", []) or [])

        inputs_used["soc_offpeak_start_hours_until"] = float(hours_until_offpeak_start)
        inputs_used["soc_offpeak_start_confidence"] = str(soc_offpeak_confidence)
        inputs_used["soc_offpeak_start_method"] = str(soc_offpeak_method)
        inputs_used["soc_offpeak_start_load_kwh_window"] = float(soc_offpeak_debug.get("load_kwh_window", 0.0))
        inputs_used["soc_offpeak_start_pv_credit_kwh"] = float(soc_offpeak_debug.get("pv_credit_kwh", 0.0))
        inputs_used["soc_offpeak_start_effective_daily_kwh_used"] = float(soc_offpeak_debug.get("effective_daily_kwh_used", effective_daily_kwh))
        inputs_used["soc_offpeak_start_used_history"] = bool(soc_offpeak_debug.get("used_history", used_history))
        inputs_used["soc_offpeak_start_pv_credit_available"] = bool(soc_offpeak_debug.get("pv_credit_available", pv_credit_available))
        inputs_used["soc_offpeak_start_peak_overlap"] = bool(soc_offpeak_debug.get("peak_overlap", False))
        inputs_used["pv_decision_scenario"] = decision_quantile
        inputs_used["pv_decision_reason"] = decision_reason

        diag_success = diagnostics.finalize(success=True, status=status)
        run_duration_ms = int(float(diag_success.get("total_run_ms", 0.0)))

        with diagnostics.stage("response_build"):
            metrics_payload = self._build_run_metrics_payload(
                charge_kw=charge_kw,
                cutoff_soc=cutoff_soc,
                cutoff_reason=cutoff_reason,
                charge_note=charge_note,
                charge_target_reachable=bool(charge_target_reachable),
                charge_warning_text=str(charge_warning_text or ""),
                charge_effective_cap_kw=float(charge_effective_cap_kw),
                charge_limit_reason=str(charge_limit_reason),
                grid_import=float(grid_import),
                grid_export=float(grid_export),
                pv_forecast_kwh=pv_forecast_kwh,
                cons_forecast_kwh=cons_forecast_kwh,
                tomorrow_coverage_hours=tomorrow_coverage_hours,
                offpeak_start=offpeak_start,
                offpeak_end=offpeak_end,
                estimated_soc_percent=float(estimated_soc_percent),
                hours_until_offpeak_start=float(hours_until_offpeak_start),
                soc_offpeak_confidence=str(soc_offpeak_confidence),
                soc_offpeak_method=str(soc_offpeak_method),
                soc_offpeak_debug=soc_offpeak_debug,
                effective_daily_kwh=float(effective_daily_kwh),
                used_history=bool(used_history),
                pv_credit_available=bool(pv_credit_available),
                decision_quantile=decision_quantile,
                decision_reason=decision_reason,
                cutoff_adjusted_for_pv_headroom=bool(cutoff_adjusted_for_pv_headroom),
                cutoff_adjustment_reason=str(cutoff_adjustment_reason),
                pv_surplus_store_econ_enabled=bool(pv_surplus_store_econ_enabled),
                pv_surplus_export_preferred_kwh=float(pv_surplus_export_preferred_kwh),
                pv_surplus_store_preferred_kwh=float(pv_surplus_store_preferred_kwh),
                pv_store_vs_export_decisions_count=int(pv_store_vs_export_decisions_count),
                allow_injection_to_grid=bool(allow_injection_to_grid),
                export_blocked_by_policy=bool(export_blocked_by_policy),
                blocked_export_kwh_total=float(blocked_export_kwh_total),
                max_grid_import_kw=float(max_grid_import_kw),
                grid_import_cap_active=bool(grid_import_cap_active),
                grid_import_cap_binding_events=int(grid_import_cap_binding_events),
                grid_import_cap_limited_charge_kwh_total=float(grid_import_cap_limited_charge_kwh_total),
                grid_import_cap_load_exceeds_events=int(grid_import_cap_load_exceeds_events),
                week_models=week_models,
                ensemble_tomorrow=ensemble_tomorrow,
                week_models_used_count_per_hour=week_models_used_count_per_hour,
                week_models_count_per_day=week_models_count_per_day,
                week_coverage_per_day=week_coverage_per_day,
            )
            payload = self._build_success_payload(
                run_id=run_id,
                target_date=target_date,
                weather=weather,
                ensemble_tomorrow=ensemble_tomorrow,
                pv=pv,
                detail_df=detail_df,
                flows_df=flows_df,
                soc_series=soc_series,
                metrics_payload=metrics_payload,
                pv_quality=pv_quality,
                warnings=warnings,
                status=status,
                run_duration_ms=run_duration_ms,
                run_at_utc=run_at_utc,
                tz=tz,
                inputs_used=inputs_used,
                pv_totals_kwh=pv_totals_kwh,
                pv_tomorrow_low_high_kwh=pv_tomorrow_low_high_kwh,
                pv_week_ahead=pv_week_ahead,
                pv_temperature_metadata=pv_temperature_metadata,
                tomorrow_weather_code=tomorrow_weather_code,
                tomorrow_source_model_id=tomorrow_source_model_id,
                tomorrow_source_label=tomorrow_source_label,
                tomorrow_source_max_days=tomorrow_source_max_days,
                mode=mode,
                tomorrow_models_used=tomorrow_models_used,
                week_models=week_models,
                week_models_used_count_per_hour=week_models_used_count_per_hour,
                system_snapshot=system_snapshot,
                config_hash=config_hash,
                cfg=cfg,
                ensemble_method_tomorrow=ensemble_method_tomorrow,
                pv_uncertainty=bool(pv_uncertainty),
                ensemble_method_week=ensemble_method_week,
                pv_model_spread=pv_model_spread,
                effective_fast_mode=effective_fast_mode,
                fast_mode=fast_mode,
                store_provider_payloads=store_provider_payloads,
            )
        with diagnostics.stage("db_write"):
            insert_forecast_run(str(SQLITE_PATH), payload)
            self.latest_diagnostics = diag_success
            self.latest_result = payload
        self.history.append(_to_history_summary(payload))
        self.history = self.history[-MAX_HISTORY:]
        self.settings["last_successful_for_target_date"] = target_date.isoformat()
        self._save_results()
        self._save_settings()
        final_diag = diagnostics.finalize(success=True, status=status)
        payload["run_diagnostics"] = final_diag
        payload["run_duration_ms"] = int(float(final_diag.get("total_run_ms", 0.0)))
        self.latest_diagnostics = final_diag
        del detail_df, flows_df, pv, weather
        gc.collect()
        return payload

    def run_now(self, payload: RunNowPayload) -> dict:
        if not self._lock.acquire(blocking=False):
            raise HTTPException(status_code=423, detail="Run already in progress")
        try:
            source = self.last_inputs
            warnings: list[str] = [*self.last_inputs_sanitized_warnings, *self.settings_sanitized_warnings]
            soc = _resolve_soc_percent(
                payload_soc_now=payload.soc_now_percent,
                payload_soc_legacy=payload.soc_at_22_percent,
                source=source,
                warnings=warnings,
            )

            ykwh = _coerce_float(
                payload.yesterday_consumption_kwh if payload.yesterday_consumption_kwh is not None else source.get("yesterday_consumption_kwh"),
                18.0,
                field_name="yesterday_consumption_kwh",
                warnings=warnings,
            )
            if ykwh <= 0:
                warnings.append("yesterday_consumption_kwh: must be > 0 -> default 18.0")
                ykwh = 18.0

            cap = _coerce_float(
                payload.user_max_ac_kw if payload.user_max_ac_kw is not None else self.settings.get("max_ac_charge_power_kw_default"),
                DEFAULT_MAX_AC_CAP,
                field_name="max_ac_charge_power_kw_default",
                warnings=warnings,
            )
            if cap <= 0:
                warnings.append(f"max_ac_charge_power_kw_default: must be > 0 -> default {DEFAULT_MAX_AC_CAP}")
                cap = DEFAULT_MAX_AC_CAP
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
                use_satellite_nowcast_0_6h_override=payload.use_satellite_nowcast_0_6h,
            )
            result["run_type"] = "manual"
            if warnings:
                existing = result.get("warnings") if isinstance(result.get("warnings"), list) else []
                result["warnings"] = [*existing, *warnings]
                result["input_warnings"] = warnings
            self.latest_result = result
            run_response = {"ran": True, "result": result}
            payload_size = self.record_endpoint_payload_size("/v1/run/now", run_response)
            if DEBUG:
                stage_timings = self.latest_diagnostics.get("stage_timings_ms", {}) if isinstance(self.latest_diagnostics, dict) else {}
                slowest = max(stage_timings.items(), key=lambda kv: kv[1]) if isinstance(stage_timings, dict) and stage_timings else None
                logger.info("run diagnostics run_id=%s total_ms=%.1f slowest_stage=%s payload_bytes=%d status=%s", result.get("run_id"), float(result.get("run_duration_ms", 0.0)), (slowest[0] if slowest else None), int(payload_size), result.get("status"))
            if self.history:
                self.history[-1]["run_type"] = "manual"
            self._save_results()
            return run_response
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

            warnings: list[str] = [*self.last_inputs_sanitized_warnings, *self.settings_sanitized_warnings]
            soc = _resolve_soc_percent(
                payload_soc_now=None,
                payload_soc_legacy=None,
                source=self.last_inputs,
                warnings=warnings,
            )

            ykwh = _coerce_float(
                self.last_inputs.get("yesterday_consumption_kwh"),
                18.0,
                field_name="yesterday_consumption_kwh",
                warnings=warnings,
            )
            if ykwh <= 0:
                warnings.append("yesterday_consumption_kwh: must be > 0 -> default 18.0")
                ykwh = 18.0
            if not self.last_inputs:
                warnings.append("missing inputs")
            updated_at_raw = self.last_inputs.get("last_inputs_updated_at")
            if updated_at_raw:
                updated_at = dt.datetime.fromisoformat(updated_at_raw)
                if (local_now - updated_at) > dt.timedelta(hours=24):
                    warnings.append("stale inputs")

            cap = _coerce_float(
                self.settings.get("max_ac_charge_power_kw_default"),
                DEFAULT_MAX_AC_CAP,
                field_name="max_ac_charge_power_kw_default",
                warnings=warnings,
            )
            if cap <= 0:
                warnings.append(f"max_ac_charge_power_kw_default: must be > 0 -> default {DEFAULT_MAX_AC_CAP}")
                cap = DEFAULT_MAX_AC_CAP

            result = self._run(
                target_date,
                soc,
                ykwh,
                0.0,
                cap,
                DEFAULT_ACCURACY_MODELS,
                "auto",
                "weighted",
                False,
            )
            existing = result.get("warnings") if isinstance(result.get("warnings"), list) else []
            result["warnings"] = [*existing, *warnings]
            if warnings:
                result["input_warnings"] = warnings
            result["run_type"] = "nightly"
            self.latest_result = result
            run_response = {"ran": True, "result": result}
            payload_size = self.record_endpoint_payload_size("/v1/run/now", run_response)
            if DEBUG:
                stage_timings = self.latest_diagnostics.get("stage_timings_ms", {}) if isinstance(self.latest_diagnostics, dict) else {}
                slowest = max(stage_timings.items(), key=lambda kv: kv[1]) if isinstance(stage_timings, dict) and stage_timings else None
                logger.info("run diagnostics run_id=%s total_ms=%.1f slowest_stage=%s payload_bytes=%d status=%s", result.get("run_id"), float(result.get("run_duration_ms", 0.0)), (slowest[0] if slowest else None), int(payload_size), result.get("status"))
            if self.history:
                self.history[-1]["warnings"] = warnings
                self.history[-1]["run_type"] = "nightly"
            self._save_results()
            return {"ran": True, "reason": "ran", "target_date": target_date.isoformat(), "warnings": warnings, "result": result}
        finally:
            self._lock.release()


app = FastAPI(title="PV Battery Planner Backend")


@app.on_event("startup")
def _validate_forecast_runtime_dependencies() -> None:
    validate_pvlib_runtime(require_production_quality=True)



def _log_backend_error_event(*, request: Request, exc: BaseException, error_type: str, severity: str, title: str, extra: dict | None = None) -> None:
    where = f"backend_api:{request.url.path}"
    body = format_exception_body(title=title, where=where, exc=exc, extra=extra)
    dedupe_key = compute_dedupe_key(source="backend", error_type=error_type, where=where, title=title, body=body)
    try:
        insert_error_event(
            str(SQLITE_PATH),
            source="backend",
            severity=severity,
            error_type=error_type,
            where=where,
            title=title,
            body=body,
            context=extra,
            dedupe_key=dedupe_key,
        )
    except Exception:
        pass


@app.exception_handler(core.ExternalServiceError)
def external_service_error_handler(request: Request, exc: core.ExternalServiceError):
    extra = {
        "method": request.method,
        "path": request.url.path,
        "query": dict(request.query_params),
        "client_host": getattr(request.client, "host", None),
        "service": getattr(exc, "service", None),
        "category": getattr(exc, "category", None),
        "hint": getattr(exc, "hint", None),
    }
    _log_backend_error_event(
        request=request,
        exc=exc,
        error_type="external_service",
        severity="error",
        title=f"Backend external service error: {request.method} {request.url.path}",
        extra=extra,
    )
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


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    _log_backend_error_event(
        request=request,
        exc=exc,
        error_type="http_error",
        severity="warning" if 400 <= int(exc.status_code) < 500 else "error",
        title=f"Backend HTTP error: {request.method} {request.url.path}",
        extra={"status_code": exc.status_code, "detail": exc.detail},
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    _log_backend_error_event(
        request=request,
        exc=exc,
        error_type="validation",
        severity="warning",
        title=f"Backend validation error: {request.method} {request.url.path}",
        extra={"errors": exc.errors()},
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    _log_backend_error_event(
        request=request,
        exc=exc,
        error_type="exception",
        severity="error",
        title=f"Backend exception: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "query": dict(request.query_params),
            "client_host": getattr(request.client, "host", None),
        },
    )
    payload = {
        "error": "internal_server_error",
        "detail": str(exc),
    }
    if DEBUG:
        payload["traceback"] = traceback.format_exc()
    return JSONResponse(status_code=500, content=payload)


state = BackendState()
evse_mgr = ocpp_evse.OcppEvseManager()


@app.websocket("/ocpp")
async def ocpp_ws(websocket: WebSocket):
    enabled = False
    user = ""
    pw = ""
    try:
        if SETTINGS_PATH.exists():
            payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
            cc = cfg.get("car_charger", {}) if isinstance(cfg, dict) else {}
            enabled = bool(cc.get("enabled", False))
            user = str(cc.get("basic_user", "") or "")
            pw = str(cc.get("basic_pass", "") or "")
    except Exception:
        pass

    await evse_mgr.handle_websocket(websocket, enabled=enabled, basic_user=user, basic_pass=pw)


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


@app.post("/v1/settings/factory_settings")
def reset_settings_to_factory_defaults(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.reset_settings_to_factory_defaults()


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




@app.get("/v1/score/day")
def get_score_day(date: str, source: str = "manual_csv", authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    row = fetch_daily_score_for_date(str(SQLITE_PATH), score_date=date, source=source)
    if row is None:
        raise HTTPException(status_code=404, detail="Daily score not found")
    return row


@app.get("/v1/ev/vehicles")
def get_ev_vehicles(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return {"vehicles": state.bmw_service.vehicles()}


@app.get("/v1/ev/provider_status")
def get_ev_provider_status(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.bmw_service.provider_status()


@app.get("/v1/ev/status")
def get_ev_status(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.get_unified_ev_status()


@app.post("/v1/ev/manual_refresh")
def ev_manual_refresh(force_reprobe: bool = False, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.bmw_service.manual_refresh(force_reprobe=force_reprobe)


@app.post("/v1/ev/bmw/device_flow/start")
def ev_bmw_device_flow_start(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.bmw_service.start_device_flow()


class BmwDeviceTokenPayload(BaseModel):
    device_code: str


@app.post("/v1/ev/bmw/device_flow/poll")
def ev_bmw_device_flow_poll(payload: BmwDeviceTokenPayload, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return state.bmw_service.poll_device_token(payload.device_code)


@app.get("/v1/ev/bmw/device_flow/debug")
def ev_bmw_device_flow_debug(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    if not DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    return state.bmw_service.device_flow_debug_info()


@app.get("/v1/evse/status")
def evse_status(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    enabled = False
    try:
        if SETTINGS_PATH.exists():
            payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
            cc = cfg.get("car_charger", {}) if isinstance(cfg, dict) else {}
            enabled = bool(cc.get("enabled", False))
    except Exception:
        pass
    out = evse_mgr.status_dict()
    out["enabled"] = enabled
    out["ws_path"] = "/ocpp"
    return out


@app.post("/v1/evse/stop")
async def evse_stop(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return await evse_mgr.remote_stop()


@app.post("/v1/evse/resume")
async def evse_resume(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return await evse_mgr.remote_resume(connector_id=1, id_tag="LOCAL")


@app.get("/v1/errors")
def get_errors(
    limit: int = 200,
    include_fixed: bool = False,
    authorization: str | None = Header(default=None),
) -> dict:
    _require_token(authorization)
    parsed_limit = int(limit)
    limit_arg: int | None
    if parsed_limit <= 0:
        limit_arg = None
    else:
        limit_arg = max(1, min(10000, parsed_limit))
    return {"items": fetch_error_events(str(SQLITE_PATH), limit=limit_arg, include_fixed=bool(include_fixed))}


@app.get("/v1/errors/{error_id}")
def get_error_by_id(error_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    item = fetch_error_event_by_id(str(SQLITE_PATH), error_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Error event not found")
    return item


@app.post("/v1/errors")
def post_error(payload: ErrorEventPayload, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    if payload.source not in {"frontend", "backend"}:
        raise HTTPException(status_code=400, detail="Invalid source")
    if payload.severity not in {"error", "warning"}:
        raise HTTPException(status_code=400, detail="Invalid severity")
    if payload.error_type not in {"exception", "http_error", "network", "validation", "external_service", "ui_state", "unknown"}:
        raise HTTPException(status_code=400, detail="Invalid error_type")

    dedupe_key = compute_dedupe_key(
        source=payload.source,
        error_type=payload.error_type,
        where=payload.where,
        title=payload.title,
        body=payload.body,
    )
    error_id = insert_error_event(
        str(SQLITE_PATH),
        source=payload.source,
        severity=payload.severity,
        error_type=payload.error_type,
        where=payload.where,
        title=payload.title,
        body=payload.body,
        context=payload.context,
        dedupe_key=dedupe_key,
    )
    return {"error_id": error_id}


@app.post("/v1/errors/{error_id}/fixed")
def post_error_fixed(error_id: str, payload: ErrorFixedPayload, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    set_error_fixed(str(SQLITE_PATH), error_id=error_id, fixed=bool(payload.fixed))
    return {"ok": True}


@app.delete("/v1/errors/{error_id}")
def delete_one_error(error_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    delete_error_event(str(SQLITE_PATH), error_id=error_id)
    return {"ok": True}


@app.delete("/v1/errors")
def delete_errors(only_fixed: bool = False, authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    deleted = delete_all_error_events(str(SQLITE_PATH), only_fixed=bool(only_fixed))
    return {"ok": True, "deleted": int(deleted)}


@app.get("/v1/weather/models")
def weather_models(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    return {"items": weather_models_payload()}


@app.get("/v1/results/latest")
def latest_result(authorization: str | None = Header(default=None)) -> dict:
    _require_token(authorization)
    db_payload = fetch_latest_full_run(str(SQLITE_PATH))
    out = db_payload if db_payload is not None else state.latest_result
    state.record_endpoint_payload_size("/v1/results/latest", out)
    return out



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
    out = {"items": items} if items else {"items": state.history[-limit_days:]}
    state.record_endpoint_payload_size("/v1/results/history", out)
    return out
