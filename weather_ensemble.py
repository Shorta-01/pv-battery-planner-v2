from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import planner_core as core
import db_sqlite

DEFAULT_ACCURACY_MODELS = ["knmi_harmonie_arome", "dwd_icon_d2", "ecmwf_ifs"]
DEFAULT_WEIGHTED_BELGIUM = {
    "knmi_harmonie_arome": 0.45,
    "dwd_icon_d2": 0.35,
    "ecmwf_ifs": 0.20,
    "dwd_icon_eu": 0.10,
    "meteofrance_seamless": 0.10,
}

WEATHER_MODELS: dict[str, dict[str, Any]] = {
    "knmi_harmonie_arome": {
        "label": "KNMI HARMONIE-AROME",
        "endpoint": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "knmi_harmonie_arome_netherlands"},
        "badges": ["⭐", "🔎", "🧩"],
        "recommended_for_be": True,
        "capability": {
            "ghi_native": True,
            "direct_native": False,
            "diffuse_native": True,
            "dni_native": True,
            "notes": "GHI only; direct/diffuse derived by Open-Meteo separation.",
        },
    },
    "dwd_icon_d2": {
        "label": "DWD ICON-D2",
        "endpoint": "https://api.open-meteo.com/v1/dwd-icon",
        "params": {"models": "icon_d2"},
        "badges": ["⭐", "🔎", "🟩", "⏱️"],
        "recommended_for_be": True,
        "capability": {
            "ghi_native": True,
            "direct_native": True,
            "diffuse_native": True,
            "dni_native": True,
            "notes": "Can provide shortwave/direct/diffuse/DNI.",
        },
    },
    "ecmwf_ifs": {
        "label": "ECMWF IFS",
        "endpoint": "https://api.open-meteo.com/v1/ecmwf",
        "params": {},
        "badges": ["⭐", "🌍", "🟩"],
        "recommended_for_be": True,
        "capability": {
            "ghi_native": True,
            "direct_native": True,
            "diffuse_native": True,
            "dni_native": True,
            "notes": "Direct/diffuse may be approximated on open-data feeds.",
        },
    },
    "dwd_icon_eu": {
        "label": "DWD ICON-EU",
        "endpoint": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "icon_eu"},
        "badges": ["🗺️", "🟩"],
        "recommended_for_be": True,
        "capability": {
            "ghi_native": True,
            "direct_native": True,
            "diffuse_native": True,
            "dni_native": True,
            "notes": "ICON EU via /v1/forecast. Radiation components should be available depending on Open-Meteo feed.",
        },
    },
    "meteofrance_seamless": {
        "label": "METEO-FRANCE SEAMLESS",
        "endpoint": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "meteofrance_seamless"},
        "badges": ["🗺️", "🧩"],
        "recommended_for_be": True,
        "capability": {
            "ghi_native": True,
            "direct_native": False,
            "diffuse_native": True,
            "dni_native": True,
            "notes": "Seamless provider blend; direct/diffuse/DNI may be derived depending on Open-Meteo feed.",
        },
    },
}

BASE_HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
]

IRRADIANCE_HOURLY_VARIABLES = [
    "direct_normal_irradiance",
    "diffuse_radiation",
    "direct_radiation",
]

WEATHER_DISPLAY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
]

FORECAST_FALLBACK_MODELS: dict[str, str] = {
    "knmi_harmonie_arome": "knmi_seamless",
    "dwd_icon_d2": "icon_d2",
    "ecmwf_ifs": "ifs",
}

_WEATHER_CACHE: dict[tuple, tuple[float, core.ForecastResult, list[str], bool]] = {}
_WEATHER_CACHE_TTL_S = 600
_SESSION: requests.Session | None = None
_LOGGER = logging.getLogger(__name__)

PROVIDER_CACHE_DIR = Path("local_state/provider_cache")
PROVIDER_CIRCUIT_STATE_PATH = PROVIDER_CACHE_DIR / "circuit_breaker_state.json"
PROVIDER_CACHE_RETENTION_DAYS = 7
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
CIRCUIT_BREAKER_OPEN_SECONDS = 10 * 60

_CIRCUIT_BREAKER_STATE: dict[str, dict[str, float]] = {}
_CIRCUIT_BREAKER_LOCK = Lock()
_CIRCUIT_BREAKER_LOADED = False

IRRADIANCE_HOURLY_MAX_WM2 = 1400.0
IRRADIANCE_HOURLY_EXTREME_WM2 = 2500.0
IRRADIANCE_DAILY_CLEARSKY_FACTOR = 1.35
MAX_PROVIDER_RESPONSE_CHARS = 250_000


@dataclass
class EnsembleWeatherResult:
    weather_primary: core.ForecastResult
    pv_ensemble_p50: pd.Series
    pv_ensemble_unclipped_p50: pd.Series
    pv_ensemble_clipped_p50: pd.Series
    pv_ensemble_east_p50: pd.Series
    pv_ensemble_south_p50: pd.Series
    pv_ensemble_p10: pd.Series | None
    pv_ensemble_p90: pd.Series | None
    per_model_pv_totals_kwh: dict[str, float]
    missing_vars_by_model: dict[str, list[str]]
    derived_irradiance_by_model: dict[str, bool]
    failed_models: list[str]
    failed_model_reasons: dict[str, dict[str, Any]]
    model_live_failed_used_cached: dict[str, bool]
    selected_models: list[str]
    weights_used: dict[str, float] | None
    weather_primary_model_id: str
    weather_by_model: dict[str, core.ForecastResult]
    pv_by_model: dict[str, pd.DataFrame]
    weather_ensemble_table: core.ForecastResult
    provider_payloads_by_model: dict[str, dict[str, Any]]


class WeatherProviderError(RuntimeError):
    def __init__(
        self,
        *,
        category: str,
        message: str,
        status: int | None = None,
        provider_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status = status
        self.message = message
        self.provider_reason = provider_reason

    def to_reason(self) -> dict[str, Any]:
        reason = {
            "category": self.category,
            "status": self.status,
            "message": self.message,
        }
        if self.provider_reason is not None:
            reason["provider_reason"] = self.provider_reason
        return reason


def check_irradiance_sanity(
    df: pd.DataFrame,
    model_id: str,
    *,
    loc: core.Location | None = None,
    tz: str | None = None,
) -> list[str]:
    warnings: list[str] = []
    if df.empty or "ghi_wm2" not in df.columns:
        return warnings

    ghi = pd.to_numeric(df.get("ghi_wm2"), errors="coerce")
    if ghi.empty or ghi.notna().sum() == 0:
        return warnings

    max_ghi = float(ghi.max(skipna=True)) if ghi.notna().any() else 0.0
    sustained_mask = ghi > IRRADIANCE_HOURLY_MAX_WM2
    sustained_count = int(sustained_mask.sum())
    extreme_count = int((ghi > IRRADIANCE_HOURLY_EXTREME_WM2).sum())
    sustained_hours = sorted(str(ts) for ts in ghi.index[sustained_mask][:5])

    if sustained_count > 0:
        msg = (
            f"irradiance anomaly model={model_id}: hourly_ghi_exceeds={IRRADIANCE_HOURLY_MAX_WM2:.0f}W/m² "
            f"count={sustained_count} extreme_count={extreme_count} max_ghi={max_ghi:.1f} "
            f"sample_hours={sustained_hours}"
        )
        warnings.append(msg)

    tz_use = tz or (str(df.index.tz) if getattr(df.index, "tz", None) is not None else core.TIMEZONE)
    loc_use = loc or core.Location("default", core.LATITUDE, core.LONGITUDE)
    if core.PVLIB_AVAILABLE and getattr(df.index, "tz", None) is not None:
        try:
            import pvlib  # type: ignore

            pvloc = pvlib.location.Location(latitude=loc_use.latitude, longitude=loc_use.longitude, tz=tz_use)
            cs = pvloc.get_clearsky(df.index.tz_convert(tz_use), model="ineichen")
            clear_ghi = pd.to_numeric(cs.get("ghi"), errors="coerce").reindex(df.index).fillna(0.0).clip(lower=0.0)
            measured_wh = float(ghi.fillna(0.0).clip(lower=0.0).sum())
            clear_wh = float(clear_ghi.sum())
            ratio = measured_wh / max(clear_wh, 1.0)
            if measured_wh > (clear_wh * IRRADIANCE_DAILY_CLEARSKY_FACTOR):
                warnings.append(
                    "irradiance anomaly model="
                    f"{model_id}: daily_ghi_integral_whm2={measured_wh:.1f} clear_sky_whm2={clear_wh:.1f} "
                    f"ratio={ratio:.2f} limit={IRRADIANCE_DAILY_CLEARSKY_FACTOR:.2f} "
                    f"date={df.index[0].date().isoformat()} lat={loc_use.latitude:.4f} lon={loc_use.longitude:.4f}"
                )
        except Exception as exc:
            _LOGGER.debug("[weather_ensemble] check_irradiance_sanity skipped model=%s err=%s", model_id, exc)

    return warnings




def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _to_utc_iso(ts: dt.datetime | None = None) -> str:
    use = ts or _utc_now()
    if use.tzinfo is None:
        use = use.replace(tzinfo=dt.timezone.utc)
    return use.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()


def _parse_utc_timestamp(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return str(value)


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _load_circuit_breaker_state() -> None:
    global _CIRCUIT_BREAKER_LOADED
    with _CIRCUIT_BREAKER_LOCK:
        if _CIRCUIT_BREAKER_LOADED:
            return
        payload = _read_json_file(PROVIDER_CIRCUIT_STATE_PATH, default={})
        if isinstance(payload, dict):
            _CIRCUIT_BREAKER_STATE.clear()
            for model_id, state in payload.items():
                if not isinstance(state, dict):
                    continue
                _CIRCUIT_BREAKER_STATE[str(model_id)] = {
                    "consecutive_failures": float(state.get("consecutive_failures", 0) or 0),
                    "last_failure_ts": float(state.get("last_failure_ts", 0) or 0),
                    "circuit_open_until_ts": float(state.get("circuit_open_until_ts", 0) or 0),
                }
        _CIRCUIT_BREAKER_LOADED = True


def _persist_circuit_breaker_state() -> None:
    with _CIRCUIT_BREAKER_LOCK:
        payload = {
            model_id: {
                "consecutive_failures": int(state.get("consecutive_failures", 0) or 0),
                "last_failure_ts": float(state.get("last_failure_ts", 0) or 0),
                "circuit_open_until_ts": float(state.get("circuit_open_until_ts", 0) or 0),
            }
            for model_id, state in _CIRCUIT_BREAKER_STATE.items()
        }
    _write_json_file(PROVIDER_CIRCUIT_STATE_PATH, payload)


def _is_circuit_open(model_id: str) -> tuple[bool, int]:
    _load_circuit_breaker_state()
    now_ts = time.time()
    with _CIRCUIT_BREAKER_LOCK:
        state = _CIRCUIT_BREAKER_STATE.get(model_id, {})
        open_until = float(state.get("circuit_open_until_ts", 0) or 0)
    if open_until <= now_ts:
        return False, 0
    return True, max(int(open_until - now_ts), 1)


def _mark_provider_success(model_id: str) -> None:
    _load_circuit_breaker_state()
    changed = False
    with _CIRCUIT_BREAKER_LOCK:
        state = _CIRCUIT_BREAKER_STATE.get(model_id)
        if not isinstance(state, dict):
            return
        if int(state.get("consecutive_failures", 0) or 0) != 0 or float(state.get("circuit_open_until_ts", 0) or 0) != 0.0:
            state["consecutive_failures"] = 0
            state["circuit_open_until_ts"] = 0.0
            _CIRCUIT_BREAKER_STATE[model_id] = state
            changed = True
    if changed:
        _persist_circuit_breaker_state()


def _mark_provider_failure(model_id: str) -> dict[str, Any]:
    _load_circuit_breaker_state()
    now_ts = time.time()
    with _CIRCUIT_BREAKER_LOCK:
        state = _CIRCUIT_BREAKER_STATE.setdefault(model_id, {
            "consecutive_failures": 0.0,
            "last_failure_ts": 0.0,
            "circuit_open_until_ts": 0.0,
        })
        failures = int(state.get("consecutive_failures", 0) or 0) + 1
        state["consecutive_failures"] = float(failures)
        state["last_failure_ts"] = now_ts
        if failures >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            state["circuit_open_until_ts"] = now_ts + float(CIRCUIT_BREAKER_OPEN_SECONDS)
        open_until = float(state.get("circuit_open_until_ts", 0) or 0)
        _CIRCUIT_BREAKER_STATE[model_id] = state
    _persist_circuit_breaker_state()
    return {
        "consecutive_failures": failures,
        "circuit_open_until_ts": open_until,
        "circuit_open_for_seconds": max(int(open_until - now_ts), 0),
    }


def _provider_cache_path(model_id: str, target_date: dt.date, run_hour: int) -> Path:
    return PROVIDER_CACHE_DIR / f"{model_id}__{target_date.isoformat()}__{int(run_hour):02d}.json"


def _cleanup_provider_cache(now: dt.datetime | None = None) -> None:
    base = PROVIDER_CACHE_DIR
    if not base.exists():
        return
    now_utc = (now or _utc_now()).astimezone(dt.timezone.utc)
    cutoff = now_utc - dt.timedelta(days=PROVIDER_CACHE_RETENTION_DAYS)
    for path in base.glob("*.json"):
        if path == PROVIDER_CIRCUIT_STATE_PATH:
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
            if mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _store_provider_cache(model_id: str, target_date: dt.date, run_hour: int, data: dict[str, Any]) -> None:
    PROVIDER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "target_date": target_date.isoformat(),
        "run_hour": int(run_hour),
        "cached_at_utc": _to_utc_iso(),
        "weather_payload": data,
    }
    _write_json_file(_provider_cache_path(model_id, target_date, run_hour), payload)


def _load_provider_cache(model_id: str, target_date: dt.date, run_hour: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for hour in range(int(run_hour), -1, -1):
        path = _provider_cache_path(model_id, target_date, hour)
        payload = _read_json_file(path, default=None)
        if not isinstance(payload, dict):
            continue
        weather_payload = payload.get("weather_payload")
        if isinstance(weather_payload, dict):
            meta = {
                "model_id": model_id,
                "target_date": target_date.isoformat(),
                "requested_run_hour": int(run_hour),
                "cache_run_hour": int(payload.get("run_hour", hour) or hour),
                "cached_at_utc": str(payload.get("cached_at_utc") or ""),
            }
            return weather_payload, meta
    return None, None

def _safe_provider_reason(resp: Any) -> str:
    payload = None
    if hasattr(resp, "json"):
        try:
            payload = resp.json()
        except (ValueError, AttributeError, TypeError):
            payload = None

    reason = ""
    if isinstance(payload, dict) and payload.get("reason"):
        reason = str(payload.get("reason"))
    else:
        reason = (getattr(resp, "text", "") or "").strip()
    return reason[:200]


def _should_retry_with_forecast(exc: WeatherProviderError, fallback_model: str | None, requested_model: str) -> bool:
    if exc.category != "http_error":
        return False
    if not fallback_model or fallback_model == requested_model:
        return False
    if exc.status == 404:
        return True
    if exc.status != 400:
        return False

    reason = (exc.provider_reason or "").lower().strip()
    if not reason:
        return False

    allow = any(
        s in reason
        for s in [
            "model",
            "models",
            "not available",
            "unknown model",
            "unsupported",
            "not supported",
            "not in allowed",
            "invalid model",
            "variable",
            "variables",
            "hourly parameter",
            "daily parameter",
        ]
    )
    block = any(
        s in reason
        for s in [
            "latitude",
            "longitude",
            "timezone",
            "start_date",
            "end_date",
            "timeformat",
        ]
    )
    return allow and not block


def weather_models_payload() -> list[dict[str, Any]]:
    rows = []
    for model_id, spec in WEATHER_MODELS.items():
        rows.append(
            {
                "id": model_id,
                "label": spec["label"],
                "endpoint": spec["endpoint"],
                "badges": spec.get("badges", []),
                "recommended_for_be": bool(spec.get("recommended_for_be")),
                "capability": spec.get("capability", {}),
                "notes": spec.get("capability", {}).get("notes", ""),
            }
        )
    return rows


def _trim_text(raw: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(raw) <= max_chars:
        return raw
    suffix = f"... [truncated {len(raw) - max_chars} chars]"
    keep = max(0, max_chars - len(suffix))
    return raw[:keep] + suffix


def _serialize_bounded_json(payload: Any, max_chars: int = MAX_PROVIDER_RESPONSE_CHARS) -> str | None:
    try:
        serialized = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        return None
    if len(serialized) <= max_chars:
        return serialized
    fallback = {
        "_truncated": True,
        "original_length": len(serialized),
        "preview": _trim_text(serialized, max_chars=max(0, max_chars - 128)),
    }
    try:
        return json.dumps(fallback, sort_keys=True)
    except Exception:
        return None


def _request_open_meteo(url: str, params: dict[str, Any], model_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    global _SESSION
    if _SESSION is None:
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"GET"},
            backoff_factor=0.5,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _SESSION = session
    response = None
    request_started = time.perf_counter()
    try:
        response = _SESSION.get(url, params=params, timeout=(5, 30))
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        category = "rate_limited" if status == 429 else "provider_down" if status in {500, 502, 503, 504} else "http_error"
        provider_reason = _safe_provider_reason(exc.response) if exc.response is not None else ""
        reason_suffix = f" reason={provider_reason}" if provider_reason else ""
        raise WeatherProviderError(
            category=category,
            status=status,
            provider_reason=provider_reason,
            message=f"Open-Meteo request failed ({category}) for {model_id} status={status}{reason_suffix}",
        ) from exc
    except requests.Timeout as exc:
        raise WeatherProviderError(
            category="timeout",
            status=None,
            message=f"Open-Meteo request timeout for {model_id}",
        ) from exc
    except requests.RequestException as exc:
        raise WeatherProviderError(
            category="network_error",
            status=None,
            message=f"Open-Meteo network error for {model_id}: {exc}",
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise WeatherProviderError(
            category="malformed_json",
            status=response.status_code,
            message=f"Malformed JSON from Open-Meteo for {model_id}",
        ) from exc
    if not isinstance(data, dict):
        raise WeatherProviderError(
            category="invalid_payload",
            status=response.status_code,
            message=f"Unexpected weather payload shape for {model_id}",
        )

    response_meta = {
        "http_status": int(response.status_code),
        "latency_ms": int((time.perf_counter() - request_started) * 1000),
        "response_headers": dict(response.headers),
        "response_json": _serialize_bounded_json(data),
    }
    return data, response_meta


def _params_hash(params: dict[str, Any]) -> str:
    safe_payload = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(safe_payload.encode("utf-8")).hexdigest()[:16]


def _log_model_fetch(
    *,
    model_id: str,
    endpoint: str,
    params: dict[str, Any],
    elapsed_ms: int,
    category: str,
    status: int | None,
    outcome: str,
) -> None:
    _LOGGER.info(
        "[weather_ensemble] model_fetch model=%s endpoint=%s params_hash=%s status=%s elapsed_ms=%s category=%s outcome=%s",
        model_id,
        endpoint,
        _params_hash(params),
        status,
        elapsed_ms,
        category,
        outcome,
    )


def _decompose_from_ghi(df: pd.DataFrame, loc: core.Location, tz: str) -> pd.DataFrame:
    if not core.PVLIB_AVAILABLE:
        return df
    import pvlib  # type: ignore

    pvloc = pvlib.location.Location(latitude=loc.latitude, longitude=loc.longitude, tz=tz)
    solpos = pvloc.get_solarposition(df.index)
    ghi = pd.to_numeric(df["ghi_wm2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    disc = pvlib.irradiance.disc(ghi, solpos["apparent_zenith"], df.index)
    dni = pd.to_numeric(disc["dni"], errors="coerce").fillna(0.0).clip(lower=0.0)
    cos_zen = pd.to_numeric(solpos["apparent_zenith"], errors="coerce").apply(lambda z: max(0.0, math.cos(math.radians(z))) if pd.notna(z) else 0.0)
    dhi = (ghi - (dni * cos_zen)).fillna(0.0).clip(lower=0.0)

    dni_existing = pd.to_numeric(df.get("dni_wm2"), errors="coerce") if "dni_wm2" in df.columns else pd.Series(np.nan, index=df.index)
    dhi_existing = pd.to_numeric(df.get("dhi_wm2"), errors="coerce") if "dhi_wm2" in df.columns else pd.Series(np.nan, index=df.index)

    df["dni_wm2"] = dni_existing.where(dni_existing.notna(), dni)
    df["dhi_wm2"] = dhi_existing.where(dhi_existing.notna(), dhi)
    return df


def _cache_key(model_id: str, lat: float, lon: float, tz: str, target_date: dt.date) -> tuple:
    return (model_id, round(float(lat), 4), round(float(lon), 4), str(tz), target_date.isoformat())




def _is_central_europe(lat: float, lon: float) -> bool:
    return 43.0 <= float(lat) <= 57.5 and -2.0 <= float(lon) <= 20.0


def _aggregate_minutely_15_to_hourly(minutely_payload: dict[str, Any], tz: str) -> pd.DataFrame:
    times = pd.to_datetime(minutely_payload.get("time", []), errors="coerce")
    if len(times) == 0:
        return pd.DataFrame()
    if getattr(times, "tz", None) is None:
        times = times.tz_localize(tz)
    else:
        times = times.tz_convert(tz)

    hourly = pd.DataFrame(index=times)
    for src, dst in {
        "shortwave_radiation": "ghi_wm2",
        "direct_normal_irradiance": "dni_wm2",
        "diffuse_radiation": "dhi_wm2",
    }.items():
        vals = minutely_payload.get(src)
        if vals is None:
            continue
        hourly[dst] = pd.to_numeric(pd.Series(vals, index=times), errors="coerce")

    if hourly.empty:
        return hourly
    return hourly.groupby(hourly.index.floor("h")).mean(numeric_only=True)

def _finalize_irradiance_components(
    *,
    ghi: pd.Series,
    dni: pd.Series,
    dhi: pd.Series,
    loc: core.Location,
    tz: str,
    missing_vars: list[str],
) -> tuple[pd.Series, pd.Series, list[str], bool]:
    dni_out = pd.to_numeric(dni, errors="coerce")
    dhi_out = pd.to_numeric(dhi, errors="coerce")
    ghi_out = pd.to_numeric(ghi, errors="coerce")

    derived_irradiance = False
    needs_derivation = bool(dni_out.isna().any() or dhi_out.isna().any())
    ghi_usable = bool(((ghi_out.notna()) & (ghi_out > 0)).any())

    if needs_derivation and ghi_usable:
        irradiance_df = pd.DataFrame(
            {
                "ghi_wm2": ghi_out,
                "dni_wm2": dni_out,
                "dhi_wm2": dhi_out,
            },
            index=ghi_out.index,
        )
        irradiance_df = _decompose_from_ghi(irradiance_df, loc, tz)
        dni_out = pd.to_numeric(irradiance_df["dni_wm2"], errors="coerce")
        dhi_out = pd.to_numeric(irradiance_df["dhi_wm2"], errors="coerce")
        derived_irradiance = True

    missing_set = set(missing_vars)
    irradiance_fields = {
        "direct_normal_irradiance": dni_out,
        "diffuse_radiation": dhi_out,
    }
    for field, values in irradiance_fields.items():
        if values.isna().all():
            missing_set.add(field)
        else:
            missing_set.discard(field)

    return dni_out, dhi_out, sorted(missing_set), derived_irradiance


def fetch_open_meteo_weather(
    model_id: str,
    loc: core.Location,
    tz: str,
    target_date: dt.date,
    *,
    accuracy_mode: bool = True,
    fast_mode: bool = False,
) -> tuple[core.ForecastResult, list[str], bool, dict[str, Any]]:
    if model_id not in WEATHER_MODELS:
        raise RuntimeError(f"Unsupported weather model: {model_id}")

    key = (_cache_key(model_id, loc.latitude, loc.longitude, tz, target_date), bool(accuracy_mode), bool(fast_mode))
    now = time.time()
    cached = _WEATHER_CACHE.get(key)
    if cached and now - cached[0] < _WEATHER_CACHE_TTL_S:
        return cached[1], list(cached[2]), bool(cached[3]), {"source": "in_memory_cache"}

    spec = WEATHER_MODELS[model_id]
    hourly_variables = BASE_HOURLY_VARIABLES[:] + IRRADIANCE_HOURLY_VARIABLES

    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "timezone": tz,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timeformat": "iso8601",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "hourly": ",".join(hourly_variables),
        "daily": "sunrise,sunset",
    }
    params.update(spec.get("params", {}))

    use_icon15 = (
        model_id == "dwd_icon_d2"
        and bool(accuracy_mode)
        and not bool(fast_mode)
        and _is_central_europe(loc.latitude, loc.longitude)
    )
    if use_icon15:
        params["minutely_15"] = ",".join([
            "shortwave_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
        ])

    _cleanup_provider_cache()
    run_hour = dt.datetime.now(dt.timezone.utc).hour
    circuit_open, open_for_seconds = _is_circuit_open(model_id)
    fetch_meta: dict[str, Any] = {
        "source": "live",
        "live_failed_used_cached": False,
        "run_hour": int(run_hour),
        "circuit_breaker_open": bool(circuit_open),
        "circuit_breaker_open_for_seconds": int(open_for_seconds),
    }
    request_params = dict(params)
    request_endpoint = str(spec["endpoint"])
    response_meta: dict[str, Any] | None = None

    request_start = time.perf_counter()
    try:
        if circuit_open:
            raise WeatherProviderError(
                category="circuit_open",
                status=None,
                message=f"Circuit breaker open for {model_id}; skip live fetch for {open_for_seconds}s",
            )
        data, response_meta = _request_open_meteo(spec["endpoint"], params, model_id=model_id)
        _log_model_fetch(
            model_id=model_id,
            endpoint=spec["endpoint"],
            params=params,
            elapsed_ms=int((time.perf_counter() - request_start) * 1000),
            category="ok",
            status=200,
            outcome="success",
        )
    except WeatherProviderError as exc:
        _log_model_fetch(
            model_id=model_id,
            endpoint=spec["endpoint"],
            params=params,
            elapsed_ms=int((time.perf_counter() - request_start) * 1000),
            category=exc.category,
            status=exc.status,
            outcome="failed",
        )
        final_exc: WeatherProviderError = exc
        fallback_model = FORECAST_FALLBACK_MODELS.get(model_id)
        requested_model = str(params.get("models") or "").strip()
        if (not circuit_open) and _should_retry_with_forecast(exc, fallback_model, requested_model):
            fallback_params = dict(params)
            fallback_params["hourly"] = ",".join(BASE_HOURLY_VARIABLES + IRRADIANCE_HOURLY_VARIABLES)
            fallback_params["models"] = fallback_model
            fallback_endpoint = "https://api.open-meteo.com/v1/forecast"
            fallback_start = time.perf_counter()
            try:
                data, response_meta = _request_open_meteo(fallback_endpoint, fallback_params, model_id=model_id)
                _log_model_fetch(
                    model_id=model_id,
                    endpoint=fallback_endpoint,
                    params=fallback_params,
                    elapsed_ms=int((time.perf_counter() - fallback_start) * 1000),
                    category="ok",
                    status=200,
                    outcome="success",
                )
                fetch_meta["source"] = "forecast_fallback"
                request_params = dict(fallback_params)
                request_endpoint = str(fallback_endpoint)
            except WeatherProviderError as fallback_exc:
                _log_model_fetch(
                    model_id=model_id,
                    endpoint=fallback_endpoint,
                    params=fallback_params,
                    elapsed_ms=int((time.perf_counter() - fallback_start) * 1000),
                    category=fallback_exc.category,
                    status=fallback_exc.status,
                    outcome="failed",
                )
                final_exc = fallback_exc
                data = None
        else:
            data = None

        if data is None:
            failure_state = _mark_provider_failure(model_id)
            cached_payload, cache_meta = _load_provider_cache(model_id, target_date, run_hour)
            if isinstance(cached_payload, dict):
                data = cached_payload
                fetch_meta.update({
                    "source": "provider_cache",
                    "live_failed_used_cached": True,
                    "cache_meta": cache_meta,
                    "cache_hit": True,
                    "failure_state": failure_state,
                    "live_failure": final_exc.to_reason(),
                })
            else:
                setattr(final_exc, "fetch_meta", {
                    **fetch_meta,
                    "provider_payload": {
                        "fetched_at_utc": _to_utc_iso(),
                        "endpoint": request_endpoint,
                        "params": request_params,
                        "http_status": final_exc.status,
                        "latency_ms": None,
                        "response_headers": None,
                        "response_json": None,
                        "source": fetch_meta.get("source"),
                    },
                })
                raise final_exc

    if fetch_meta.get("source") == "live" or fetch_meta.get("source") == "forecast_fallback":
        _mark_provider_success(model_id)
        _store_provider_cache(model_id, target_date, run_hour, data)

    provider_payload = {
        "fetched_at_utc": _to_utc_iso(),
        "endpoint": request_endpoint,
        "params": request_params,
        "http_status": response_meta.get("http_status") if isinstance(response_meta, dict) else None,
        "latency_ms": response_meta.get("latency_ms") if isinstance(response_meta, dict) else None,
        "response_headers": response_meta.get("response_headers") if isinstance(response_meta, dict) else None,
        "response_json": response_meta.get("response_json") if isinstance(response_meta, dict) else _serialize_bounded_json(data),
        "source": fetch_meta.get("source"),
    }
    fetch_meta["provider_payload"] = provider_payload

    hourly = data.get("hourly") if isinstance(data.get("hourly"), dict) else {}
    times = pd.to_datetime(hourly.get("time", []), errors="coerce")
    if len(times) == 0:
        raise RuntimeError(f"No hourly weather data for {model_id}")
    if getattr(times, "tz", None) is None:
        times = times.tz_localize(tz)
    else:
        times = times.tz_convert(tz)

    missing_vars: list[str] = []

    def _series(name: str, default: float = 0.0, *, record_missing: bool = True) -> pd.Series:
        vals = hourly.get(name)
        if vals is None:
            if record_missing:
                missing_vars.append(name)
            vals = [default] * len(times)
        return pd.to_numeric(pd.Series(vals, index=times), errors="coerce")

    df = pd.DataFrame(index=times)
    df["temp_air_c"] = _series("temperature_2m", 10.0).ffill().bfill().fillna(10.0)
    df["wind_speed_ms"] = _series("wind_speed_10m", 1.0).fillna(1.0).clip(lower=0.0)
    df["cloud_cover_pct"] = _series("cloud_cover", 0.0).fillna(0.0).clip(lower=0.0)
    df["ghi_wm2"] = _series("shortwave_radiation", 0.0).fillna(0.0).clip(lower=0.0)

    minutely = data.get("minutely_15") if isinstance(data.get("minutely_15"), dict) else {}
    if use_icon15 and minutely:
        agg15 = _aggregate_minutely_15_to_hourly(minutely, tz=tz)
        if not agg15.empty:
            agg15 = agg15.reindex(df.index)
            for col in ["ghi_wm2", "dni_wm2", "dhi_wm2"]:
                if col in agg15.columns:
                    df[col] = pd.to_numeric(agg15[col], errors="coerce")

    dni = df["dni_wm2"] if "dni_wm2" in df.columns else _series("direct_normal_irradiance", np.nan, record_missing=False)
    dhi = df["dhi_wm2"] if "dhi_wm2" in df.columns else _series("diffuse_radiation", np.nan, record_missing=False)
    dni_final, dhi_final, missing_vars, derived_irradiance = _finalize_irradiance_components(
        ghi=df["ghi_wm2"],
        dni=dni,
        dhi=dhi,
        loc=loc,
        tz=tz,
        missing_vars=missing_vars,
    )
    df["dni_wm2"] = dni_final
    df["dhi_wm2"] = dhi_final

    if not df["dni_wm2"].isna().all() or not df["dhi_wm2"].isna().all():
        missing_vars = [
            v
            for v in missing_vars
            if v not in ("direct_normal_irradiance", "diffuse_radiation")
        ]

    availability = pd.DataFrame(index=df.index)
    for col in ["ghi_wm2", "dni_wm2", "dhi_wm2"]:
        availability[col] = pd.to_numeric(df.get(col), errors="coerce").notna()

    df["dni_wm2"] = pd.to_numeric(df["dni_wm2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["dhi_wm2"] = pd.to_numeric(df["dhi_wm2"], errors="coerce").fillna(0.0).clip(lower=0.0)

    df = core.normalize_hourly_forecast_index(
        df[["temp_air_c", "ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_cover_pct", "wind_speed_ms"]],
        target_date,
        tz,
    )
    availability = availability.reindex(df.index).fillna(False).astype(bool)
    for col in ["ghi_wm2", "dni_wm2", "dhi_wm2"]:
        df.loc[~availability[col], col] = np.nan

    def _alias(dst: str, src: str) -> None:
        if src in df.columns:
            df[dst] = df[src]
        else:
            df[dst] = np.nan

    _alias("shortwave_radiation", "ghi_wm2")
    _alias("direct_normal_irradiance", "dni_wm2")
    _alias("diffuse_radiation", "dhi_wm2")
    _alias("temperature_2m", "temp_air_c")
    _alias("wind_speed_10m", "wind_speed_ms")
    _alias("cloud_cover", "cloud_cover_pct")

    daily = data.get("daily") if isinstance(data.get("daily"), dict) else {}
    sunrise = pd.to_datetime((daily.get("sunrise") or [None])[0], errors="coerce")
    sunset = pd.to_datetime((daily.get("sunset") or [None])[0], errors="coerce")
    if pd.isna(sunrise) or pd.isna(sunset):
        if core.PVLIB_AVAILABLE:
            import pvlib  # type: ignore

            pvloc = pvlib.location.Location(latitude=loc.latitude, longitude=loc.longitude, tz=tz)
            date_index = pd.DatetimeIndex([pd.Timestamp(dt.datetime.combine(target_date, dt.time(12, 0)), tz=tz)])
            sun_times = pvloc.get_sun_rise_set_transit(date_index)
            sunrise = pd.to_datetime(sun_times["sunrise"].iloc[0], errors="coerce")
            sunset = pd.to_datetime(sun_times["sunset"].iloc[0], errors="coerce")
        if pd.isna(sunrise) or pd.isna(sunset):
            day_start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(6, 0)), tz=tz)
            sunrise = day_start
            sunset = day_start + dt.timedelta(hours=12)
    if sunrise.tzinfo is None:
        sunrise = sunrise.tz_localize(tz)
    else:
        sunrise = sunrise.tz_convert(tz)
    if sunset.tzinfo is None:
        sunset = sunset.tz_localize(tz)
    else:
        sunset = sunset.tz_convert(tz)

    forecast = core.ForecastResult(df=df, sunrise=sunrise.to_pydatetime(), sunset=sunset.to_pydatetime())
    _WEATHER_CACHE[key] = (time.time(), forecast, list(set(missing_vars)), bool(derived_irradiance))
    return forecast, list(set(missing_vars)), bool(derived_irradiance), fetch_meta


def _dynamic_weight_settings() -> dict[str, Any]:
    cfg = core.get_effective_config()
    weather_cfg = cfg.get("weather") if isinstance(cfg, dict) else {}
    if not isinstance(weather_cfg, dict):
        return {}
    dynamic_cfg = weather_cfg.get("dynamic_weights")
    return dynamic_cfg if isinstance(dynamic_cfg, dict) else {}


def _load_dynamic_weights(selected_models: list[str]) -> dict[str, float] | None:
    settings = _dynamic_weight_settings()
    if not bool(settings.get("enabled", False)):
        return None

    lookback_days = max(int(settings.get("lookback_days", 30)), 1)
    min_days = max(int(settings.get("min_days", 10)), 1)
    epsilon = 1e-6
    db_path = str(settings.get("db_path") or "local_state/planner_history.sqlite")

    try:
        model_stats = db_sqlite.fetch_recent_model_mae_scores(db_path, lookback_days=lookback_days)
    except Exception as exc:
        _LOGGER.debug("[weather_ensemble] dynamic_weights disabled err=%s", exc)
        return None

    mae_by_model: dict[str, float] = {}
    for model_id in sorted(selected_models):
        stats = model_stats.get(model_id) if isinstance(model_stats, dict) else None
        if not isinstance(stats, dict):
            continue
        days = float(stats.get("days") or 0.0)
        mae = stats.get("pv_mae_kwh")
        if days < min_days:
            continue
        try:
            mae_f = float(mae)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(mae_f) or mae_f < 0.0:
            continue
        mae_by_model[model_id] = mae_f

    if not mae_by_model:
        return None

    inv = {model_id: 1.0 / (mae + epsilon) for model_id, mae in mae_by_model.items()}
    total = float(sum(inv.values()))
    if total <= 0.0:
        return None
    return {model_id: weight / total for model_id, weight in inv.items()}


def _weighted_ensemble(
    series_map: dict[str, pd.Series],
    selected_models: list[str],
    dynamic_weights: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, float] | None]:
    weighted_subset = dict(dynamic_weights or {})
    if not weighted_subset:
        weighted_subset = {m: DEFAULT_WEIGHTED_BELGIUM[m] for m in selected_models if m in DEFAULT_WEIGHTED_BELGIUM}
    if not weighted_subset:
        return pd.concat(series_map.values(), axis=1).mean(axis=1), None
    total = sum(weighted_subset.values())
    if total <= 0:
        return pd.concat(series_map.values(), axis=1).mean(axis=1), None
    normalized = {m: w / total for m, w in weighted_subset.items()}
    matrix = pd.DataFrame({m: series_map[m] for m in normalized if m in series_map})
    if matrix.empty:
        return pd.concat(series_map.values(), axis=1).mean(axis=1), None
    weighted_values = matrix.mul(pd.Series(normalized), axis=1)
    numerator = weighted_values.sum(axis=1, skipna=True)
    denominator = matrix.notna().mul(pd.Series(normalized), axis=1).sum(axis=1)
    out = numerator.div(denominator.where(denominator > 0))
    return out.astype(float), {m: normalized[m] for m in matrix.columns}


def build_weather_ensemble_table(
    weather_ok: dict[str, core.ForecastResult],
    index: pd.DatetimeIndex,
    ensemble_method: str,
    weights: dict[str, float] | None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    normalized_method = str(ensemble_method).lower().strip()
    for var in WEATHER_DISPLAY_VARS:
        series_by_model: dict[str, pd.Series] = {}
        for model_id, forecast in weather_ok.items():
            series = forecast.df.get(var)
            if series is None:
                continue
            series_by_model[model_id] = pd.to_numeric(series, errors="coerce").reindex(index)

        if not series_by_model:
            out[var] = pd.Series(np.nan, index=index, dtype=float)
            out[f"{var}_min"] = pd.Series(np.nan, index=index, dtype=float)
            out[f"{var}_max"] = pd.Series(np.nan, index=index, dtype=float)
            continue

        matrix = pd.DataFrame(series_by_model, index=index)
        out[f"{var}_min"] = matrix.min(axis=1, skipna=True)
        out[f"{var}_max"] = matrix.max(axis=1, skipna=True)

        if normalized_method == "median":
            out[var] = matrix.median(axis=1, skipna=True)
            continue
        if normalized_method == "mean":
            out[var] = matrix.mean(axis=1, skipna=True)
            continue

        weight_map = dict(weights or {})
        weighted_columns = [model_id for model_id in matrix.columns if model_id in weight_map]
        if not weighted_columns:
            out[var] = matrix.mean(axis=1, skipna=True)
            continue
        weighted_matrix = matrix[weighted_columns]
        weight_series = pd.Series({model_id: float(weight_map[model_id]) for model_id in weighted_columns}, dtype=float)
        weighted_values = weighted_matrix.mul(weight_series, axis=1)
        numerator = weighted_values.sum(axis=1, skipna=True)
        denominator = weighted_matrix.notna().mul(weight_series, axis=1).sum(axis=1)
        out[var] = numerator.div(denominator.where(denominator > 0))

    return out.reindex(index)


def build_ensemble_forecast(
    loc: core.Location,
    target_date: dt.date,
    tz: str,
    weather_models: list[str] | None,
    ensemble_method: str,
    pv_uncertainty: bool,
    accuracy_mode: bool = True,
    fast_mode: bool = False,
) -> EnsembleWeatherResult:
    selected = weather_models[:] if weather_models else DEFAULT_ACCURACY_MODELS[:]
    selected = [m for m in selected if m in WEATHER_MODELS]
    if fast_mode:
        if weather_models:
            selected = selected[:2]
        else:
            selected = [m for m in DEFAULT_ACCURACY_MODELS if m in WEATHER_MODELS][:2]
    if not selected:
        raise RuntimeError("Select at least one weather model.")

    per_model_pv_columns: dict[str, dict[str, pd.Series]] = {
        "pv_total_kwh": {},
        "pv_total_unclipped_kwh": {},
        "pv_east_kwh": {},
        "pv_south_kwh": {},
        "pv_clipped_kwh": {},
    }
    per_model_pv_totals: dict[str, float] = {}
    missing_vars_by_model: dict[str, list[str]] = {}
    derived_irradiance_by_model: dict[str, bool] = {}
    failed_models: list[str] = []
    failed_model_reasons: dict[str, dict[str, Any]] = {}
    model_live_failed_used_cached: dict[str, bool] = {}
    weather_ok: dict[str, core.ForecastResult] = {}
    pv_by_model: dict[str, pd.DataFrame] = {}
    provider_payloads_by_model: dict[str, dict[str, Any]] = {}

    canonical_index = pd.date_range(
        pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=tz),
        pd.Timestamp(dt.datetime.combine(target_date + dt.timedelta(days=1), dt.time(0, 0)), tz=tz),
        freq="h",
        inclusive="left",
    )

    def _fetch_and_prepare(model_id: str) -> tuple[str, core.ForecastResult, dict[str, pd.Series], float, list[str], bool, float, dict[str, Any]]:
        weather, missing_vars, derived_irradiance, fetch_meta = fetch_open_meteo_weather(
            model_id,
            loc,
            tz,
            target_date,
            accuracy_mode=accuracy_mode,
            fast_mode=fast_mode,
        )
        missing_hours = float(weather.df.reindex(canonical_index).isna().all(axis=1).sum())
        model_weather_df = weather.df.reindex(canonical_index).copy()
        sanity_warnings = check_irradiance_sanity(model_weather_df, model_id, loc=loc, tz=tz)
        if sanity_warnings:
            raise WeatherProviderError(
                category="irradiance_anomaly",
                message=sanity_warnings[0],
                provider_reason=" | ".join(sanity_warnings),
            )
        for irr_col in ["ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_cover_pct"]:
            if irr_col in model_weather_df.columns:
                model_weather_df[irr_col] = pd.to_numeric(model_weather_df[irr_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if "temp_air_c" in model_weather_df.columns:
            model_weather_df["temp_air_c"] = pd.to_numeric(model_weather_df["temp_air_c"], errors="coerce").ffill().bfill().fillna(10.0)
        if "wind_speed_ms" in model_weather_df.columns:
            model_weather_df["wind_speed_ms"] = pd.to_numeric(model_weather_df["wind_speed_ms"], errors="coerce").fillna(1.0).clip(lower=0.0)
        weather = core.ForecastResult(df=model_weather_df, sunrise=weather.sunrise, sunset=weather.sunset)

        model_pv = core.build_pv_forecast(weather.df, loc, tz=tz).reindex(canonical_index)
        for req in ["pv_total_kwh", "pv_total_unclipped_kwh", "pv_east_kwh", "pv_south_kwh", "pv_clipped_kwh"]:
            if req not in model_pv.columns:
                model_pv[req] = np.nan
        pv_total = pd.to_numeric(model_pv["pv_total_kwh"], errors="coerce").clip(lower=0.0)
        pv_unclipped = pd.to_numeric(model_pv["pv_total_unclipped_kwh"], errors="coerce").clip(lower=0.0)
        pv_unclipped = pd.Series(np.maximum(pv_unclipped, pv_total), index=pv_total.index)
        pv_east = pd.to_numeric(model_pv["pv_east_kwh"], errors="coerce").clip(lower=0.0)
        pv_south = pd.to_numeric(model_pv["pv_south_kwh"], errors="coerce").clip(lower=0.0)
        pv_clipped = pd.to_numeric(model_pv["pv_clipped_kwh"], errors="coerce").clip(lower=0.0)
        return model_id, weather, {
            "pv_total_kwh": pv_total,
            "pv_total_unclipped_kwh": pv_unclipped,
            "pv_east_kwh": pv_east,
            "pv_south_kwh": pv_south,
            "pv_clipped_kwh": pv_clipped,
        }, float(pv_total.sum()), missing_vars, bool(derived_irradiance), missing_hours, fetch_meta

    max_workers = min(max(len(selected), 1), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_fetch_and_prepare, model_id): model_id for model_id in selected}
        for fut in as_completed(future_map):
            model_id = future_map[fut]
            try:
                model_id, weather, pv_cols, pv_total_sum, missing_vars, derived_irradiance, missing_hours, fetch_meta = fut.result()
                for col_name, series in pv_cols.items():
                    per_model_pv_columns[col_name][model_id] = series
                pv_by_model[model_id] = pd.DataFrame(pv_cols).reindex(canonical_index)
                per_model_pv_totals[model_id] = pv_total_sum
                missing_vars_by_model[model_id] = missing_vars
                derived_irradiance_by_model[model_id] = bool(derived_irradiance)
                weather_ok[model_id] = weather
                model_live_failed_used_cached[model_id] = bool(fetch_meta.get("live_failed_used_cached", False))
                provider_payload = fetch_meta.get("provider_payload") if isinstance(fetch_meta, dict) else None
                if isinstance(provider_payload, dict):
                    provider_payloads_by_model[model_id] = provider_payload
                if missing_hours > 2:
                    _LOGGER.warning("[weather_ensemble] model=%s missing_hours=%s on canonical index", model_id, int(missing_hours))
            except WeatherProviderError as exc:
                failed_models.append(model_id)
                failed_model_reasons[model_id] = exc.to_reason()
                exc_fetch_meta = getattr(exc, "fetch_meta", None)
                if isinstance(exc_fetch_meta, dict):
                    provider_payload = exc_fetch_meta.get("provider_payload")
                    if isinstance(provider_payload, dict):
                        provider_payloads_by_model[model_id] = provider_payload
                _LOGGER.warning(
                    "[weather_ensemble] model_failed model=%s category=%s status=%s message=%s",
                    model_id,
                    exc.category,
                    exc.status,
                    exc.message,
                )
            except Exception as exc:
                failed_models.append(model_id)
                failed_model_reasons[model_id] = {
                    "category": "unexpected_error",
                    "status": None,
                    "message": str(exc),
                }
                _LOGGER.exception(
                    "[weather_ensemble] model_failed model=%s category=unexpected_error message=%s",
                    model_id,
                    exc,
                )

    if not per_model_pv_columns["pv_total_kwh"]:
        err = RuntimeError("All weather model requests failed.")
        setattr(err, "failed_models", list(failed_models))
        setattr(err, "failed_model_reasons", dict(failed_model_reasons))
        setattr(err, "weights_used", None)
        raise err

    def _ensemble_column(column_name: str) -> tuple[pd.Series, dict[str, float] | None]:
        model_series = per_model_pv_columns[column_name]
        matrix = pd.concat(model_series.values(), axis=1)
        if ensemble_method == "median":
            return matrix.median(axis=1, skipna=True), None
        if ensemble_method == "mean":
            return matrix.mean(axis=1, skipna=True), None
        model_keys = list(model_series.keys())
        dynamic_weights = _load_dynamic_weights(model_keys)
        return _weighted_ensemble(model_series, model_keys, dynamic_weights=dynamic_weights)

    ensemble_ac_p50, weights_used = _ensemble_column("pv_total_kwh")
    ensemble_unclipped_p50, _ = _ensemble_column("pv_total_unclipped_kwh")
    ensemble_east_p50, _ = _ensemble_column("pv_east_kwh")
    ensemble_south_p50, _ = _ensemble_column("pv_south_kwh")

    if len(per_model_pv_columns["pv_total_kwh"]) >= 3 and ensemble_method != "median":
        matrix = pd.concat(per_model_pv_columns["pv_total_kwh"].values(), axis=1)
        spread = (matrix.max(axis=1) - matrix.min(axis=1)).fillna(0.0)
        spread_median = float(spread.median()) if not spread.empty else 0.0
        extreme_mask = spread > max(0.5, 2.0 * spread_median)
        if int(extreme_mask.sum()) >= 3:
            median_ac = matrix.median(axis=1)
            matrix_unclip = pd.concat(per_model_pv_columns["pv_total_unclipped_kwh"].values(), axis=1)
            median_unclip = matrix_unclip.median(axis=1)
            ensemble_ac_p50.loc[extreme_mask] = median_ac.loc[extreme_mask]
            ensemble_unclipped_p50.loc[extreme_mask] = median_unclip.loc[extreme_mask]

    ensemble_ac_p50 = ensemble_ac_p50.reindex(canonical_index)
    ensemble_unclipped_p50 = ensemble_unclipped_p50.reindex(canonical_index)
    ensemble_east_p50 = ensemble_east_p50.reindex(canonical_index)
    ensemble_south_p50 = ensemble_south_p50.reindex(canonical_index)

    ensemble_unclipped_p50 = pd.Series(np.maximum(ensemble_unclipped_p50, ensemble_ac_p50), index=ensemble_ac_p50.index)
    east_south_total = (ensemble_east_p50 + ensemble_south_p50).fillna(0.0)
    rebalance = pd.Series(1.0, index=ensemble_ac_p50.index, dtype=float)
    positive_split = east_south_total > 0
    rebalance.loc[positive_split] = (ensemble_ac_p50.loc[positive_split] / east_south_total.loc[positive_split]).astype(float)
    ensemble_east_p50 = (ensemble_east_p50 * rebalance).fillna(0.0).clip(lower=0.0)
    ensemble_south_p50 = (ensemble_south_p50 * rebalance).fillna(0.0).clip(lower=0.0)
    ensemble_clipped_p50 = (ensemble_unclipped_p50 - ensemble_ac_p50).clip(lower=0.0)

    p10 = None
    p90 = None
    if pv_uncertainty:
        matrix = pd.concat(per_model_pv_columns["pv_total_kwh"].values(), axis=1)
        p10 = matrix.quantile(0.10, axis=1)
        p90 = matrix.quantile(0.90, axis=1)

    primary_model = next((m for m in selected if m in weather_ok), next(iter(weather_ok.keys())))
    weather_index = canonical_index
    ensemble_weather_df = build_weather_ensemble_table(
        weather_ok=weather_ok,
        index=weather_index,
        ensemble_method=ensemble_method,
        weights=weights_used,
    )
    ensemble_weather = core.ForecastResult(
        df=ensemble_weather_df,
        sunrise=weather_ok[primary_model].sunrise,
        sunset=weather_ok[primary_model].sunset,
    )

    return EnsembleWeatherResult(
        weather_primary=weather_ok[primary_model],
        pv_ensemble_p50=ensemble_ac_p50.astype(float),
        pv_ensemble_unclipped_p50=ensemble_unclipped_p50.astype(float),
        pv_ensemble_clipped_p50=ensemble_clipped_p50.astype(float),
        pv_ensemble_east_p50=ensemble_east_p50.astype(float),
        pv_ensemble_south_p50=ensemble_south_p50.astype(float),
        pv_ensemble_p10=p10.astype(float) if p10 is not None else None,
        pv_ensemble_p90=p90.astype(float) if p90 is not None else None,
        per_model_pv_totals_kwh=per_model_pv_totals,
        missing_vars_by_model=missing_vars_by_model,
        derived_irradiance_by_model=derived_irradiance_by_model,
        failed_models=failed_models,
        failed_model_reasons=failed_model_reasons,
        model_live_failed_used_cached=model_live_failed_used_cached,
        selected_models=[m for m in selected if m in per_model_pv_columns["pv_total_kwh"]],
        weights_used=weights_used,
        weather_primary_model_id=primary_model,
        weather_by_model=weather_ok,
        pv_by_model=pv_by_model,
        weather_ensemble_table=ensemble_weather,
        provider_payloads_by_model=provider_payloads_by_model,
    )
