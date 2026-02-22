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
from zoneinfo import ZoneInfo

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
    "gfs": 0.05,
}

WEATHER_MODELS: dict[str, dict[str, Any]] = {
    "knmi_harmonie_arome": {
        "label": "KNMI HARMONIE-AROME",
        "endpoint": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "knmi_harmonie_arome_netherlands"},
        "badges": ["🏅", "📡", "∑"],
        "recommended_for_be": True,
        "max_days": 7,
        "tier": "short",
        "supports_15min_radiation": False,
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
        "badges": ["🏅", "📡", "☀️", "⏱️"],
        "recommended_for_be": True,
        "max_days": 2,
        "tier": "short",
        "supports_15min_radiation": True,
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
        "badges": ["🏅", "🌐", "☀️"],
        "recommended_for_be": True,
        "max_days": 7,
        "tier": "global",
        "supports_15min_radiation": False,
        "capability": {
            "ghi_native": True,
            "direct_native": True,
            "diffuse_native": True,
            "dni_native": True,
            "notes": "Direct/diffuse may be approximated on open-data feeds.",
        },
    },
    "ecmwf_aifs": {
        "label": "ECMWF AIFS 0.25° Single",
        "endpoint": "https://api.open-meteo.com/v1/ecmwf",
        "params": {"models": "ecmwf_aifs"},
        "badges": ["🌐", "🆕", "☀️"],
        "recommended_for_be": True,
        "max_days": 15,
        "tier": "global",
        "supports_15min_radiation": False,
        "capability": {
            "ghi_native": True,
            "direct_native": True,
            "diffuse_native": True,
            "dni_native": True,
            "notes": "ECMWF AIFS Single 0.25° open-data; 6-hourly native steps interpolated by Open-Meteo.",
        },
    },
    "dwd_icon_eu": {
        "label": "DWD ICON-EU",
        "endpoint": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "icon_eu"},
        "badges": ["🇪🇺", "☀️"],
        "recommended_for_be": True,
        "max_days": 7,
        "tier": "medium",
        "supports_15min_radiation": False,
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
        "badges": ["🇪🇺", "∑"],
        "recommended_for_be": True,
        "max_days": 4,
        "tier": "medium",
        "supports_15min_radiation": False,
        "capability": {
            "ghi_native": True,
            "direct_native": False,
            "diffuse_native": True,
            "dni_native": True,
            "notes": "Seamless provider blend; direct/diffuse/DNI may be derived depending on Open-Meteo feed.",
        },
    },
    "gfs": {
        "label": "NOAA GFS",
        "endpoint": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "gfs_seamless"},
        "badges": ["🌐", "☀️", "🗓️"],
        "recommended_for_be": True,
        "max_days": 16,
        "tier": "global",
        "supports_15min_radiation": False,
        "capability": {
            "ghi_native": True,
            "direct_native": True,
            "diffuse_native": True,
            "dni_native": True,
            "notes": "Global fallback with long horizon coverage.",
        },
    },
}

MODEL_CAPS: dict[str, dict[str, Any]] = {
    model_id: {
        "max_days": int(spec.get("max_days", 0) or 0),
        "has_native_dni_dhi": bool(
            (spec.get("capability", {}) or {}).get("direct_native")
            and (spec.get("capability", {}) or {}).get("diffuse_native")
        ),
        "supports_15min_radiation": bool(spec.get("supports_15min_radiation", False)),
        "tier": str(spec.get("tier") or "global"),
    }
    for model_id, spec in WEATHER_MODELS.items()
}


def get_model_caps(model_id: str) -> dict[str, Any]:
    caps = MODEL_CAPS.get(model_id)
    if caps is None:
        return {
            "max_days": 0,
            "has_native_dni_dhi": False,
            "supports_15min_radiation": False,
            "tier": "global",
        }
    return dict(caps)


def auto_select_models_for_location(lat: float | object, lon: float | None = None, requested_days: int = 1) -> list[str]:
    lat_valid = True
    if lon is None and hasattr(lat, "latitude") and hasattr(lat, "longitude"):
        _lat = float(getattr(lat, "latitude"))
        _lon = float(getattr(lat, "longitude"))
    else:
        try:
            _lat = float(lat)
            _lon = float(lon if lon is not None else 0.0)
        except (TypeError, ValueError):
            _lat = 0.0
            _lon = 0.0
            lat_valid = False

    if not (-90.0 <= _lat <= 90.0 and -180.0 <= _lon <= 180.0):
        lat_valid = False

    horizon = max(1, int(requested_days or 1))

    eligible = [
        model_id
        for model_id, spec in WEATHER_MODELS.items()
        if int(spec.get("max_days", 0) or 0) >= horizon
    ]

    stable_priority = [
        "ecmwf_ifs",
        "dwd_icon_eu",
        "meteofrance_seamless",
        "gfs",
        "knmi_harmonie_arome",
        "dwd_icon_d2",
    ]

    def _pick_preferred(preferred_order: list[str], *, max_models: int = 4) -> list[str]:
        selected = [m for m in preferred_order if m in eligible]
        if len(selected) < 2:
            for model_id in stable_priority:
                if model_id in eligible and model_id not in selected:
                    selected.append(model_id)
                    if len(selected) >= max(2, max_models):
                        break
        return selected[:max_models]

    if not lat_valid:
        chosen = _pick_preferred(stable_priority)
        return chosen or eligible[:1]

    in_benelux = 49.0 <= _lat <= 54.0 and 2.0 <= _lon <= 8.0
    in_europe = 35.0 <= _lat <= 72.0 and -15.0 <= _lon <= 35.0

    if in_benelux:
        preferred = (
            ["knmi_harmonie_arome", "dwd_icon_d2", "dwd_icon_eu", "ecmwf_ifs"]
            if horizon <= 2
            else ["knmi_harmonie_arome", "dwd_icon_eu", "ecmwf_ifs", "gfs"]
        )
        chosen = _pick_preferred(preferred)
    elif in_europe:
        chosen = _pick_preferred(["dwd_icon_eu", "meteofrance_seamless", "ecmwf_ifs", "gfs"])
    else:
        chosen = _pick_preferred(["ecmwf_ifs", "gfs"])

    if chosen:
        return chosen
    if eligible:
        return eligible[:1]
    fallback_any = [m for m in stable_priority if m in WEATHER_MODELS]
    return fallback_any[:1]


def select_week_ahead_models(*, requested_days: int = 7) -> list[str]:
    """Return all weather models valid for week-ahead horizons in stable deterministic order."""
    horizon = max(1, int(requested_days or 1))
    tier_order = {"short": 0, "medium": 1, "global": 2}

    eligible = [
        model_id
        for model_id, spec in WEATHER_MODELS.items()
        if int(spec.get("max_days", 0) or 0) >= horizon
    ]

    return sorted(
        eligible,
        key=lambda model_id: (
            0 if bool(WEATHER_MODELS[model_id].get("recommended_for_be", False)) else 1,
            tier_order.get(str(WEATHER_MODELS[model_id].get("tier") or "global"), 9),
            -int(WEATHER_MODELS[model_id].get("max_days", 0) or 0),
            model_id,
        ),
    )


def _nan_safe_hourly_median(matrix: pd.DataFrame) -> pd.Series:
    median = matrix.median(axis=1, skipna=True)
    all_missing_mask = matrix.notna().sum(axis=1) == 0
    median.loc[all_missing_mask] = np.nan
    return median.astype(float)


def should_use_satellite_nowcast_auto(
    *,
    latitude: float,
    longitude: float,
    timezone_name: str,
    requested_days: int,
    now_utc: dt.datetime | None = None,
) -> bool:
    if int(requested_days or 1) > 1:
        return False

    tz = ZoneInfo(str(timezone_name or "Europe/Brussels"))
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    now_local = now_utc.astimezone(tz)
    horizon_end = now_local + dt.timedelta(hours=6)
    sample_times = pd.date_range(start=now_local, end=horizon_end, freq="30min", tz=tz)
    if sample_times.empty:
        return False

    try:
        import pvlib  # type: ignore

        pvloc = pvlib.location.Location(latitude=float(latitude), longitude=float(longitude), tz=str(tz.key))
        solpos = pvloc.get_solarposition(sample_times)
        elevations = pd.to_numeric(solpos.get("apparent_elevation"), errors="coerce")
        return bool((elevations > 0).fillna(False).any())
    except Exception:
        return False



def _nowcast_sky_is_volatile(primary_weather: pd.DataFrame) -> bool:
    if primary_weather is None or primary_weather.empty:
        return True

    if "cloud_cover_pct" in primary_weather.columns:
        cloud = pd.to_numeric(primary_weather["cloud_cover_pct"], errors="coerce").dropna().clip(lower=0.0, upper=100.0)
        if len(cloud) >= 4:
            cloud_range = float(cloud.max() - cloud.min())
            mean_abs_diff = float(cloud.diff().abs().dropna().mean()) if len(cloud) > 1 else 0.0
            return bool(cloud_range >= 30.0 and mean_abs_diff >= 10.0)

    if "ghi_wm2" in primary_weather.columns:
        ghi = pd.to_numeric(primary_weather["ghi_wm2"], errors="coerce").dropna().clip(lower=0.0)
        if len(ghi) >= 4:
            mean_ghi = float(ghi.mean())
            if mean_ghi > 20.0:
                cv = float(ghi.std(ddof=0) / mean_ghi) if mean_ghi > 0 else 0.0
                return bool(cv >= 0.6)

    return True
WEATHER_MODEL_ALIASES: dict[str, str] = {
    "icon_d2": "dwd_icon_d2",
    "icon_eu": "dwd_icon_eu",
    "ifs": "ecmwf_ifs",
}

HISTORICAL_FORECAST_MODEL_PARAMS: dict[str, str] = {
    "knmi_harmonie_arome": "knmi_harmonie_arome_netherlands",
    "dwd_icon_d2": "icon_d2",
    "ecmwf_ifs": "ecmwf_ifs",
    "dwd_icon_eu": "icon_eu",
    "meteofrance_seamless": "meteofrance_seamless",
    "gfs": "gfs_seamless",
}

BASE_HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "weather_code",
]

IRRADIANCE_HOURLY_VARIABLES = [
    "direct_normal_irradiance",
    "diffuse_radiation",
    "direct_radiation",
]

IRR_CRITICAL = {"shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "direct_radiation"}

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

_WEATHER_CACHE: dict[tuple, tuple[float, core.ForecastResult, list[str], bool, Any]] = {}
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
    pv_ensemble_p25: pd.Series | None
    pv_ensemble_p90: pd.Series | None
    per_model_pv_totals_kwh: dict[str, float]
    missing_vars_by_model: dict[str, list[str]]
    derived_irradiance_by_model: dict[str, bool]
    derived_weather_code_by_model: dict[str, bool]
    derived_irradiance_hours_by_model: dict[str, int]
    quality_weight_factors_by_model: dict[str, float]
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
    fetch_meta_by_model: dict[str, dict[str, Any]]
    satellite_nowcast_used: bool = False
    satellite_nowcast_hours: int = 0
    satellite_nowcast_weight_factor: float | None = None
    satellite_nowcast_reason: str | None = None
    pv_tomorrow_low_high_kwh: dict[str, float | int | None] | None = None
    pv_models_used_count_per_hour: pd.Series | None = None


def _local_day_window(target_date: dt.date, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=tz)
    return start, start + pd.Timedelta(days=1)


def _sum_pv_kwh_for_local_day(pv_df: pd.DataFrame, target_date: dt.date, tz: str) -> tuple[float | None, int]:
    if not isinstance(pv_df, pd.DataFrame) or pv_df.empty:
        return None, 0

    series = None
    for col in ("pv_total_kwh", "pv_kwh", "pv_kwh_p50"):
        if col in pv_df.columns:
            series = pd.to_numeric(pv_df[col], errors="coerce")
            break
    if series is None:
        return None, 0

    start, end = _local_day_window(target_date, tz)
    index = pv_df.index
    if not isinstance(index, pd.DatetimeIndex):
        return None, 0

    if index.tz is None:
        index = index.tz_localize(tz)
    else:
        index = index.tz_convert(tz)

    mask = (index >= start) & (index < end)
    day_series = series.loc[mask]
    hours_count = int(day_series.notna().sum())
    if day_series.empty:
        return None, 0

    total_kwh = float(day_series.fillna(0.0).sum())
    if not math.isfinite(total_kwh):
        return None, hours_count
    return total_kwh, hours_count


def compute_pv_tomorrow_low_high_kwh(
    pv_by_model: dict[str, pd.DataFrame],
    target_date: dt.date,
    tz: str,
    *,
    min_hours: int = 18,
) -> dict[str, float | int | None]:
    model_totals: list[float] = []
    for _model_id, model_df in (pv_by_model or {}).items():
        total_kwh, hours_count = _sum_pv_kwh_for_local_day(model_df, target_date, tz)
        if total_kwh is None or hours_count < int(min_hours):
            continue
        if math.isfinite(total_kwh):
            model_totals.append(float(total_kwh))

    valid_models = len(model_totals)
    if valid_models >= 2:
        return {
            "low": float(min(model_totals)),
            "high": float(max(model_totals)),
            "valid_models": int(valid_models),
        }
    return {
        "low": None,
        "high": None,
        "valid_models": int(valid_models),
    }


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


def _provider_cache_location_bucket(lat: float, lon: float, elevation_m: float | None = None) -> str:
    elev_bucket = int(round(float(elevation_m))) if elevation_m is not None else "none"
    return f"{float(lat):.4f}_{float(lon):.4f}_{elev_bucket}"


def _provider_cache_path(model_id: str, target_date: dt.date, run_hour: int, location_bucket: str) -> Path:
    safe_bucket = "".join(ch if ch.isalnum() or ch in {"_", ".", "-"} else "_" for ch in str(location_bucket))
    return PROVIDER_CACHE_DIR / f"{model_id}__{safe_bucket}__{target_date.isoformat()}__{int(run_hour):02d}.json"


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


def _store_provider_cache(
    model_id: str,
    target_date: dt.date,
    run_hour: int,
    data: dict[str, Any],
    *,
    location_bucket: str,
) -> None:
    PROVIDER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "target_date": target_date.isoformat(),
        "run_hour": int(run_hour),
        "location_bucket": str(location_bucket),
        "cached_at_utc": _to_utc_iso(),
        "weather_payload": data,
    }
    _write_json_file(_provider_cache_path(model_id, target_date, run_hour, location_bucket), payload)


def _load_provider_cache(
    model_id: str,
    target_date: dt.date,
    run_hour: int,
    *,
    location_bucket: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for hour in range(int(run_hour), -1, -1):
        path = _provider_cache_path(model_id, target_date, hour, location_bucket)
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
                "location_bucket": str(payload.get("location_bucket") or location_bucket),
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


def _cache_key(model_id: str, lat: float, lon: float, tz: str, target_date: dt.date, elevation_m: float | None = None) -> tuple:
    elev_bucket = int(round(float(elevation_m))) if elevation_m is not None else None
    return (model_id, round(float(lat), 4), round(float(lon), 4), elev_bucket, str(tz), target_date.isoformat())




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
    return hourly.groupby(hourly.index.ceil("h")).mean(numeric_only=True)


def _align_backward_hourly_mean_to_hour_start(s: pd.Series) -> pd.Series:
    # Open-Meteo radiation series are preceding-hour means timestamped at hour-end.
    # We want hour-start labeling for planning tables and PV-per-hour.
    return pd.to_numeric(s, errors="coerce").shift(-1)

def _finalize_irradiance_components(
    *,
    ghi: pd.Series,
    dni: pd.Series,
    dhi: pd.Series,
    loc: core.Location,
    tz: str,
    missing_vars: list[str],
) -> tuple[pd.Series, pd.Series, list[str], bool, int]:
    dni_out = pd.to_numeric(dni, errors="coerce")
    dhi_out = pd.to_numeric(dhi, errors="coerce")
    ghi_out = pd.to_numeric(ghi, errors="coerce")

    needs_derivation_mask = (
        (ghi_out.notna())
        & (
            dni_out.isna()
            | dhi_out.isna()
        )
    )
    needs_derivation = bool(needs_derivation_mask.any())
    ghi_usable = bool(((ghi_out.notna()) & (ghi_out > 0)).any())
    before_missing_count = int((dni_out.isna() | dhi_out.isna()).sum())

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
    after_missing_count = int((dni_out.isna() | dhi_out.isna()).sum())
    derived_irradiance_hours = max(0, before_missing_count - after_missing_count)
    derived_irradiance = derived_irradiance_hours > 0

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

    return dni_out, dhi_out, sorted(missing_set), derived_irradiance, int(derived_irradiance_hours)




def normalize_weather_model_id(model_id: str) -> str:
    key = str(model_id or "").strip().lower()
    return WEATHER_MODEL_ALIASES.get(key, key)


def supported_weather_models() -> list[str]:
    return sorted(WEATHER_MODELS.keys())


def historical_forecast_params(model_id: str) -> tuple[str, dict[str, Any]]:
    canonical = normalize_weather_model_id(model_id)
    if canonical not in WEATHER_MODELS:
        raise RuntimeError(f"Unsupported weather model: {model_id}")
    params = {
        "models": HISTORICAL_FORECAST_MODEL_PARAMS.get(canonical, canonical),
    }
    return "https://historical-forecast-api.open-meteo.com/v1/forecast", params
def fetch_open_meteo_weather(
    model_id: str,
    loc: core.Location,
    tz: str,
    target_date: dt.date,
    *,
    accuracy_mode: bool = True,
    fast_mode: bool = False,
    endpoint_override: str | None = None,
    extra_params: dict[str, Any] | None = None,
    requested_days: int = 1,
) -> tuple[core.ForecastResult, list[str], bool, dict[str, Any]]:
    model_id = normalize_weather_model_id(model_id)
    if model_id not in WEATHER_MODELS:
        raise RuntimeError(f"Unsupported weather model: {model_id}")

    spec = WEATHER_MODELS[model_id]
    model_max_days = max(1, int(spec.get("max_days", 1)))
    requested_days_int = int(max(1, requested_days))
    horizon_days = max(1, min(requested_days_int, model_max_days))

    cache_extra_key = json.dumps(extra_params, sort_keys=True, default=str) if extra_params else ""
    key = (
        _cache_key(model_id, loc.latitude, loc.longitude, tz, target_date, loc.elevation_m),
        bool(accuracy_mode),
        bool(fast_mode),
        str(endpoint_override or ""),
        cache_extra_key,
        int(requested_days_int),
    )
    now = time.time()
    cached = _WEATHER_CACHE.get(key)
    if cached and now - cached[0] < _WEATHER_CACHE_TTL_S:
        cached_meta = cached[4] if len(cached) >= 5 else None
        cached_derived_hours = 1 if bool(cached[3]) else 0
        if isinstance(cached_meta, dict):
            cached_derived_hours = int(cached_meta.get("derived_irradiance_hours", cached_derived_hours))
        elif cached_meta is not None:
            cached_derived_hours = int(cached_meta)
        return cached[1], list(cached[2]), bool(cached[3]), {
            "source": "in_memory_cache",
            "cache_hit": True,
            "requested_days": int(requested_days_int),
            "horizon_days": int(horizon_days),
            "model_max_days": int(model_max_days),
            "derived_irradiance_hours": int(cached_derived_hours),
        }

    hourly_variables = BASE_HOURLY_VARIABLES[:] + IRRADIANCE_HOURLY_VARIABLES
    location_bucket = _provider_cache_location_bucket(loc.latitude, loc.longitude, loc.elevation_m)

    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "timezone": tz,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timeformat": "iso8601",
        "start_date": target_date.isoformat(),
        "end_date": (target_date + dt.timedelta(days=horizon_days - 1)).isoformat(),
        "hourly": ",".join(hourly_variables),
        "daily": "sunrise,sunset",
    }
    params.update(spec.get("params", {}))
    if extra_params:
        params.update(extra_params)
    if loc.elevation_m is not None:
        params["elevation"] = int(round(float(loc.elevation_m)))

    use_icon15 = (
        model_id == "dwd_icon_d2"
        and bool(spec.get("supports_15min_radiation", False))
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
        "requested_days": int(requested_days_int),
        "horizon_days": int(horizon_days),
        "model_max_days": int(model_max_days),
    }
    request_params = dict(params)
    request_endpoint = str(endpoint_override or spec["endpoint"])
    response_meta: dict[str, Any] | None = None

    request_start = time.perf_counter()
    try:
        if circuit_open:
            raise WeatherProviderError(
                category="circuit_open",
                status=None,
                message=f"Circuit breaker open for {model_id}; skip live fetch for {open_for_seconds}s",
            )
        request_out = _request_open_meteo(request_endpoint, params, model_id=model_id)
        if isinstance(request_out, tuple) and len(request_out) == 2:
            data, response_meta = request_out
        else:
            data, response_meta = request_out, None
        _log_model_fetch(
            model_id=model_id,
            endpoint=request_endpoint,
            params=params,
            elapsed_ms=int((time.perf_counter() - request_start) * 1000),
            category="ok",
            status=200,
            outcome="success",
        )
    except WeatherProviderError as exc:
        _log_model_fetch(
            model_id=model_id,
            endpoint=request_endpoint,
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
                request_out = _request_open_meteo(fallback_endpoint, fallback_params, model_id=model_id)
                if isinstance(request_out, tuple) and len(request_out) == 2:
                    data, response_meta = request_out
                else:
                    data, response_meta = request_out, None
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
            cached_payload, cache_meta = _load_provider_cache(
                model_id,
                target_date,
                run_hour,
                location_bucket=location_bucket,
            )
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
        _store_provider_cache(
            model_id,
            target_date,
            run_hour,
            data,
            location_bucket=location_bucket,
        )

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
    ghi = _series("shortwave_radiation", 0.0, record_missing=False).fillna(0.0).clip(lower=0.0)
    bhi = _series("direct_radiation", np.nan, record_missing=False)
    dni_api = _series("direct_normal_irradiance", np.nan, record_missing=False)
    dhi_api = _series("diffuse_radiation", np.nan, record_missing=False)

    df["ghi_wm2"] = ghi
    weather_code_series = _series("weather_code", np.nan, record_missing=False)

    dni_candidate = dni_api.copy()
    dhi_candidate = dhi_api.copy()

    minutely = data.get("minutely_15") if isinstance(data.get("minutely_15"), dict) else {}
    if use_icon15 and minutely:
        agg15 = _aggregate_minutely_15_to_hourly(minutely, tz=tz)
        if not agg15.empty:
            agg15 = agg15.reindex(df.index)
            if "ghi_wm2" in agg15.columns:
                ghi_15 = pd.to_numeric(agg15["ghi_wm2"], errors="coerce")
                df["ghi_wm2"] = ghi_15.combine_first(df["ghi_wm2"])
            dni_15 = pd.to_numeric(agg15["dni_wm2"], errors="coerce") if "dni_wm2" in agg15.columns else pd.Series(np.nan, index=df.index)
            dhi_15 = pd.to_numeric(agg15["dhi_wm2"], errors="coerce") if "dhi_wm2" in agg15.columns else pd.Series(np.nan, index=df.index)
            dni_candidate = dni_15.combine_first(dni_candidate)
            dhi_candidate = dhi_15.combine_first(dhi_candidate)

    df["ghi_wm2"] = _align_backward_hourly_mean_to_hour_start(df["ghi_wm2"])
    dni_candidate = _align_backward_hourly_mean_to_hour_start(dni_candidate)
    dhi_candidate = _align_backward_hourly_mean_to_hour_start(dhi_candidate)
    bhi = _align_backward_hourly_mean_to_hour_start(bhi)

    if core.PVLIB_AVAILABLE and dni_candidate.isna().any() and bhi.notna().any():
        import pvlib  # type: ignore

        pvloc = pvlib.location.Location(latitude=loc.latitude, longitude=loc.longitude, tz=tz)
        solpos = pvloc.get_solarposition(df.index)
        cos_zen = pd.to_numeric(solpos.get("apparent_zenith"), errors="coerce").apply(
            lambda z: max(0.0, math.cos(math.radians(z))) if pd.notna(z) else 0.0
        )
        dni_from_bhi = pd.Series(0.0, index=df.index, dtype=float)
        daylight_mask = cos_zen > 0
        dni_from_bhi.loc[daylight_mask] = (
            pd.to_numeric(bhi.loc[daylight_mask], errors="coerce")
            / cos_zen.loc[daylight_mask]
        )
        dni_from_bhi = pd.to_numeric(dni_from_bhi, errors="coerce").clip(lower=0.0)
        fill_mask = dni_candidate.isna() & bhi.notna()
        dni_candidate = dni_candidate.where(~fill_mask, dni_from_bhi)

    if dhi_candidate.isna().any() and bhi.notna().any():
        dhi_from_bhi = (pd.to_numeric(df["ghi_wm2"], errors="coerce") - pd.to_numeric(bhi, errors="coerce")).clip(lower=0.0)
        fill_mask = dhi_candidate.isna() & bhi.notna()
        dhi_candidate = dhi_candidate.where(~fill_mask, dhi_from_bhi)

    dni_final, dhi_final, missing_vars, derived_irradiance, derived_irradiance_hours = _finalize_irradiance_components(
        ghi=df["ghi_wm2"],
        dni=dni_candidate,
        dhi=dhi_candidate,
        loc=loc,
        tz=tz,
        missing_vars=missing_vars,
    )
    fetch_meta["derived_irradiance_hours"] = int(derived_irradiance_hours)
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

    df = df[["temp_air_c", "ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_cover_pct", "wind_speed_ms"]]
    idx = pd.to_datetime(df.index, errors="coerce")
    if idx.isna().all():
        raise RuntimeError(f"Open-Meteo hourly forecast index invalid for {model_id}")
    if idx.tz is None:
        idx = idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    else:
        idx = idx.tz_convert(tz)
    df.index = idx
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    range_start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=tz)
    range_end = pd.Timestamp(dt.datetime.combine(target_date + dt.timedelta(days=horizon_days), dt.time(0, 0)), tz=tz)
    expected_index = pd.date_range(range_start, range_end, freq="h", inclusive="left")
    df = df.reindex(expected_index)

    availability = availability.reindex(df.index).fillna(False).astype(bool)
    for col in ["ghi_wm2", "dni_wm2", "dhi_wm2"]:
        df.loc[~availability[col], col] = np.nan

    weather_code_series = pd.to_numeric(weather_code_series, errors="coerce").reindex(df.index)
    derived_weather_code = False
    if weather_code_series.isna().all():
        cloud_cover_series = pd.to_numeric(df.get("cloud_cover_pct"), errors="coerce") if "cloud_cover_pct" in df.columns else pd.Series(np.nan, index=df.index)
        if cloud_cover_series.notna().any():
            # UI continuity fallback only: this is intentionally simple and does not attempt full WMO weather code logic.
            weather_code_series = pd.Series(3.0, index=df.index, dtype=float)
            weather_code_series.loc[cloud_cover_series <= 20.0] = 0.0
            weather_code_series.loc[(cloud_cover_series > 20.0) & (cloud_cover_series <= 50.0)] = 2.0
            weather_code_series.loc[(cloud_cover_series > 50.0) & (cloud_cover_series <= 80.0)] = 3.0
            weather_code_series.loc[cloud_cover_series > 80.0] = 3.0
        else:
            weather_code_series = pd.Series(3.0, index=df.index, dtype=float)
        derived_weather_code = True
    df["weather_code"] = weather_code_series
    fetch_meta["derived_weather_code"] = bool(derived_weather_code)

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
    _WEATHER_CACHE[key] = (
        time.time(),
        forecast,
        list(set(missing_vars)),
        bool(derived_irradiance),
        {
            "derived_irradiance_hours": int(derived_irradiance_hours),
            "requested_days": int(requested_days_int),
            "horizon_days": int(horizon_days),
            "model_max_days": int(model_max_days),
        },
    )
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
    missing_vars_by_model: dict[str, list[str]] | None = None,
    derived_irradiance_by_model: dict[str, bool] | None = None,
    derived_irradiance_hours_by_model: dict[str, int] | None = None,
) -> tuple[pd.Series, dict[str, float] | None, dict[str, float]]:
    weighted_subset = dict(dynamic_weights or {})
    if not weighted_subset:
        weighted_subset = {m: DEFAULT_WEIGHTED_BELGIUM[m] for m in selected_models if m in DEFAULT_WEIGHTED_BELGIUM}
    if not weighted_subset:
        return pd.concat(series_map.values(), axis=1).mean(axis=1), None, {}

    min_weight_factor = 0.20
    quality_factors: dict[str, float] = {}
    adjusted_weights: dict[str, float] = {}
    for model_id, base_w in weighted_subset.items():
        missing = set((missing_vars_by_model or {}).get(model_id, []))
        derived = bool((derived_irradiance_by_model or {}).get(model_id))
        has_irr_missing = bool(missing.intersection(IRR_CRITICAL))
        factor = 1.0
        if has_irr_missing and derived:
            factor = 0.50
        elif has_irr_missing:
            factor = 0.60
        elif derived:
            factor = 0.80
        derived_hours = int((derived_irradiance_hours_by_model or {}).get(model_id, 0) or 0)
        if derived_hours <= 0:
            hours_factor = 1.0
        elif derived_hours <= 6:
            hours_factor = 0.90
        elif derived_hours <= 12:
            hours_factor = 0.75
        else:
            hours_factor = 0.60
        factor *= float(hours_factor)

        factor = max(min_weight_factor, float(factor))
        quality_factors[model_id] = factor
        adjusted_weights[model_id] = float(base_w) * factor

    total = sum(adjusted_weights.values())
    if total <= 0:
        return pd.concat(series_map.values(), axis=1).mean(axis=1), None, quality_factors
    normalized = {m: w / total for m, w in adjusted_weights.items()}
    matrix = pd.DataFrame({m: series_map[m] for m in normalized if m in series_map})
    if matrix.empty:
        return pd.concat(series_map.values(), axis=1).mean(axis=1), None, quality_factors
    weight_series = pd.Series(normalized)
    weighted_values = matrix.mul(weight_series, axis=1)
    numerator = weighted_values.sum(axis=1, skipna=True)
    denominator = matrix.notna().mul(weight_series, axis=1).sum(axis=1)
    out = numerator.div(denominator.where(denominator > 0))
    return out.astype(float), {m: normalized[m] for m in matrix.columns}, {m: quality_factors[m] for m in matrix.columns}


def fetch_satellite_radiation_nowcast(lat: float, lon: float, tz: str, forecast_hours: int = 6) -> pd.DataFrame:
    endpoint = "https://satellite-api.open-meteo.com/v1/satellite-radiation"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "timezone": str(tz),
        "hourly": "shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
        "forecast_hours": int(max(1, forecast_hours)),
        "past_hours": 0,
    }
    data, _ = _request_open_meteo(endpoint, params=params, timeout=20)
    hourly = data.get("hourly") if isinstance(data.get("hourly"), dict) else {}
    times = pd.to_datetime(hourly.get("time", []), errors="coerce")
    if len(times) == 0:
        return pd.DataFrame()
    if getattr(times, "tz", None) is None:
        times = times.tz_localize(tz)
    else:
        times = times.tz_convert(tz)
    df = pd.DataFrame(index=times)
    df["shortwave_radiation"] = pd.to_numeric(pd.Series(hourly.get("shortwave_radiation", []), index=times), errors="coerce")
    df["direct_normal_irradiance"] = pd.to_numeric(pd.Series(hourly.get("direct_normal_irradiance", []), index=times), errors="coerce")
    df["diffuse_radiation"] = pd.to_numeric(pd.Series(hourly.get("diffuse_radiation", []), index=times), errors="coerce")
    return df.sort_index()


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
            out[var] = _nan_safe_hourly_median(matrix)
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
    requested_days: int = 1,
    use_satellite_nowcast_0_6h: bool = False,
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
    derived_weather_code_by_model: dict[str, bool] = {}
    derived_irradiance_hours_by_model: dict[str, int] = {}
    quality_weight_factors_by_model: dict[str, float] = {}
    failed_models: list[str] = []
    failed_model_reasons: dict[str, dict[str, Any]] = {}
    model_live_failed_used_cached: dict[str, bool] = {}
    weather_ok: dict[str, core.ForecastResult] = {}
    pv_by_model: dict[str, pd.DataFrame] = {}
    provider_payloads_by_model: dict[str, dict[str, Any]] = {}
    fetch_meta_by_model: dict[str, dict[str, Any]] = {}

    horizon_days = max(1, int(requested_days))
    canonical_index = pd.date_range(
        pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=tz),
        pd.Timestamp(dt.datetime.combine(target_date + dt.timedelta(days=horizon_days), dt.time(0, 0)), tz=tz),
        freq="h",
        inclusive="left",
    )

    def _fetch_and_prepare(model_id: str) -> tuple[str, core.ForecastResult, dict[str, pd.Series], float, list[str], bool, int, dict[str, Any]]:
        weather, missing_vars, derived_irradiance, fetch_meta = fetch_open_meteo_weather(
            model_id,
            loc,
            tz,
            target_date,
            accuracy_mode=accuracy_mode,
            fast_mode=fast_mode,
            requested_days=horizon_days,
        )
        requested_days_int = int(max(1, fetch_meta.get("requested_days", horizon_days)))
        model_horizon_days = int(max(1, fetch_meta.get("horizon_days", requested_days_int)))
        overlap_hours = int(min(requested_days_int, model_horizon_days) * 24)
        tail_hours_expected = int(max(0, (requested_days_int - model_horizon_days) * 24))
        overlap_index = canonical_index[:overlap_hours] if overlap_hours > 0 else canonical_index[:0]
        tail_index = canonical_index[overlap_hours: overlap_hours + tail_hours_expected] if tail_hours_expected > 0 else canonical_index[:0]
        missing_overlap = int(weather.df.reindex(overlap_index).isna().all(axis=1).sum()) if len(overlap_index) else 0
        missing_tail = int(weather.df.reindex(tail_index).isna().all(axis=1).sum()) if len(tail_index) else 0
        missing_total = int(weather.df.reindex(canonical_index).isna().all(axis=1).sum())
        fetch_meta["missing_hours_overlap"] = int(missing_overlap)
        fetch_meta["missing_hours_tail"] = int(missing_tail)
        fetch_meta["missing_hours_total"] = int(missing_total)
        fetch_meta["expected_tail_hours"] = int(tail_hours_expected)
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "[weather_ensemble] model=%s requested_days=%s spec_max_days=%s fetch_meta_keys=%s model_horizon_days=%s",
                model_id,
                requested_days_int,
                int(WEATHER_MODELS.get(model_id, {}).get("max_days", requested_days_int)),
                sorted(fetch_meta.keys()) if isinstance(fetch_meta, dict) else [],
                model_horizon_days,
            )
        if len(overlap_index) and missing_overlap > 0 and _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "[weather_ensemble] model=%s index_alignment weather_first=%s weather_last=%s canonical_first=%s canonical_last=%s overlap_first=%s overlap_last=%s weather_tz=%s canonical_tz=%s overlap_tz=%s weather_dupes=%s canonical_dupes=%s overlap_dupes=%s",
                model_id,
                weather.df.index[0] if len(weather.df.index) else None,
                weather.df.index[-1] if len(weather.df.index) else None,
                canonical_index[0] if len(canonical_index) else None,
                canonical_index[-1] if len(canonical_index) else None,
                overlap_index[0] if len(overlap_index) else None,
                overlap_index[-1] if len(overlap_index) else None,
                str(weather.df.index.tz),
                str(canonical_index.tz),
                str(overlap_index.tz),
                int(weather.df.index.duplicated().sum()),
                int(canonical_index.duplicated().sum()),
                int(overlap_index.duplicated().sum()),
            )
        if len(overlap_index) and missing_overlap == len(overlap_index):
            exc = WeatherProviderError(
                category="invalid_payload",
                status=None,
                message=(
                    f"Weather payload for {model_id} has no overlapping hourly coverage "
                    f"for requested horizon ({len(overlap_index)}h missing)."
                ),
            )
            setattr(exc, "fetch_meta", dict(fetch_meta))
            raise exc
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
                model_weather_df[irr_col] = pd.to_numeric(model_weather_df[irr_col], errors="coerce").clip(lower=0.0)
        if "temp_air_c" in model_weather_df.columns:
            model_weather_df["temp_air_c"] = pd.to_numeric(model_weather_df["temp_air_c"], errors="coerce")
        if "wind_speed_ms" in model_weather_df.columns:
            model_weather_df["wind_speed_ms"] = pd.to_numeric(model_weather_df["wind_speed_ms"], errors="coerce").clip(lower=0.0)
        weather = core.ForecastResult(df=model_weather_df, sunrise=weather.sunrise, sunset=weather.sunset)

        overlap_hours = int(min(requested_days_int, model_horizon_days) * 24)
        overlap_index = canonical_index[:max(overlap_hours, 0)]
        pv_input_df = model_weather_df.loc[overlap_index].copy() if len(overlap_index) else model_weather_df.iloc[:0].copy()
        model_pv = core.build_pv_forecast(pv_input_df, loc, tz=tz).reindex(canonical_index)
        for req in ["pv_total_kwh", "pv_total_unclipped_kwh", "pv_east_kwh", "pv_south_kwh", "pv_clipped_kwh"]:
            if req not in model_pv.columns:
                model_pv[req] = np.nan
        pv_total = pd.to_numeric(model_pv["pv_total_kwh"], errors="coerce").where(lambda x: x >= 0)
        pv_unclipped = pd.to_numeric(model_pv["pv_total_unclipped_kwh"], errors="coerce").where(lambda x: x >= 0)
        pv_unclipped = pd.Series(np.maximum(pv_unclipped, pv_total), index=pv_total.index)
        pv_east = pd.to_numeric(model_pv["pv_east_kwh"], errors="coerce").where(lambda x: x >= 0)
        pv_south = pd.to_numeric(model_pv["pv_south_kwh"], errors="coerce").where(lambda x: x >= 0)
        pv_clipped = pd.to_numeric(model_pv["pv_clipped_kwh"], errors="coerce").where(lambda x: x >= 0)
        return model_id, weather, {
            "pv_total_kwh": pv_total,
            "pv_total_unclipped_kwh": pv_unclipped,
            "pv_east_kwh": pv_east,
            "pv_south_kwh": pv_south,
            "pv_clipped_kwh": pv_clipped,
        }, float(pv_total.sum()), missing_vars, bool(derived_irradiance), int(fetch_meta.get("derived_irradiance_hours", 0)), fetch_meta

    max_workers = min(max(len(selected), 1), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_fetch_and_prepare, model_id): model_id for model_id in selected}
        for fut in as_completed(future_map):
            model_id = future_map[fut]
            try:
                model_id, weather, pv_cols, pv_total_sum, missing_vars, derived_irradiance, derived_irradiance_hours, fetch_meta = fut.result()
                for col_name, series in pv_cols.items():
                    per_model_pv_columns[col_name][model_id] = series
                pv_by_model[model_id] = pd.DataFrame(pv_cols).reindex(canonical_index)
                per_model_pv_totals[model_id] = pv_total_sum
                missing_vars_by_model[model_id] = missing_vars
                derived_irradiance_by_model[model_id] = bool(derived_irradiance)
                derived_weather_code_by_model[model_id] = bool(fetch_meta.get("derived_weather_code", False)) if isinstance(fetch_meta, dict) else False
                derived_irradiance_hours_by_model[model_id] = int(derived_irradiance_hours)
                fetch_meta_by_model[model_id] = dict(fetch_meta) if isinstance(fetch_meta, dict) else {}
                weather_ok[model_id] = weather
                model_live_failed_used_cached[model_id] = bool(fetch_meta.get("live_failed_used_cached", False))
                provider_payload = fetch_meta.get("provider_payload") if isinstance(fetch_meta, dict) else None
                if isinstance(provider_payload, dict):
                    provider_payloads_by_model[model_id] = provider_payload
                missing_overlap = int(fetch_meta.get("missing_hours_overlap", 0))
                missing_tail = int(fetch_meta.get("missing_hours_tail", 0))
                expected_tail = int(fetch_meta.get("expected_tail_hours", 0))
                missing_total = int(fetch_meta.get("missing_hours_total", 0))
                if missing_overlap > 2:
                    _LOGGER.warning(
                        "[weather_ensemble] model=%s missing_hours_overlap=%s missing_hours_tail=%s missing_hours_total=%s expected_tail_hours=%s",
                        model_id,
                        missing_overlap,
                        missing_tail,
                        missing_total,
                        expected_tail,
                    )
                elif missing_tail > 0 and expected_tail > 0:
                    tail_log = _LOGGER.debug if missing_tail == expected_tail else _LOGGER.info
                    tail_log(
                        "[weather_ensemble] model=%s missing_hours_tail=%s expected_tail_hours=%s missing_hours_overlap=%s",
                        model_id,
                        missing_tail,
                        expected_tail,
                        missing_overlap,
                    )
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

    satellite_nowcast_used = False
    satellite_nowcast_hours = 0
    satellite_nowcast_weight_factor = 2.0
    satellite_nowcast_reason = None

    if not per_model_pv_columns["pv_total_kwh"]:
        err = RuntimeError("All weather model requests failed.")
        setattr(err, "failed_models", list(failed_models))
        setattr(err, "failed_model_reasons", dict(failed_model_reasons))
        setattr(err, "weights_used", None)
        raise err

    if use_satellite_nowcast_0_6h and requested_days <= 1 and weather_ok:
        now_local = pd.Timestamp.now(tz=tz).floor("h")
        sat_hours = min(6, len(canonical_index))
        sat_index = canonical_index[:sat_hours]
        if len(sat_index) == 0:
            satellite_nowcast_reason = "skipped (no overlap)"
        else:
            primary_weather = weather_ok[next(iter(weather_ok.keys()))].df.reindex(sat_index)
            ghi_probe = pd.to_numeric(primary_weather.get("ghi_wm2"), errors="coerce") if "ghi_wm2" in primary_weather.columns else pd.Series(np.nan, index=sat_index)
            if ghi_probe.fillna(0.0).max() <= 5.0:
                satellite_nowcast_reason = "skipped (night)"
            elif not _nowcast_sky_is_volatile(primary_weather):
                satellite_nowcast_reason = "skipped (stable sky)"
            else:
                try:
                    sat_df = fetch_satellite_radiation_nowcast(loc.latitude, loc.longitude, tz, forecast_hours=6)
                    sat_df = sat_df.reindex(sat_index)
                    if not sat_df.empty and sat_df[["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"]].notna().any().any():
                        pseudo = pd.DataFrame(index=sat_index)
                        pseudo["temp_air_c"] = pd.to_numeric(primary_weather.get("temp_air_c"), errors="coerce") if "temp_air_c" in primary_weather.columns else np.nan
                        pseudo["wind_speed_ms"] = pd.to_numeric(primary_weather.get("wind_speed_ms"), errors="coerce") if "wind_speed_ms" in primary_weather.columns else np.nan
                        pseudo["cloud_cover_pct"] = pd.to_numeric(primary_weather.get("cloud_cover_pct"), errors="coerce") if "cloud_cover_pct" in primary_weather.columns else np.nan
                        pseudo["ghi_wm2"] = pd.to_numeric(sat_df["shortwave_radiation"], errors="coerce")
                        pseudo["dni_wm2"] = pd.to_numeric(sat_df["direct_normal_irradiance"], errors="coerce")
                        pseudo["dhi_wm2"] = pd.to_numeric(sat_df["diffuse_radiation"], errors="coerce")
                        sat_pv = core.build_pv_forecast(pseudo, loc, tz=tz).reindex(canonical_index)
                        for req in ["pv_total_kwh", "pv_total_unclipped_kwh", "pv_east_kwh", "pv_south_kwh", "pv_clipped_kwh"]:
                            if req not in sat_pv.columns:
                                sat_pv[req] = np.nan
                        model_id = "sat_nowcast"
                        for col_name in per_model_pv_columns:
                            series = pd.Series(np.nan, index=canonical_index, dtype=float)
                            series.loc[sat_index] = pd.to_numeric(sat_pv[col_name].reindex(sat_index), errors="coerce")
                            per_model_pv_columns[col_name][model_id] = series
                        pv_by_model[model_id] = sat_pv
                        per_model_pv_totals[model_id] = float(pd.to_numeric(sat_pv["pv_total_kwh"], errors="coerce").sum(min_count=1))
                        missing_vars_by_model[model_id] = []
                        derived_irradiance_by_model[model_id] = False
                        derived_weather_code_by_model[model_id] = False
                        quality_weight_factors_by_model[model_id] = 1.0
                        satellite_nowcast_used = True
                        satellite_nowcast_hours = int(len(sat_index))
                        satellite_nowcast_reason = "used"
                    else:
                        satellite_nowcast_reason = "skipped (no satellite data)"
                except Exception as exc:
                    satellite_nowcast_reason = f"skipped ({type(exc).__name__})"
    elif use_satellite_nowcast_0_6h and requested_days > 1:
        satellite_nowcast_reason = "skipped (week-ahead horizon)"

    def _ensemble_column(column_name: str) -> tuple[pd.Series, dict[str, float] | None, dict[str, float]]:
        model_series = per_model_pv_columns[column_name]
        matrix = pd.concat(model_series.values(), axis=1)
        if ensemble_method == "median":
            return _nan_safe_hourly_median(matrix), None, {}
        if ensemble_method == "mean":
            return matrix.mean(axis=1, skipna=True), None, {}
        model_keys = list(model_series.keys())
        dynamic_weights = _load_dynamic_weights(model_keys) or {}
        if satellite_nowcast_used and "sat_nowcast" in model_series:
            dynamic_weights["sat_nowcast"] = max(float(dynamic_weights.get("sat_nowcast", 1.0)), float(satellite_nowcast_weight_factor))
        return _weighted_ensemble(
            model_series,
            model_keys,
            dynamic_weights=dynamic_weights,
            missing_vars_by_model=missing_vars_by_model,
            derived_irradiance_by_model=derived_irradiance_by_model,
            derived_irradiance_hours_by_model=derived_irradiance_hours_by_model,
        )

    ensemble_ac_p50, weights_used, quality_weight_factors_by_model = _ensemble_column("pv_total_kwh")
    ensemble_unclipped_p50, _, _ = _ensemble_column("pv_total_unclipped_kwh")
    ensemble_east_p50, _, _ = _ensemble_column("pv_east_kwh")
    ensemble_south_p50, _, _ = _ensemble_column("pv_south_kwh")

    pv_matrix = pd.DataFrame(per_model_pv_columns["pv_total_kwh"]).reindex(canonical_index)
    if ensemble_method == "weighted":
        base_w = dict(weights_used or {})
        if base_w:
            w = pd.Series(base_w, dtype=float)
            num = pv_matrix.mul(w, axis=1).sum(axis=1, skipna=True)
            den = pv_matrix.notna().mul(w, axis=1).sum(axis=1)
            ensemble_ac_p50 = (num / den).where(den > 0)
    pv_models_used_count = pv_matrix.notna().sum(axis=1).astype(int)

    if len(per_model_pv_columns["pv_total_kwh"]) >= 3 and ensemble_method != "median":
        matrix = pd.concat(per_model_pv_columns["pv_total_kwh"].values(), axis=1)
        spread = (matrix.max(axis=1) - matrix.min(axis=1)).fillna(0.0)
        spread_median = float(spread.median()) if not spread.empty else 0.0
        extreme_mask = spread > max(0.5, 2.0 * spread_median)
        if int(extreme_mask.sum()) >= 3:
            median_ac = _nan_safe_hourly_median(matrix)
            matrix_unclip = pd.concat(per_model_pv_columns["pv_total_unclipped_kwh"].values(), axis=1)
            median_unclip = _nan_safe_hourly_median(matrix_unclip)
            ensemble_ac_p50.loc[extreme_mask] = median_ac.loc[extreme_mask]
            ensemble_unclipped_p50.loc[extreme_mask] = median_unclip.loc[extreme_mask]

    ensemble_ac_p50 = ensemble_ac_p50.reindex(canonical_index)
    ensemble_unclipped_p50 = ensemble_unclipped_p50.reindex(canonical_index)
    ensemble_east_p50 = ensemble_east_p50.reindex(canonical_index)
    ensemble_south_p50 = ensemble_south_p50.reindex(canonical_index)

    ensemble_unclipped_p50 = pd.Series(np.maximum(ensemble_unclipped_p50, ensemble_ac_p50), index=ensemble_ac_p50.index)
    east_south_total = (ensemble_east_p50 + ensemble_south_p50)
    rebalance = pd.Series(1.0, index=ensemble_ac_p50.index, dtype=float)
    positive_split = east_south_total > 0
    rebalance.loc[positive_split] = (ensemble_ac_p50.loc[positive_split] / east_south_total.loc[positive_split]).astype(float)
    split_known = ensemble_east_p50.notna() | ensemble_south_p50.notna()
    ensemble_east_p50 = (ensemble_east_p50 * rebalance).clip(lower=0.0).where(split_known)
    ensemble_south_p50 = (ensemble_south_p50 * rebalance).clip(lower=0.0).where(split_known)
    ensemble_clipped_p50 = (ensemble_unclipped_p50 - ensemble_ac_p50).clip(lower=0.0)

    matrix = pd.DataFrame(per_model_pv_columns["pv_total_kwh"]).reindex(canonical_index)
    p10 = matrix.quantile(0.10, axis=1)
    p25 = matrix.quantile(0.25, axis=1)
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
    pv_tomorrow_low_high_kwh = compute_pv_tomorrow_low_high_kwh(
        pv_by_model=pv_by_model,
        target_date=target_date,
        tz=tz,
    )

    return EnsembleWeatherResult(
        weather_primary=weather_ok[primary_model],
        pv_ensemble_p50=ensemble_ac_p50.astype(float),
        pv_ensemble_unclipped_p50=ensemble_unclipped_p50.astype(float),
        pv_ensemble_clipped_p50=ensemble_clipped_p50.astype(float),
        pv_ensemble_east_p50=ensemble_east_p50.astype(float),
        pv_ensemble_south_p50=ensemble_south_p50.astype(float),
        pv_ensemble_p10=p10.astype(float) if p10 is not None else None,
        pv_ensemble_p25=p25.astype(float) if p25 is not None else None,
        pv_ensemble_p90=p90.astype(float) if p90 is not None else None,
        per_model_pv_totals_kwh=per_model_pv_totals,
        missing_vars_by_model=missing_vars_by_model,
        derived_irradiance_by_model=derived_irradiance_by_model,
        derived_weather_code_by_model=derived_weather_code_by_model,
        derived_irradiance_hours_by_model=derived_irradiance_hours_by_model,
        quality_weight_factors_by_model=quality_weight_factors_by_model,
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
        fetch_meta_by_model=fetch_meta_by_model,
        pv_tomorrow_low_high_kwh=pv_tomorrow_low_high_kwh,
        pv_models_used_count_per_hour=pv_models_used_count,
        satellite_nowcast_used=bool(satellite_nowcast_used),
        satellite_nowcast_hours=int(satellite_nowcast_hours),
        satellite_nowcast_weight_factor=float(satellite_nowcast_weight_factor) if satellite_nowcast_used else None,
        satellite_nowcast_reason=satellite_nowcast_reason,
    )
