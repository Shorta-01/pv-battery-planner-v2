from __future__ import annotations

import datetime as dt
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import planner_core as core

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
        "badges": ["⭐", "🧩"],
        "recommended_for_be": True,
        "capability": {
            "ghi_native": True,
            "direct_native": False,
            "diffuse_native": False,
            "dni_native": False,
            "notes": "GHI only; direct/diffuse derived by Open-Meteo separation.",
        },
    },
    "dwd_icon_d2": {
        "label": "DWD ICON-D2",
        "endpoint": "https://api.open-meteo.com/v1/dwd-icon",
        "params": {"models": "icon_d2"},
        "badges": ["⭐", "🟩"],
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
        "badges": ["⭐", "🧩"],
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
        "badges": ["🧩"],
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
        "badges": ["🧩"],
        "recommended_for_be": True,
        "capability": {
            "ghi_native": True,
            "direct_native": False,
            "diffuse_native": False,
            "dni_native": False,
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
    selected_models: list[str]
    weights_used: dict[str, float] | None
    weather_primary_model_id: str
    weather_by_model: dict[str, core.ForecastResult]
    weather_ensemble_table: core.ForecastResult


class WeatherProviderError(RuntimeError):
    def __init__(self, *, category: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.status = status
        self.message = message

    def to_reason(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "status": self.status,
            "message": self.message,
        }


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


def _request_open_meteo(url: str, params: dict[str, Any], model_id: str) -> dict[str, Any]:
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
    try:
        response = _SESSION.get(url, params=params, timeout=(5, 30))
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        category = "rate_limited" if status == 429 else "provider_down" if status in {500, 502, 503, 504} else "http_error"
        provider_reason = ""
        if exc.response is not None:
            try:
                error_payload = exc.response.json()
            except ValueError:
                error_payload = None
            if isinstance(error_payload, dict) and error_payload.get("reason"):
                provider_reason = str(error_payload.get("reason"))
            else:
                provider_reason = (exc.response.text or "").strip()[:200]
        reason_suffix = f" reason={provider_reason}" if provider_reason else ""
        raise WeatherProviderError(
            category=category,
            status=status,
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
    return data


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

def fetch_open_meteo_weather(
    model_id: str,
    loc: core.Location,
    tz: str,
    target_date: dt.date,
    *,
    accuracy_mode: bool = True,
    fast_mode: bool = False,
) -> tuple[core.ForecastResult, list[str], bool]:
    if model_id not in WEATHER_MODELS:
        raise RuntimeError(f"Unsupported weather model: {model_id}")

    key = (_cache_key(model_id, loc.latitude, loc.longitude, tz, target_date), bool(accuracy_mode), bool(fast_mode))
    now = time.time()
    cached = _WEATHER_CACHE.get(key)
    if cached and now - cached[0] < _WEATHER_CACHE_TTL_S:
        return cached[1], list(cached[2]), bool(cached[3])

    spec = WEATHER_MODELS[model_id]
    capability = spec.get("capability", {}) if isinstance(spec, dict) else {}
    hourly_variables = BASE_HOURLY_VARIABLES[:]
    if bool(capability.get("dni_native")):
        hourly_variables.append("direct_normal_irradiance")
    if bool(capability.get("diffuse_native")):
        hourly_variables.append("diffuse_radiation")

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

    try:
        data = _request_open_meteo(spec["endpoint"], params, model_id=model_id)
    except WeatherProviderError as exc:
        fallback_model = FORECAST_FALLBACK_MODELS.get(model_id)
        requested_model = str(params.get("models") or "").strip()
        can_retry_with_forecast = (
            exc.category == "http_error"
            and exc.status in {400, 404}
            and bool(fallback_model)
            and fallback_model != requested_model
        )
        if not can_retry_with_forecast:
            raise

        fallback_params = dict(params)
        fallback_params["hourly"] = ",".join(BASE_HOURLY_VARIABLES + IRRADIANCE_HOURLY_VARIABLES)
        fallback_params["models"] = fallback_model
        data = _request_open_meteo("https://api.open-meteo.com/v1/forecast", fallback_params, model_id=model_id)

    hourly = data.get("hourly") if isinstance(data.get("hourly"), dict) else {}
    times = pd.to_datetime(hourly.get("time", []), errors="coerce")
    if len(times) == 0:
        raise RuntimeError(f"No hourly weather data for {model_id}")
    if getattr(times, "tz", None) is None:
        times = times.tz_localize(tz)
    else:
        times = times.tz_convert(tz)

    missing_vars: list[str] = []

    def _series(name: str, default: float = 0.0) -> pd.Series:
        vals = hourly.get(name)
        if vals is None:
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

    dni = df["dni_wm2"] if "dni_wm2" in df.columns else _series("direct_normal_irradiance", float("nan"))
    dhi = df["dhi_wm2"] if "dhi_wm2" in df.columns else _series("diffuse_radiation", float("nan"))
    df["dni_wm2"] = pd.to_numeric(dni, errors="coerce")
    df["dhi_wm2"] = pd.to_numeric(dhi, errors="coerce")
    derived_irradiance = bool(df[["dni_wm2", "dhi_wm2"]].isna().any(axis=None))
    if derived_irradiance:
        df = _decompose_from_ghi(df, loc, tz)
    df["dni_wm2"] = pd.to_numeric(df["dni_wm2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["dhi_wm2"] = pd.to_numeric(df["dhi_wm2"], errors="coerce").fillna(0.0).clip(lower=0.0)

    availability = pd.DataFrame(index=df.index)
    for col in ["ghi_wm2", "dni_wm2", "dhi_wm2"]:
        availability[col] = pd.to_numeric(df.get(col), errors="coerce").notna()

    df = core.normalize_hourly_forecast_index(
        df[["temp_air_c", "ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_cover_pct", "wind_speed_ms"]],
        target_date,
        tz,
    )
    availability = availability.reindex(df.index).fillna(False)
    for col in ["ghi_wm2", "dni_wm2", "dhi_wm2"]:
        df.loc[~availability[col], col] = np.nan

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
    return forecast, list(set(missing_vars)), bool(derived_irradiance)


def _weighted_ensemble(series_map: dict[str, pd.Series], selected_models: list[str]) -> tuple[pd.Series, dict[str, float] | None]:
    weighted_subset = {m: DEFAULT_WEIGHTED_BELGIUM[m] for m in selected_models if m in DEFAULT_WEIGHTED_BELGIUM}
    if not weighted_subset:
        return pd.concat(series_map.values(), axis=1).mean(axis=1), None
    total = sum(weighted_subset.values())
    normalized = {m: w / total for m, w in weighted_subset.items()}
    matrix = pd.DataFrame({m: series_map[m] for m in normalized})
    weighted_values = matrix.mul(pd.Series(normalized), axis=1)
    numerator = weighted_values.sum(axis=1, skipna=True)
    denominator = matrix.notna().mul(pd.Series(normalized), axis=1).sum(axis=1)
    out = numerator.div(denominator.where(denominator > 0))
    return out.astype(float), normalized


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
    weather_ok: dict[str, core.ForecastResult] = {}

    for model_id in selected:
        try:
            weather, missing_vars, derived_irradiance = fetch_open_meteo_weather(
                model_id,
                loc,
                tz,
                target_date,
                accuracy_mode=accuracy_mode,
                fast_mode=fast_mode,
            )
            model_pv = core.build_pv_forecast(weather.df, loc, tz=tz)
            for req in ["pv_total_kwh", "pv_total_unclipped_kwh", "pv_east_kwh", "pv_south_kwh", "pv_clipped_kwh"]:
                if req not in model_pv.columns:
                    model_pv[req] = np.nan
            pv_total = pd.to_numeric(model_pv["pv_total_kwh"], errors="coerce").clip(lower=0.0)
            pv_unclipped = pd.to_numeric(model_pv["pv_total_unclipped_kwh"], errors="coerce").clip(lower=0.0)
            pv_unclipped = pd.Series(np.maximum(pv_unclipped, pv_total), index=pv_total.index)
            pv_east = pd.to_numeric(model_pv["pv_east_kwh"], errors="coerce").clip(lower=0.0)
            pv_south = pd.to_numeric(model_pv["pv_south_kwh"], errors="coerce").clip(lower=0.0)
            pv_clipped = pd.to_numeric(model_pv["pv_clipped_kwh"], errors="coerce").clip(lower=0.0)

            per_model_pv_columns["pv_total_kwh"][model_id] = pv_total
            per_model_pv_columns["pv_total_unclipped_kwh"][model_id] = pv_unclipped
            per_model_pv_columns["pv_east_kwh"][model_id] = pv_east
            per_model_pv_columns["pv_south_kwh"][model_id] = pv_south
            per_model_pv_columns["pv_clipped_kwh"][model_id] = pv_clipped
            per_model_pv_totals[model_id] = float(pv_total.sum())
            missing_vars_by_model[model_id] = missing_vars
            derived_irradiance_by_model[model_id] = bool(derived_irradiance)
            weather_ok[model_id] = weather
        except WeatherProviderError as exc:
            failed_models.append(model_id)
            failed_model_reasons[model_id] = exc.to_reason()
            print(
                "[weather_ensemble] model_failed "
                f"model={model_id} category={exc.category} status={exc.status} message={exc.message}"
            )
        except Exception as exc:
            failed_models.append(model_id)
            failed_model_reasons[model_id] = {
                "category": "unexpected_error",
                "status": None,
                "message": str(exc),
            }
            print(f"[weather_ensemble] model_failed model={model_id} category=unexpected_error message={exc}")

    if not per_model_pv_columns["pv_total_kwh"]:
        raise RuntimeError("All weather model requests failed.")

    canonical_index = next(iter(weather_ok.values())).df.index

    def _ensemble_column(column_name: str) -> tuple[pd.Series, dict[str, float] | None]:
        model_series = per_model_pv_columns[column_name]
        matrix = pd.concat(model_series.values(), axis=1)
        if ensemble_method == "median":
            return matrix.median(axis=1, skipna=True), None
        if ensemble_method == "mean":
            return matrix.mean(axis=1, skipna=True), None
        model_keys = list(model_series.keys())
        return _weighted_ensemble(model_series, model_keys)

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

    primary_model = next(iter(weather_ok.keys()))
    weather_index = weather_ok[primary_model].df.index
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
        selected_models=list(per_model_pv_columns["pv_total_kwh"].keys()),
        weights_used=weights_used,
        weather_primary_model_id=primary_model,
        weather_by_model=weather_ok,
        weather_ensemble_table=ensemble_weather,
    )
