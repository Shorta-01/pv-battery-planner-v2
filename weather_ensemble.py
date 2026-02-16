from __future__ import annotations

import datetime as dt
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

import planner_core as core

DEFAULT_ACCURACY_MODELS = ["knmi_harmonie_arome", "dwd_icon_d2", "ecmwf_ifs"]
DEFAULT_WEIGHTED_BELGIUM = {
    "knmi_harmonie_arome": 0.45,
    "dwd_icon_d2": 0.35,
    "ecmwf_ifs": 0.20,
}

WEATHER_MODELS: dict[str, dict[str, Any]] = {
    "knmi_harmonie_arome": {
        "label": "KNMI HARMONIE-AROME",
        "endpoint": "https://api.open-meteo.com/v1/knmi",
        "params": {},
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
}

_WEATHER_CACHE: dict[tuple, tuple[float, core.ForecastResult, list[str], bool]] = {}
_WEATHER_CACHE_TTL_S = 600


@dataclass
class EnsembleWeatherResult:
    weather_primary: core.ForecastResult
    pv_ensemble_p50: pd.Series
    pv_ensemble_unclipped_p50: pd.Series
    pv_ensemble_clipped_p50: pd.Series
    pv_ensemble_p10: pd.Series | None
    pv_ensemble_p90: pd.Series | None
    per_model_pv_totals_kwh: dict[str, float]
    missing_vars_by_model: dict[str, list[str]]
    derived_irradiance_by_model: dict[str, bool]
    failed_models: list[str]
    selected_models: list[str]
    weights_used: dict[str, float] | None


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


def _request_open_meteo(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected weather payload shape")
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
    df["dni_wm2"] = dni
    df["dhi_wm2"] = (ghi - (dni * cos_zen)).fillna(0.0).clip(lower=0.0)
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
    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "timezone": tz,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timeformat": "iso8601",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "hourly": ",".join([
            "temperature_2m",
            "wind_speed_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "cloud_cover",
        ]),
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
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
        ])

    data = _request_open_meteo(spec["endpoint"], params)

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
    derived_irradiance = False
    if dni.isna().all() or dhi.isna().all():
        df = _decompose_from_ghi(df, loc, tz)
        derived_irradiance = True
    else:
        df["dni_wm2"] = pd.to_numeric(dni, errors="coerce").fillna(0.0).clip(lower=0.0)
        df["dhi_wm2"] = pd.to_numeric(dhi, errors="coerce").fillna(0.0).clip(lower=0.0)

    if model_id == "ecmwf_ifs":
        derived_irradiance = True

    df = core.normalize_hourly_forecast_index(
        df[["temp_air_c", "ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_cover_pct", "wind_speed_ms"]],
        target_date,
        tz,
    )

    daily = data.get("daily") if isinstance(data.get("daily"), dict) else {}
    sunrise = pd.to_datetime((daily.get("sunrise") or [None])[0], errors="coerce")
    sunset = pd.to_datetime((daily.get("sunset") or [None])[0], errors="coerce")
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
    out = sum(series_map[m] * normalized[m] for m in normalized)
    return out.astype(float), normalized


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
    if not selected:
        raise RuntimeError("Select at least one weather model.")

    per_model_pv_ac_series: dict[str, pd.Series] = {}
    per_model_pv_unclipped_series: dict[str, pd.Series] = {}
    per_model_pv_totals: dict[str, float] = {}
    missing_vars_by_model: dict[str, list[str]] = {}
    derived_irradiance_by_model: dict[str, bool] = {}
    failed_models: list[str] = []
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
            model_pv = core.ensure_pv_columns(core.build_pv_forecast(weather.df, loc, tz=tz))
            pv_ac = pd.to_numeric(model_pv["pv_total_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)

            if "pv_dc_available_kwh" in model_pv.columns:
                pv_unclipped = model_pv["pv_dc_available_kwh"]
            elif "pv_total_unclipped_kwh" in model_pv.columns:
                pv_unclipped = model_pv["pv_total_unclipped_kwh"]
            else:
                pv_unclipped = pv_ac

            pv_unclipped = pd.to_numeric(pv_unclipped, errors="coerce").fillna(0.0).clip(lower=0.0)
            pv_unclipped = pv_unclipped.reindex(pv_ac.index).fillna(0.0)
            pv_unclipped = pd.Series(np.maximum(pv_unclipped, pv_ac), index=pv_ac.index)

            per_model_pv_ac_series[model_id] = pv_ac
            per_model_pv_unclipped_series[model_id] = pv_unclipped
            per_model_pv_totals[model_id] = float(pv_ac.sum())
            missing_vars_by_model[model_id] = missing_vars
            derived_irradiance_by_model[model_id] = bool(derived_irradiance)
            weather_ok[model_id] = weather
        except Exception:
            failed_models.append(model_id)

    if not per_model_pv_ac_series:
        raise RuntimeError("All weather model requests failed.")

    aligned_index = next(iter(per_model_pv_ac_series.values())).index
    for model_id in list(per_model_pv_ac_series.keys()):
        per_model_pv_ac_series[model_id] = per_model_pv_ac_series[model_id].reindex(aligned_index).fillna(0.0)
        per_model_pv_unclipped_series[model_id] = per_model_pv_unclipped_series[model_id].reindex(aligned_index).fillna(0.0)

    if ensemble_method == "median":
        ensemble_ac_p50 = pd.concat(per_model_pv_ac_series.values(), axis=1).median(axis=1)
        ensemble_unclipped_p50 = pd.concat(per_model_pv_unclipped_series.values(), axis=1).median(axis=1)
        weights_used = None
    elif ensemble_method == "mean":
        ensemble_ac_p50 = pd.concat(per_model_pv_ac_series.values(), axis=1).mean(axis=1)
        ensemble_unclipped_p50 = pd.concat(per_model_pv_unclipped_series.values(), axis=1).mean(axis=1)
        weights_used = None
    else:
        model_keys = list(per_model_pv_ac_series.keys())
        ensemble_ac_p50, weights_used = _weighted_ensemble(per_model_pv_ac_series, model_keys)
        ensemble_unclipped_p50, _ = _weighted_ensemble(per_model_pv_unclipped_series, model_keys)

    if len(per_model_pv_ac_series) >= 3 and ensemble_method != "median":
        matrix = pd.concat(per_model_pv_ac_series.values(), axis=1)
        spread = (matrix.max(axis=1) - matrix.min(axis=1)).fillna(0.0)
        spread_median = float(spread.median()) if not spread.empty else 0.0
        extreme_mask = spread > max(0.5, 2.0 * spread_median)
        if int(extreme_mask.sum()) >= 3:
            median_ac = matrix.median(axis=1)
            matrix_unclip = pd.concat(per_model_pv_unclipped_series.values(), axis=1)
            median_unclip = matrix_unclip.median(axis=1)
            ensemble_ac_p50.loc[extreme_mask] = median_ac.loc[extreme_mask]
            ensemble_unclipped_p50.loc[extreme_mask] = median_unclip.loc[extreme_mask]

    ensemble_unclipped_p50 = pd.Series(np.maximum(ensemble_unclipped_p50, ensemble_ac_p50), index=ensemble_ac_p50.index)
    ensemble_clipped_p50 = (ensemble_unclipped_p50 - ensemble_ac_p50).clip(lower=0.0)

    p10 = None
    p90 = None
    if pv_uncertainty:
        matrix = pd.concat(per_model_pv_ac_series.values(), axis=1)
        p10 = matrix.quantile(0.10, axis=1)
        p90 = matrix.quantile(0.90, axis=1)

    primary_model = next(iter(weather_ok.keys()))
    return EnsembleWeatherResult(
        weather_primary=weather_ok[primary_model],
        pv_ensemble_p50=ensemble_ac_p50.astype(float),
        pv_ensemble_unclipped_p50=ensemble_unclipped_p50.astype(float),
        pv_ensemble_clipped_p50=ensemble_clipped_p50.astype(float),
        pv_ensemble_p10=p10.astype(float) if p10 is not None else None,
        pv_ensemble_p90=p90.astype(float) if p90 is not None else None,
        per_model_pv_totals_kwh=per_model_pv_totals,
        missing_vars_by_model=missing_vars_by_model,
        derived_irradiance_by_model=derived_irradiance_by_model,
        failed_models=failed_models,
        selected_models=list(per_model_pv_ac_series.keys()),
        weights_used=weights_used,
    )
