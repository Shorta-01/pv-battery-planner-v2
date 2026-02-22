from __future__ import annotations

import datetime as dt
import math
from typing import Any, TYPE_CHECKING

import pandas as pd

import db_sqlite
import planner_core as core

if TYPE_CHECKING:
    from planner_core import Location


def compute_pv_quality_score(
    pv_df,
    weather_df,
    target_date,
    tz: str,
    fallback_score: int = 55,
    loc: "Location | None" = None,
) -> dict[str, Any]:
    """Compatibility helper for backend_api run payload quality metadata."""
    pv_total = 0.0
    if hasattr(pv_df, "get"):
        try:
            pv_total = float(pd.to_numeric(pv_df.get("pv_total_kwh", 0.0), errors="coerce").sum(min_count=1) or 0.0)
        except Exception:
            pv_total = 0.0

    def _bucket(score: int) -> tuple[str, str]:
        if score >= 85:
            return "Excellent", "#22c55e"
        if score >= 70:
            return "Good", "#84cc16"
        if score >= 50:
            return "Mixed", "#f59e0b"
        if score >= 30:
            return "Poor", "#f97316"
        return "Very low", "#ef4444"

    try:
        if loc is not None and hasattr(pv_df, "index") and getattr(pv_df.index, "tz", None) is not None:
            idx = pv_df.index
            clear_df = pd.DataFrame(index=idx)
            if hasattr(weather_df, "reindex"):
                clear_df["temp_air_c"] = pd.to_numeric(weather_df.reindex(idx).get("temp_air_c"), errors="coerce").fillna(10.0)
                clear_df["wind_speed_ms"] = pd.to_numeric(weather_df.reindex(idx).get("wind_speed_ms"), errors="coerce").fillna(1.0).clip(lower=0.0)
            else:
                clear_df["temp_air_c"] = 10.0
                clear_df["wind_speed_ms"] = 1.0
            clear_df["cloud_cover_pct"] = 0.0
            clear_pv = core.build_pv_forecast(clear_df, loc, tz=tz)
            clear_kwh = float(pd.to_numeric(clear_pv.get("pv_ac_limited_kwh", clear_pv.get("pv_total_kwh", 0.0)), errors="coerce").sum(min_count=1) or 0.0)
            ratio = max(0.0, min(pv_total / max(clear_kwh, 0.1), 1.0))
            score = int(max(0, min(100, round(100.0 * ratio))))
            label, color = _bucket(score)
            return {
                "score": score,
                "label": label,
                "ratio": float(ratio),
                "color": color,
                "is_fallback": False,
                "pv_total_kwh": pv_total,
                "clear_sky_kwh": clear_kwh,
                "target_date": str(target_date),
                "timezone": tz,
            }
    except Exception:
        pass

    score = int(max(0, min(100, fallback_score)))
    if pv_total > 0:
        score = int(max(score, min(100, round(40 + min(pv_total, 15.0) * 4))))
    label, color = _bucket(score)
    return {
        "score": score,
        "label": label,
        "ratio": None,
        "color": color,
        "is_fallback": True,
        "pv_total_kwh": pv_total,
        "target_date": str(target_date),
        "timezone": tz,
    }


def _series_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    series: dict[str, float] = {}
    for row in rows:
        ts_local = str(row.get("ts_local") or "")
        if not ts_local:
            continue
        value = row.get("pv_kwh")
        if value is None:
            continue
        series[ts_local] = float(value)
    return series


def _compute_metrics(forecast: dict[str, float], actual: dict[str, float]) -> dict[str, float | int]:
    joined = sorted(set(forecast.keys()) & set(actual.keys()))
    if not joined:
        raise ValueError("No overlapping forecast and actual rows for this day")

    errors = [forecast[ts] - actual[ts] for ts in joined]
    abs_errors = [abs(err) for err in errors]

    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
    bias = sum(errors) / len(errors)

    forecast_daily = sum(forecast[ts] for ts in joined)
    actual_daily = sum(actual[ts] for ts in joined)

    return {
        "pv_mae_kwh": mae,
        "pv_rmse_kwh": rmse,
        "pv_bias_kwh": bias,
        "pv_daily_forecast_kwh": forecast_daily,
        "pv_daily_actual_kwh": actual_daily,
        "pv_daily_error_kwh": forecast_daily - actual_daily,
        "pv_hourly_points": len(joined),
    }


def score_day(db_path: str, score_date: str, source: str = "manual_csv") -> dict[str, Any]:
    run_id = db_sqlite.fetch_latest_run_id_for_date(db_path, score_date)
    if not run_id:
        raise ValueError(f"No forecast run found for date {score_date}")

    forecast_rows = db_sqlite.fetch_forecast_pv_hourly(db_path, run_id)
    actual_rows = db_sqlite.fetch_actual_pv_hourly_for_date(db_path, score_date, source=source)
    forecast_series = _series_from_rows(forecast_rows)
    actual_series = _series_from_rows(actual_rows)
    ensemble = _compute_metrics(forecast_series, actual_series)

    per_model = db_sqlite.fetch_forecast_pv_hourly_by_model(db_path, run_id)
    per_model_scores: dict[str, dict[str, float | int]] = {}
    for model_id, rows in per_model.items():
        model_series = _series_from_rows(rows)
        try:
            per_model_scores[model_id] = _compute_metrics(model_series, actual_series)
        except ValueError:
            continue

    out = {
        "run_id": run_id,
        "score_date": score_date,
        "source": source,
        **ensemble,
        "model_scores": per_model_scores,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    db_sqlite.upsert_daily_score(db_path, out)
    return out
