from __future__ import annotations

import datetime as dt
import math
from typing import Any

import db_sqlite


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
