from __future__ import annotations

import argparse
import datetime as dt
import math
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

import db_sqlite
import planner_core as core
import weather_ensemble


def _normalize_ts_key(ts: Any) -> str:
    parsed = pd.to_datetime(ts, errors="coerce")
    if pd.isna(parsed):
        return str(ts)
    if isinstance(parsed, pd.Timestamp) and parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.isoformat()


def _date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    days = (end - start).days
    if days < 0:
        raise ValueError("start date must be <= end date")
    return [start + dt.timedelta(days=i) for i in range(days + 1)]


def _series_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        ts = str(row.get("ts_local") or "")
        val = row.get("pv_kwh")
        if ts and val is not None:
            out[_normalize_ts_key(ts)] = float(val)
    return out


def _compute_metrics(forecast: dict[str, float], actual: dict[str, float]) -> dict[str, float | int]:
    joined = sorted(set(forecast.keys()) & set(actual.keys()))
    if not joined:
        raise ValueError("No overlapping forecast and actual rows")

    errors = [forecast[ts] - actual[ts] for ts in joined]
    mae = sum(abs(err) for err in errors) / len(errors)
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
    bias = sum(errors) / len(errors)

    forecast_daily = sum(forecast[ts] for ts in joined)
    actual_daily = sum(actual[ts] for ts in joined)
    return {
        "pv_mae_kwh": mae,
        "pv_rmse_kwh": rmse,
        "pv_bias_kwh": bias,
        "pv_forecast_kwh": forecast_daily,
        "pv_actual_kwh": actual_daily,
        "pv_daily_error_kwh": forecast_daily - actual_daily,
        "pv_hourly_points": len(joined),
    }


def run_backtest(
    *,
    db_path: str,
    start_date: dt.date,
    end_date: dt.date,
    models: list[str],
    location_name: str = "Halle BE",
    latitude: float = 50.7339,
    longitude: float = 4.2343,
    timezone: str = "Europe/Brussels",
    actual_source: str = "manual_csv",
) -> dict[str, Any]:
    db_sqlite.init_db(db_path)
    loc = core.Location(location_name, latitude, longitude)

    days = _date_range(start_date, end_date)
    canonical_models = [weather_ensemble.normalize_weather_model_id(m) for m in models]

    processed_days = 0
    scored_days = 0
    best_model_counts = Counter()
    aggregate = defaultdict(list)

    for day in days:
        processed_days += 1
        actual_rows = db_sqlite.fetch_actual_pv_hourly_for_date(db_path, day.isoformat(), source=actual_source)
        actual_series = _series_from_rows(actual_rows)

        day_scored: dict[str, float] = {}

        for model_id in canonical_models:
            endpoint, extra_params = weather_ensemble.historical_forecast_params(model_id)
            weather, _missing, _derived, _meta = weather_ensemble.fetch_open_meteo_weather(
                model_id=model_id,
                loc=loc,
                tz=timezone,
                target_date=day,
                accuracy_mode=True,
                fast_mode=False,
                endpoint_override=endpoint,
                extra_params=extra_params,
            )
            pv_df = core.build_pv_forecast(weather.df, loc, tz=timezone)

            forecast_series: dict[str, float] = {}
            for ts, value in pd.to_numeric(pv_df.get("pv_total_kwh"), errors="coerce").fillna(0.0).items():
                forecast_series[_normalize_ts_key(ts)] = float(value)

            forecast_daily = float(sum(forecast_series.values()))
            payload = {
                "score_date": day.isoformat(),
                "model_id": model_id,
                "source": "backtest",
                "pv_forecast_kwh": forecast_daily,
            }

            if actual_series:
                metrics = _compute_metrics(forecast_series, actual_series)
                payload.update(metrics)
                day_scored[model_id] = float(metrics["pv_mae_kwh"])
                aggregate[model_id].append(metrics)

            db_sqlite.upsert_backtest_daily_score(db_path, payload)

        if day_scored:
            scored_days += 1
            best_model_counts[min(day_scored, key=day_scored.get)] += 1

    model_summary: dict[str, dict[str, float]] = {}
    for model_id in canonical_models:
        rows = aggregate.get(model_id, [])
        if not rows:
            model_summary[model_id] = {
                "days": 0.0,
                "mae": float("nan"),
                "bias": float("nan"),
                "best_model_frequency": 0.0,
            }
            continue
        mae = sum(float(r["pv_mae_kwh"]) for r in rows) / len(rows)
        bias = sum(float(r["pv_bias_kwh"]) for r in rows) / len(rows)
        best_freq = best_model_counts.get(model_id, 0) / max(scored_days, 1)
        model_summary[model_id] = {
            "days": float(len(rows)),
            "mae": mae,
            "bias": bias,
            "best_model_frequency": best_freq,
        }

    top_models = sorted(
        [m for m in canonical_models if model_summary[m]["days"] > 0],
        key=lambda m: (model_summary[m]["mae"], -model_summary[m]["best_model_frequency"]),
    )[:3]

    return {
        "location": location_name,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "processed_days": processed_days,
        "scored_days": scored_days,
        "models": canonical_models,
        "summary": model_summary,
        "best_3_models": top_models,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest weather models with Open-Meteo historical forecast API")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--models", nargs="+", required=True, help="Model IDs (e.g. icon_d2 ecmwf_ifs)")
    parser.add_argument("--db-path", default="local_state/planner_history.sqlite")
    parser.add_argument("--location-name", default="Halle BE")
    parser.add_argument("--lat", type=float, default=50.7339)
    parser.add_argument("--lon", type=float, default=4.2343)
    parser.add_argument("--timezone", default="Europe/Brussels")
    parser.add_argument("--actual-source", default="manual_csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

    report = run_backtest(
        db_path=args.db_path,
        start_date=start_date,
        end_date=end_date,
        models=args.models,
        location_name=args.location_name,
        latitude=float(args.lat),
        longitude=float(args.lon),
        timezone=args.timezone,
        actual_source=args.actual_source,
    )

    print(f"Backtest {report['start_date']}..{report['end_date']} for {report['location']}")
    print(f"Processed days: {report['processed_days']} | Scored days: {report['scored_days']}")
    print("Per-model summary:")
    for model_id in report["models"]:
        item = report["summary"][model_id]
        mae_txt = f"{item['mae']:.4f}" if not math.isnan(item["mae"]) else "n/a"
        bias_txt = f"{item['bias']:.4f}" if not math.isnan(item["bias"]) else "n/a"
        print(
            f"- {model_id}: days={int(item['days'])} mae={mae_txt} bias={bias_txt} "
            f"best_freq={item['best_model_frequency']:.2%}"
        )
    if report["best_3_models"]:
        print("Best 3 models for Halle BE: " + ", ".join(report["best_3_models"]))
    else:
        print("Best 3 models for Halle BE: n/a (no scored days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
