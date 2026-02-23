from pathlib import Path

import db_sqlite


REQUIRED_COLUMNS = {
    "run_id",
    "score_date",
    "pv_mae_kwh",
    "pv_rmse_kwh",
    "pv_bias_kwh",
    "pv_daily_forecast_kwh",
    "pv_daily_actual_kwh",
    "pv_daily_error_kwh",
    "pv_hourly_points",
    "source",
    "model_scores_json",
    "created_at_utc",
}


def test_daily_scores_schema_and_fetch(tmp_path: Path):
    db_path = tmp_path / "scores.sqlite"
    db_sqlite.init_db(str(db_path))

    with db_sqlite._connect(str(db_path)) as conn:  # noqa: SLF001 - test-only schema check
        cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(daily_scores)").fetchall()}

    assert REQUIRED_COLUMNS.issubset(cols)

    payload = {
        "run_id": "run-1",
        "score_date": "2026-01-02",
        "pv_mae_kwh": 0.12,
        "pv_rmse_kwh": 0.2,
        "pv_bias_kwh": -0.03,
        "pv_daily_forecast_kwh": 8.5,
        "pv_daily_actual_kwh": 8.2,
        "pv_daily_error_kwh": 0.3,
        "pv_hourly_points": 24,
        "source": "manual_csv",
        "model_scores": {"m1": {"pv_mae_kwh": 0.1}},
    }
    db_sqlite.upsert_daily_score(str(db_path), payload)

    fetched = db_sqlite.fetch_daily_score_for_date(str(db_path), "2026-01-02", source="manual_csv")

    assert fetched is not None
    assert fetched["run_id"] == "run-1"
    assert fetched["pv_rmse_kwh"] == 0.2
    assert fetched["pv_daily_actual_kwh"] == 8.2
