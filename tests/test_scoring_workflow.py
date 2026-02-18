import math

import pytest
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_sqlite import init_db, insert_actual_hourly_rows, insert_forecast_run
import scoring


def _frame_payload(index, columns, rows):
    return {
        "columns": columns,
        "index": index,
        "data": rows,
    }


def test_score_day_writes_daily_scores_and_exact_metrics(tmp_path):
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))

    index = [
        "2026-03-01T10:00:00+01:00",
        "2026-03-01T11:00:00+01:00",
        "2026-03-01T12:00:00+01:00",
    ]

    payload = {
        "run_id": "run-score-1",
        "target_date": "2026-03-01",
        "run_at_utc": "2026-02-29T23:00:00+00:00",
        "metrics": {"pv_forecast_kwh": 15.1, "cons_forecast_kwh": 10.0},
        "pv": _frame_payload(
            index,
            ["pv_total_kwh", "pv_total_unclipped_kwh", "pv_east_kwh", "pv_south_kwh", "pv_clipped_kwh", "load_kwh"],
            [
                [4.0, 4.0, 2.0, 2.0, 0.0, 0.0],
                [5.1, 5.1, 2.5, 2.6, 0.0, 0.0],
                [6.0, 6.0, 3.0, 3.0, 0.0, 0.0],
            ],
        ),
        "flows": _frame_payload(
            index,
            ["grid_import_kwh", "grid_export_kwh", "batt_charge_kwh", "batt_discharge_kwh", "soc_end_pct"],
            [
                [0.0, 0.0, 0.0, 0.0, 50.0],
                [0.0, 0.0, 0.0, 0.0, 50.0],
                [0.0, 0.0, 0.0, 0.0, 50.0],
            ],
        ),
        "pv_by_model": {
            "ecmwf_ifs": _frame_payload(
                index,
                ["pv_east_kwh", "pv_south_kwh", "pv_total_kwh", "pv_total_unclipped_kwh", "pv_clipped_kwh"],
                [
                    [1.8, 2.0, 3.8, 3.8, 0.0],
                    [2.4, 2.5, 4.9, 4.9, 0.0],
                    [3.0, 2.9, 5.9, 5.9, 0.0],
                ],
            )
        },
    }
    insert_forecast_run(str(db_path), payload)

    inserted = insert_actual_hourly_rows(
        str(db_path),
        [
            {
                "ts_local": "2026-03-01T10:00:00",
                "pv_kwh": 3.0,
                "load_kwh": 0.0,
                "grid_import_kwh": 0.0,
                "grid_export_kwh": 0.0,
                "soc_pct": 50.0,
            },
            {
                "ts_local": "2026-03-01T11:00:00",
                "pv_kwh": 4.3,
                "load_kwh": 0.0,
                "grid_import_kwh": 0.0,
                "grid_export_kwh": 0.0,
                "soc_pct": 50.0,
            },
            {
                "ts_local": "2026-03-01T12:00:00",
                "pv_kwh": 5.0,
                "load_kwh": 0.0,
                "grid_import_kwh": 0.0,
                "grid_export_kwh": 0.0,
                "soc_pct": 50.0,
            },
        ],
        source="manual_csv",
    )
    assert inserted == 3

    result = scoring.score_day(str(db_path), "2026-03-01", source="manual_csv")

    errors = [1.0, 0.8, 1.0]
    expected_mae = sum(abs(e) for e in errors) / 3.0
    expected_rmse = math.sqrt(sum(e * e for e in errors) / 3.0)
    expected_bias = sum(errors) / 3.0

    assert result["run_id"] == "run-score-1"
    assert result["pv_mae_kwh"] == pytest.approx(expected_mae)
    assert result["pv_rmse_kwh"] == pytest.approx(expected_rmse)
    assert result["pv_bias_kwh"] == pytest.approx(expected_bias)
    assert result["pv_daily_forecast_kwh"] == 15.1
    assert result["pv_daily_actual_kwh"] == 12.3
    assert result["pv_daily_error_kwh"] == pytest.approx(2.8)
    assert result["pv_hourly_points"] == 3
    assert "ecmwf_ifs" in result["model_scores"]

    with sqlite3.connect(db_path) as conn:
        stored = conn.execute(
            "SELECT run_id, score_date, pv_mae_kwh, pv_rmse_kwh, pv_bias_kwh, pv_daily_error_kwh FROM daily_scores"
        ).fetchall()

    assert len(stored) == 1
    assert stored[0][0] == "run-score-1"
    assert stored[0][1] == "2026-03-01"
    assert stored[0][2] == pytest.approx(expected_mae)
    assert stored[0][3] == pytest.approx(expected_rmse)
    assert stored[0][4] == pytest.approx(expected_bias)
    assert stored[0][5] == pytest.approx(2.8)
