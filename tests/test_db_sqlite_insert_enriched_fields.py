import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_sqlite import init_db, insert_forecast_run


def test_insert_forecast_run_persists_enriched_fields(tmp_path):
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))

    payload = {
        "run_id": "run-1",
        "target_date": "2026-02-01",
        "run_at_utc": "2026-01-31T23:00:00+00:00",
        "run_type": "manual",
        "status": "degraded",
        "timezone": "Europe/Brussels",
        "warnings": ["w1", "w2"],
        "warnings_count": 2,
        "inputs_used": {
            "soc_at_22_percent": 38.5,
            "yesterday_consumption_kwh": 11.2,
            "ensemble_method": "weighted",
        },
        "weather_ensemble": {
            "selected_models": ["ecmwf_ifs"],
            "weights_used": {"ecmwf_ifs": 1.0},
            "failed_models": [],
        },
        "pv_totals_kwh": {"p10": 1.1, "p50": 2.2, "p90": 3.3},
        "run_duration_ms": 1234,
        "config_schema_version": 3,
        "metrics": {
            "charge_kw": 2.0,
            "cutoff_soc": 0.45,
            "pv_forecast_kwh": 2.2,
            "cons_forecast_kwh": 11.2,
        },
    }

    insert_forecast_run(str(db_path), payload)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT status, warnings_count, inputs_used_json, weather_ensemble_json,
                   pv_p10_kwh, pv_p50_kwh, pv_p90_kwh, run_duration_ms, config_schema_version
            FROM forecast_runs
            WHERE run_id = ?
            """,
            ("run-1",),
        ).fetchone()

    assert row is not None
    assert row["status"] == "degraded"
    assert row["warnings_count"] == 2
    assert json.loads(row["inputs_used_json"])["ensemble_method"] == "weighted"
    assert json.loads(row["weather_ensemble_json"])["selected_models"] == ["ecmwf_ifs"]
    assert row["pv_p10_kwh"] == 1.1
    assert row["pv_p50_kwh"] == 2.2
    assert row["pv_p90_kwh"] == 3.3
    assert row["run_duration_ms"] == 1234
    assert row["config_schema_version"] == 3
