import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_sqlite import fetch_full_run_by_id, fetch_history_all_runs, fetch_history_latest_per_day, init_db, insert_forecast_run


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


def test_insert_forecast_run_defaults_config_schema_version_when_missing(tmp_path):
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))

    payload = {
        "run_id": "run-default-schema",
        "target_date": "2026-02-03",
        "run_at_utc": "2026-02-02T23:00:00+00:00",
        "metrics": {
            "pv_forecast_kwh": 1.0,
            "cons_forecast_kwh": 2.0,
        },
    }

    insert_forecast_run(str(db_path), payload)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT config_schema_version FROM forecast_runs WHERE run_id = ?",
            ("run-default-schema",),
        ).fetchone()

    assert row is not None
    assert row["config_schema_version"] == 1

def test_fetch_history_includes_run_id_and_summary_fields(tmp_path):
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))

    payload_old = {
        "run_id": "run-old",
        "target_date": "2026-02-01",
        "run_at_utc": "2026-01-31T22:00:00+00:00",
        "run_type": "manual",
        "status": "ok",
        "warnings_count": 1,
        "weather_ensemble": {
            "selected_models": ["ecmwf_ifs", "gfs"],
            "weights_used": {"ecmwf_ifs": 0.6, "gfs": 0.4},
            "failed_models": ["icon"],
        },
        "pv_totals_kwh": {"p10": 1.0, "p50": 2.0, "p90": 3.0},
        "metrics": {
            "charge_kw": 2.5,
            "cutoff_soc": 0.4,
            "pv_forecast_kwh": 2.0,
            "cons_forecast_kwh": 10.0,
        },
    }
    payload_new = {
        "run_id": "run-new",
        "target_date": "2026-02-01",
        "run_at_utc": "2026-01-31T23:00:00+00:00",
        "run_type": "nightly",
        "status": "degraded",
        "warnings_count": 3,
        "weather_ensemble": {
            "selected_models": ["ecmwf_ifs"],
            "weights_used": {"ecmwf_ifs": 1.0},
            "failed_models": [],
        },
        "pv_totals_kwh": {"p10": 1.1, "p50": 2.2, "p90": 3.3},
        "metrics": {
            "charge_kw": 3.0,
            "cutoff_soc": 0.5,
            "pv_forecast_kwh": 2.2,
            "cons_forecast_kwh": 11.0,
        },
    }

    insert_forecast_run(str(db_path), payload_old)
    insert_forecast_run(str(db_path), payload_new)

    latest_per_day = fetch_history_latest_per_day(str(db_path))
    assert len(latest_per_day) == 1
    latest = latest_per_day[0]
    assert latest["run_id"] == "run-new"
    assert latest["status"] == "degraded"
    assert latest["warnings_count"] == 3
    assert latest["pv_p10_kwh"] == 1.1
    assert latest["pv_p50_kwh"] == 2.2
    assert latest["pv_p90_kwh"] == 3.3
    assert latest["models_summary"] == {
        "selected_models": ["ecmwf_ifs"],
        "weights_used": {"ecmwf_ifs": 1.0},
        "failed_models": [],
    }

    all_runs = fetch_history_all_runs(str(db_path))
    assert [row["run_id"] for row in all_runs] == ["run-old", "run-new"]
    assert all_runs[0]["models_summary"] == {
        "selected_models": ["ecmwf_ifs", "gfs"],
        "weights_used": {"ecmwf_ifs": 0.6, "gfs": 0.4},
        "failed_models": ["icon"],
    }


def test_fetch_full_run_by_id_returns_stored_run_details(tmp_path):
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))

    payload = {
        "run_id": "run-full",
        "target_date": "2026-02-02",
        "run_at_utc": "2026-02-01T22:00:00+00:00",
        "run_type": "manual",
        "status": "ok",
        "warnings_count": 1,
        "warnings": ["check-1"],
        "inputs_used": {
            "ensemble_method": "weighted",
            "weather_models_selected": ["ecmwf_ifs"],
        },
        "config_hash": "cfg-hash-123",
        "config": {"reserve_soc": 20, "timezone": "Europe/Brussels"},
        "weather_ensemble": {
            "selected_models": ["ecmwf_ifs"],
            "failed_models": ["gfs"],
            "missing_variables_by_model": {"gfs": ["dni"]},
            "derived_irradiance_by_model": {"ecmwf_ifs": False, "gfs": True},
            "weights_used": {"ecmwf_ifs": 1.0},
        },
        "pv_totals_kwh": {"p10": 6.8, "p50": 8.0, "p90": 9.4},
        "metrics": {
            "charge_kw": 3.5,
            "cutoff_soc": 0.5,
            "pv_forecast_kwh": 8.0,
            "cons_forecast_kwh": 12.0,
        },
        "pv": {
            "columns": [
                "pv_total_kwh",
                "pv_total_unclipped_kwh",
                "pv_east_kwh",
                "pv_south_kwh",
                "pv_clipped_kwh",
                "load_kwh",
            ],
            "index": ["2026-02-02T00:00:00"],
            "data": [[1.2, 1.3, 0.4, 0.8, 0.1, 0.6]],
        },
        "flows": {
            "columns": ["grid_import_kwh", "grid_export_kwh", "batt_charge_kwh", "batt_discharge_kwh", "soc_end_pct"],
            "index": ["2026-02-02T00:00:00"],
            "data": [[0.2, 0.1, 0.3, 0.0, 55.0]],
        },
        "soc": {
            "columns": ["value"],
            "index": ["2026-02-02T00:00:00"],
            "data": [[0.55]],
        },
    }

    insert_forecast_run(str(db_path), payload)

    full = fetch_full_run_by_id(str(db_path), "run-full")
    assert full is not None
    assert full["run_id"] == "run-full"
    assert full["target_date"] == "2026-02-02"
    assert full["run_at_utc"] == "2026-02-01T22:00:00+00:00"
    assert full["run_type"] == "manual"
    assert full["status"] == "ok"
    assert full["warnings_count"] == 1
    assert full["metrics"]["charge_kw"] == 3.5
    assert full["metrics"]["cutoff_soc"] == 0.5
    assert full["warnings"] == ["check-1"]
    assert full["inputs_used"] == {
        "ensemble_method": "weighted",
        "weather_models_selected": ["ecmwf_ifs"],
    }
    assert full["config_hash"] == "cfg-hash-123"
    assert full["config_json"] == {"reserve_soc": 20, "timezone": "Europe/Brussels"}
    assert full["weather_ensemble"] == {
        "selected_models": ["ecmwf_ifs"],
        "failed_models": ["gfs"],
        "missing_variables_by_model": {"gfs": ["dni"]},
        "derived_irradiance_by_model": {"ecmwf_ifs": False, "gfs": True},
        "weights_used": {"ecmwf_ifs": 1.0},
    }
    assert full["pv_totals_kwh"] == {"p10": 6.8, "p50": 8.0, "p90": 9.4}
    assert full["pv"]["columns"] == [
        "pv_total_kwh",
        "pv_total_unclipped_kwh",
        "pv_east_kwh",
        "pv_south_kwh",
        "pv_clipped_kwh",
        "load_kwh",
    ]
    assert len(full["pv"]["data"]) == 1
    assert len(full["flows"]["data"]) == 1
    assert len(full["soc"]["data"]) == 1

    assert fetch_full_run_by_id(str(db_path), "does-not-exist") is None
