import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_sqlite import init_db, insert_forecast_run


def _frame_payload(index, columns, rows):
    return {
        "columns": columns,
        "index": index,
        "data": rows,
    }


def test_insert_forecast_run_persists_model_hourly_weather_and_pv(tmp_path):
    db_path = tmp_path / "planner.sqlite"
    init_db(str(db_path))

    index = [
        "2026-03-01T00:00:00+01:00",
        "2026-03-01T01:00:00+01:00",
        "2026-03-01T02:00:00+01:00",
    ]
    models = ["ecmwf_ifs", "dwd_icon_d2", "knmi_harmonie_arome"]

    weather_by_model = {}
    pv_by_model = {}
    for offset, model_id in enumerate(models):
        weather_by_model[model_id] = _frame_payload(
            index,
            ["ghi_wm2", "dni_wm2", "dhi_wm2", "temp_air_c", "wind_speed_ms", "cloud_cover_pct"],
            [
                [100.0 + offset, 60.0 + offset, 40.0 + offset, 8.0 + offset, 1.0 + offset, 25.0 + offset],
                [120.0 + offset, 70.0 + offset, 50.0 + offset, 9.0 + offset, 1.2 + offset, 30.0 + offset],
                [90.0 + offset, 50.0 + offset, 35.0 + offset, 7.5 + offset, 0.8 + offset, 35.0 + offset],
            ],
        )
        pv_by_model[model_id] = _frame_payload(
            index,
            ["pv_east_kwh", "pv_south_kwh", "pv_total_kwh", "pv_total_unclipped_kwh", "pv_clipped_kwh"],
            [
                [0.2 + offset, 0.3 + offset, 0.5 + offset, 0.6 + offset, 0.1],
                [0.4 + offset, 0.6 + offset, 1.0 + offset, 1.1 + offset, 0.1],
                [0.1 + offset, 0.2 + offset, 0.3 + offset, 0.35 + offset, 0.05],
            ],
        )

    payload = {
        "run_id": "run-model-hourly-1",
        "target_date": "2026-03-01",
        "run_at_utc": "2026-02-29T23:10:00+00:00",
        "metrics": {"pv_forecast_kwh": 3.0, "cons_forecast_kwh": 10.0},
        "weather_by_model": weather_by_model,
        "pv_by_model": pv_by_model,
    }

    insert_forecast_run(str(db_path), payload)

    with sqlite3.connect(db_path) as conn:
        weather_count = conn.execute(
            "SELECT COUNT(*) FROM weather_hourly_by_model WHERE run_id = ?",
            ("run-model-hourly-1",),
        ).fetchone()[0]
        pv_count = conn.execute(
            "SELECT COUNT(*) FROM pv_hourly_by_model WHERE run_id = ?",
            ("run-model-hourly-1",),
        ).fetchone()[0]
        weather_distinct_pk = conn.execute(
            "SELECT COUNT(DISTINCT model_id || '|' || ts_local) FROM weather_hourly_by_model WHERE run_id = ?",
            ("run-model-hourly-1",),
        ).fetchone()[0]
        pv_distinct_pk = conn.execute(
            "SELECT COUNT(DISTINCT model_id || '|' || ts_local) FROM pv_hourly_by_model WHERE run_id = ?",
            ("run-model-hourly-1",),
        ).fetchone()[0]

    expected = len(models) * len(index)
    assert weather_count == expected
    assert pv_count == expected
    assert weather_distinct_pk == expected
    assert pv_distinct_pk == expected
