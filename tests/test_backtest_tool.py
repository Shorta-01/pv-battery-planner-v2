import pytest
import datetime as dt
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db_sqlite
from tools import backtest


class _FakeWeather:
    def __init__(self, idx, ghi_vals):
        self.df = pd.DataFrame(
            {
                "ghi_wm2": ghi_vals,
                "dni_wm2": ghi_vals,
                "dhi_wm2": [0.0 for _ in ghi_vals],
                "temp_air_c": [20.0 for _ in ghi_vals],
                "wind_speed_ms": [1.0 for _ in ghi_vals],
            },
            index=idx,
        )


def test_backtest_deterministic_with_and_without_actuals(tmp_path, monkeypatch):
    db_path = tmp_path / "backtest.sqlite"
    db_sqlite.init_db(str(db_path))

    for day in ["2026-03-01", "2026-03-02"]:
        rows = [
            {"ts_local": f"{day}T10:00:00", "pv_kwh": 0.95, "load_kwh": 0.0, "grid_import_kwh": 0.0, "grid_export_kwh": 0.0, "soc_pct": 50.0},
            {"ts_local": f"{day}T11:00:00", "pv_kwh": 1.95, "load_kwh": 0.0, "grid_import_kwh": 0.0, "grid_export_kwh": 0.0, "soc_pct": 50.0},
            {"ts_local": f"{day}T12:00:00", "pv_kwh": 2.95, "load_kwh": 0.0, "grid_import_kwh": 0.0, "grid_export_kwh": 0.0, "soc_pct": 50.0},
        ]
        db_sqlite.insert_actual_hourly_rows(str(db_path), rows, source="manual_csv")

    def fake_fetch(model_id, loc, tz, target_date, **kwargs):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp(f"{target_date.isoformat()}T10:00:00+01:00"),
                pd.Timestamp(f"{target_date.isoformat()}T11:00:00+01:00"),
                pd.Timestamp(f"{target_date.isoformat()}T12:00:00+01:00"),
            ]
        )
        if model_id == "dwd_icon_d2":
            ghi = [1.0, 2.0, 3.0]
        else:
            ghi = [1.2, 2.2, 3.2]
        return _FakeWeather(idx, ghi), [], False, {"source": "mock"}

    def fake_build_pv(df, loc, tz=None):
        out = df.copy()
        out["pv_total_kwh"] = pd.to_numeric(out["ghi_wm2"], errors="coerce").fillna(0.0)
        out["pv_total_unclipped_kwh"] = out["pv_total_kwh"]
        out["pv_east_kwh"] = out["pv_total_kwh"] * 0.5
        out["pv_south_kwh"] = out["pv_total_kwh"] * 0.5
        out["pv_clipped_kwh"] = 0.0
        return out

    monkeypatch.setattr(backtest.weather_ensemble, "fetch_open_meteo_weather", fake_fetch)
    monkeypatch.setattr(backtest.core, "build_pv_forecast", fake_build_pv)

    report = backtest.run_backtest(
        db_path=str(db_path),
        start_date=dt.date(2026, 3, 1),
        end_date=dt.date(2026, 3, 3),
        models=["icon_d2", "ecmwf_ifs"],
    )

    assert report["processed_days"] == 3
    assert report["scored_days"] == 2
    assert report["best_3_models"][0] == "dwd_icon_d2"
    assert report["summary"]["dwd_icon_d2"]["mae"] == pytest.approx(0.05)
    assert report["summary"]["ecmwf_ifs"]["mae"] == pytest.approx(0.25)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT score_date, model_id, pv_mae_kwh, pv_forecast_kwh FROM backtest_daily_scores ORDER BY score_date, model_id"
        ).fetchall()
    assert len(rows) == 6
    assert rows[0][0] == "2026-03-01"
    assert rows[0][1] == "dwd_icon_d2"
    assert rows[0][2] == pytest.approx(0.05)
    assert rows[-1][0] == "2026-03-03"
    assert rows[-1][2] is None
