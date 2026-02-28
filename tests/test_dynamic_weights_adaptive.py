import datetime as dt
from pathlib import Path

import pytest
import pandas as pd

import db_sqlite
import planner_core as core
import weather_ensemble as we


def _upsert_daily_with_models(db_path: str, run_id: str, score_date: str, model_scores: dict) -> None:
    db_sqlite.upsert_daily_score(
        db_path,
        {
            "run_id": run_id,
            "score_date": score_date,
            "pv_mae_kwh": 0.0,
            "pv_rmse_kwh": 0.0,
            "pv_bias_kwh": 0.0,
            "pv_daily_forecast_kwh": 0.0,
            "pv_daily_actual_kwh": 0.0,
            "pv_daily_error_kwh": 0.0,
            "pv_hourly_points": 24,
            "model_scores": model_scores,
        },
    )


def test_dynamic_weights_prefer_lower_recent_mae(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "planner.sqlite"
    db_sqlite.init_db(str(db_path))

    today = dt.date.today()
    for i in range(3):
        _upsert_daily_with_models(
            str(db_path),
            f"run-{i}",
            (today - dt.timedelta(days=i)).isoformat(),
            {
                "dwd_icon_d2": {"pv_mae_kwh": 2.1, "pv_rmse_kwh": 2.3},
                "ecmwf_ifs": {"pv_mae_kwh": 3.5, "pv_rmse_kwh": 3.8},
            },
        )

    cfg = core.get_effective_config()
    cfg.setdefault("weather", {})
    cfg["weather"]["dynamic_weights"] = {
        "enabled": True,
        "lookback_days": 30,
        "min_days": 2,
        "db_path": str(db_path),
    }
    monkeypatch.setattr(we.core, "get_effective_config", lambda: cfg)

    weights = we._load_dynamic_weights(["dwd_icon_d2", "ecmwf_ifs"])
    assert weights is not None
    assert weights["dwd_icon_d2"] > weights["ecmwf_ifs"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_dynamic_weights_fall_back_to_static_when_insufficient_days(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "planner.sqlite"
    db_sqlite.init_db(str(db_path))

    today = dt.date.today()
    _upsert_daily_with_models(
        str(db_path),
        "run-1",
        today.isoformat(),
        {
            "knmi_harmonie_arome": {"pv_mae_kwh": 0.5, "pv_rmse_kwh": 0.6},
            "dwd_icon_d2": {"pv_mae_kwh": 4.0, "pv_rmse_kwh": 4.2},
        },
    )

    cfg = core.get_effective_config()
    cfg.setdefault("weather", {})
    cfg["weather"]["dynamic_weights"] = {
        "enabled": True,
        "lookback_days": 30,
        "min_days": 10,
        "db_path": str(db_path),
    }
    monkeypatch.setattr(we.core, "get_effective_config", lambda: cfg)

    idx = pd.date_range("2026-01-01", periods=2, freq="h")
    s1 = pd.Series([1.0, 3.0], index=idx)
    s2 = pd.Series([5.0, 7.0], index=idx)

    dynamic = we._load_dynamic_weights(["knmi_harmonie_arome", "dwd_icon_d2"])
    assert dynamic is None
    _, used, _ = we._weighted_ensemble(
        {"knmi_harmonie_arome": s1, "dwd_icon_d2": s2},
        ["knmi_harmonie_arome", "dwd_icon_d2"],
        dynamic_weights=dynamic,
    )
    assert used == {
        "knmi_harmonie_arome": pytest.approx(0.5625),
        "dwd_icon_d2": pytest.approx(0.4375),
    }
