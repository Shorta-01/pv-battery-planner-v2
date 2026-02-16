import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def _load_normalizer():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    start = source.index("def normalize_detail_df_for_ui")
    end = source.index("def make_chart_pv_load")
    snippet = source[start:end]
    namespace: dict = {"pd": pd}
    exec(snippet, namespace)
    return namespace["normalize_detail_df_for_ui"]


def test_normalize_detail_df_for_ui_backfills_required_columns() -> None:
    normalize_detail_df_for_ui = _load_normalizer()
    idx = pd.date_range("2026-06-01 00:00:00", periods=3, freq="h", tz="Europe/Brussels")
    raw = pd.DataFrame(
        {
            "pv_total_kwh": [0.0, 1.2, 2.0],
            "load_kwh": [0.4, 0.5, 0.8],
            "surplus_kwh": [0.0, 0.7, 1.2],
            "deficit_kwh": [0.4, 0.0, 0.0],
        },
        index=idx,
    )

    cfg = {
        "pv": {
            "array_east_panels": 7,
            "array_south_panels": 11,
        }
    }

    normalized = normalize_detail_df_for_ui(raw, cfg)

    required_cols = {
        "pv_east_kwh",
        "pv_south_kwh",
        "pv_total_kwh",
        "pv_total_unclipped_kwh",
        "pv_clipped_kwh",
        "pv_surplus_kwh",
        "pv_deficit_kwh",
    }
    assert required_cols.issubset(set(normalized.columns))
    assert (normalized["pv_total_unclipped_kwh"] >= normalized["pv_total_kwh"]).all()


def test_simulate_expensive_hours_detailed_outputs_gui_columns() -> None:
    target_date = dt.date(2026, 1, 5)  # Monday => has expensive windows
    idx = pd.date_range(
        pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=core.TIMEZONE),
        periods=24,
        freq="h",
    )
    df = pd.DataFrame(
        {
            "pv_total_kwh": [0.0] * 24,
            "pv_total_unclipped_kwh": [0.0] * 24,
            "pv_east_kwh": [0.0] * 24,
            "pv_south_kwh": [0.0] * 24,
            "pv_clipped_kwh": [0.0] * 24,
        },
        index=idx,
    )

    detail_df, *_ = core.simulate_expensive_hours_detailed(
        df=df,
        total_consumption_kwh=18.0,
        start_soc=0.5,
        for_date=target_date,
    )

    required_cols = {
        "pv_east_kwh",
        "pv_south_kwh",
        "pv_total_kwh",
        "pv_total_unclipped_kwh",
        "pv_clipped_kwh",
        "load_kwh",
        "pv_surplus_kwh",
        "pv_deficit_kwh",
    }
    assert required_cols.issubset(set(detail_df.columns))
