import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_chart_functions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    start = source.index("def normalize_detail_df_for_ui")
    end = source.index("def make_chart_surplus")
    snippet = source[start:end]
    namespace: dict = {
        "pd": pd,
        "go": go,
        "PLOTLY_DARK": "plotly_dark",
        "logging": logging,
    }
    exec(snippet, namespace)
    return namespace["normalize_detail_df_for_ui"], namespace["make_chart_pv_load"]


def _extract_soc_trace_y(fig: go.Figure):
    soc_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "Battery SOC")
    return list(soc_trace.y)


def test_make_chart_pv_load_converts_soc_fraction_to_percent() -> None:
    normalize_detail_df_for_ui, make_chart_pv_load = _load_chart_functions()
    idx = pd.date_range("2026-06-01 00:00:00", periods=24, freq="h", tz="Europe/Brussels")
    base_df = pd.DataFrame(
        {
            "pv_total_kwh": [1.0] * 24,
            "pv_east_kwh": [0.4] * 24,
            "pv_south_kwh": [0.6] * 24,
            "load_kwh": [0.8] * 24,
        },
        index=idx,
    )
    soc_fraction = pd.Series([0.65] * 24, index=idx)

    fig = make_chart_pv_load(normalize_detail_df_for_ui(base_df, {"pv": {}}), soc_fraction, 0.2, {"pv": {}})

    soc_y = _extract_soc_trace_y(fig)
    assert max(soc_y) <= 100.0
    assert min(soc_y) >= 0.0
    assert any(v > 1.0 for v in soc_y)
    assert abs(soc_y[0] - 65.0) < 1e-9


def test_make_chart_pv_load_keeps_soc_percent_unchanged() -> None:
    normalize_detail_df_for_ui, make_chart_pv_load = _load_chart_functions()
    idx = pd.date_range("2026-06-01 00:00:00", periods=24, freq="h", tz="Europe/Brussels")
    base_df = pd.DataFrame(
        {
            "pv_total_kwh": [1.2] * 24,
            "pv_east_kwh": [0.5] * 24,
            "pv_south_kwh": [0.7] * 24,
            "load_kwh": [0.9] * 24,
        },
        index=idx,
    )
    soc_percent = pd.Series([65.0 + i * 0.1 for i in range(24)], index=idx)

    fig = make_chart_pv_load(normalize_detail_df_for_ui(base_df, {"pv": {}}), soc_percent, 0.2, {"pv": {}})

    soc_y = _extract_soc_trace_y(fig)
    assert max(soc_y) < 100.0
    assert soc_y[0] == soc_percent.iloc[0]
    assert soc_y[-1] == soc_percent.iloc[-1]


def test_normalize_detail_df_for_ui_sets_synthetic_split_flag_when_missing() -> None:
    normalize_detail_df_for_ui, _ = _load_chart_functions()
    idx = pd.date_range("2026-06-01 00:00:00", periods=24, freq="h", tz="Europe/Brussels")
    raw = pd.DataFrame({"pv_total_kwh": [1.5] * 24, "load_kwh": [0.7] * 24}, index=idx)
    cfg = {"pv": {"array_east_panels": 6, "array_south_panels": 4}}

    normalized = normalize_detail_df_for_ui(raw, cfg)

    assert normalized.attrs.get("synthetic_pv_split_used") is True
    totals = normalized["pv_east_kwh"] + normalized["pv_south_kwh"]
    assert (totals - normalized["pv_total_kwh"]).abs().max() < 1e-9


def test_normalize_detail_df_for_ui_flag_false_when_real_split_present() -> None:
    normalize_detail_df_for_ui, _ = _load_chart_functions()
    idx = pd.date_range("2026-06-01 00:00:00", periods=24, freq="h", tz="Europe/Brussels")
    raw = pd.DataFrame(
        {
            "pv_total_kwh": [1.5] * 24,
            "pv_east_kwh": [0.9] * 24,
            "pv_south_kwh": [0.6] * 24,
            "load_kwh": [0.7] * 24,
        },
        index=idx,
    )
    cfg = {"pv": {"array_east_panels": 6, "array_south_panels": 4}}

    normalized = normalize_detail_df_for_ui(raw, cfg)

    assert normalized.attrs.get("synthetic_pv_split_used") is False
