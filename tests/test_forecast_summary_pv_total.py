import ast
from pathlib import Path

import pandas as pd


def _load_resolver():
    source = Path("app.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="app.py")
    target = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_forecast_summary_pv_kwh"
    )
    isolated_module = ast.Module(body=[target], type_ignores=[])
    namespace: dict[str, object] = {"pd": pd}
    exec(compile(isolated_module, filename="app.py", mode="exec"), namespace)
    return namespace["resolve_forecast_summary_pv_kwh"]


def test_forecast_summary_pv_uses_tomorrow_value_not_week_sum() -> None:
    resolve_forecast_summary_pv_kwh = _load_resolver()

    pv_week_ahead = [
        {"pv_p50_kwh": 7.8},
        {"pv_p50_kwh": 4.7},
        {"pv_p50_kwh": 0.2},
        {"pv_p50_kwh": 0.3},
        {"pv_p50_kwh": 0.1},
        {"pv_p50_kwh": 0.2},
        {"pv_p50_kwh": 0.4},
    ]
    pv_quality_dict = {"pv_total_kwh": 7.8}

    resolved = resolve_forecast_summary_pv_kwh(
        pv_quality_dict=pv_quality_dict,
        pv_week_ahead=pv_week_ahead,
        pv_df=pd.DataFrame({"pv_total_kwh": [13.7]}),
        result={},
        metrics={},
        weather_ensemble={},
    )

    assert resolved == 7.8
    assert resolved != 13.7


def test_forecast_summary_pv_falls_back_to_week_ahead_day1() -> None:
    resolve_forecast_summary_pv_kwh = _load_resolver()

    pv_week_ahead = [7.8, 4.7, 0.2, 0.3, 0.1, 0.2, 0.4]
    resolved = resolve_forecast_summary_pv_kwh(
        pv_quality_dict={},
        pv_week_ahead=pv_week_ahead,
        pv_df=pd.DataFrame({"pv_total_kwh": [13.7]}),
        result={},
        metrics={},
        weather_ensemble={},
    )

    assert resolved == 7.8
