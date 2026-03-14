import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we
import ast



TZ = "Europe/Brussels"
TARGET_DATE = dt.date(2026, 1, 10)


def _load_app_resolver():
    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "app.py"
    try:
        source = app_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback: still allow parsing even if a stray byte exists
        source = app_path.read_text(encoding="utf-8", errors="replace")
    module = ast.parse(source)
    fn_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_tomorrow_pv_low_high_kwh"
    )
    isolated_module = ast.Module(body=[fn_node], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {"pd": pd}
    exec(compile(isolated_module, filename="app.py", mode="exec"), namespace)
    return namespace["resolve_tomorrow_pv_low_high_kwh"]


RESOLVE_TOMORROW_PV_LOW_HIGH_KWH = _load_app_resolver()


def _hourly_df(start: str, periods: int, value: float) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start, tz=TZ), periods=periods, freq="h")
    return pd.DataFrame({"pv_total_kwh": [value] * periods}, index=idx)


def test_tomorrow_low_high_uses_tomorrow_window_not_horizon_totals() -> None:
    model_a = _hourly_df("2026-01-10 00:00:00", 48, 0.1)  # tomorrow total 2.4
    model_b = _hourly_df("2026-01-10 00:00:00", 168, 0.3)  # tomorrow total 7.2

    out = we.compute_pv_tomorrow_model_spread_kwh(
        pv_by_model={"a": model_a, "b": model_b},
        target_date=TARGET_DATE,
        tz=TZ,
    )

    assert out["valid_models"] == 2
    assert out["low"] == pytest.approx(2.4)
    assert out["high"] == pytest.approx(7.2)


def test_tomorrow_low_high_excludes_partial_day_models() -> None:
    model_a = _hourly_df("2026-01-10 00:00:00", 6, 0.2)
    model_b = _hourly_df("2026-01-10 00:00:00", 24, 0.25)

    out = we.compute_pv_tomorrow_model_spread_kwh(
        pv_by_model={"a": model_a, "b": model_b},
        target_date=TARGET_DATE,
        tz=TZ,
        min_hours=18,
    )

    assert out["valid_models"] == 1
    assert out["low"] is None
    assert out["high"] is None


def test_ui_resolver_rejects_non_bracketing_ranges() -> None:
    result = {
        "pv_tomorrow_low_high_kwh": {
            "low": 9.69,
            "high": 59.12,
            "valid_models": 2,
        }
    }

    low, high = RESOLVE_TOMORROW_PV_LOW_HIGH_KWH(result, weather_ensemble={}, tomorrow_p50_kwh=5.8)

    assert low is None
    assert high is None



def test_unified_low_high_matches_p10_p90_from_quantiles() -> None:
    idx = pd.date_range(pd.Timestamp("2026-01-10 00:00:00", tz=TZ), periods=24, freq="h")
    matrix = pd.DataFrame(
        {
            "a": [0.10] * 24,
            "b": [0.20] * 24,
            "c": [0.40] * 24,
        },
        index=idx,
    )
    p10 = float(matrix.quantile(0.10, axis=1).sum())
    p50 = float(matrix.quantile(0.50, axis=1).sum())
    p90 = float(matrix.quantile(0.90, axis=1).sum())

    result = {
        "pv_totals_kwh": {"p10": p10, "p50": p50, "p90": p90},
        "pv_tomorrow_low_high_kwh": {"low": p10, "high": p90, "valid_models": 3},
    }

    low, high = RESOLVE_TOMORROW_PV_LOW_HIGH_KWH(result, weather_ensemble={}, tomorrow_p50_kwh=p50)

    assert result["pv_totals_kwh"]["p10"] is not None
    assert result["pv_totals_kwh"]["p50"] is not None
    assert result["pv_totals_kwh"]["p90"] is not None
    assert low == pytest.approx(result["pv_totals_kwh"]["p10"])
    assert high == pytest.approx(result["pv_totals_kwh"]["p90"])
    assert result["pv_tomorrow_low_high_kwh"]["valid_models"] >= 2


def test_pv_outlook_headline_not_weather_wording() -> None:
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8", errors="replace")
    assert "day · {pv_quality_dict['pv_total_kwh']:.1f} kWh" not in app_source
    assert "PV outlook · {pv_quality_dict['pv_total_kwh']:.1f} kWh" in app_source
