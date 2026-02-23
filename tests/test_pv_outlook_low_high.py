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

    out = we.compute_pv_tomorrow_low_high_kwh(
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

    out = we.compute_pv_tomorrow_low_high_kwh(
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
