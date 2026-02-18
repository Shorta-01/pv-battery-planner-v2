import ast
import datetime as dt
from pathlib import Path

import pandas as pd


def _load_builder():
    source = Path("backend_api.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="backend_api.py")
    target = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_build_pv_week_ahead"
    )
    isolated_module = ast.Module(body=[target], type_ignores=[])
    namespace: dict[str, object] = {"pd": pd, "dt": dt}
    exec(compile(isolated_module, filename="backend_api.py", mode="exec"), namespace)
    return namespace["_build_pv_week_ahead"]


def test_week_ahead_weather_icon_prefers_peak_pv_hour_code() -> None:
    build = _load_builder()
    tz = "Europe/Brussels"
    idx = pd.date_range("2026-02-19 00:00", periods=24, freq="h", tz=tz)

    pv_p50 = pd.Series(0.0, index=idx)
    pv_p50.loc[pd.Timestamp("2026-02-19 12:00", tz=tz)] = 2.0

    weather = pd.Series(3.0, index=idx)  # cloudy most of the day
    weather.loc[pd.Timestamp("2026-02-19 12:00", tz=tz)] = 0.0  # sunny at PV peak hour

    week = build(
        target_date=dt.date(2026, 2, 19),
        tz=tz,
        hourly_pv_p50=pv_p50,
        hourly_pv_p10=None,
        hourly_pv_p90=None,
        weather_code_series=weather,
    )

    assert week[0]["weather_code"] == 0


def test_week_ahead_weather_code_falls_back_to_noon_when_no_pv_signal() -> None:
    build = _load_builder()
    tz = "Europe/Brussels"
    idx = pd.date_range("2026-02-19 00:00", periods=24, freq="h", tz=tz)

    pv_p50 = pd.Series(float("nan"), index=idx)
    weather = pd.Series(3.0, index=idx)
    weather.loc[pd.Timestamp("2026-02-19 12:00", tz=tz)] = 61.0

    week = build(
        target_date=dt.date(2026, 2, 19),
        tz=tz,
        hourly_pv_p50=pv_p50,
        hourly_pv_p10=None,
        hourly_pv_p90=None,
        weather_code_series=weather,
    )

    assert week[0]["weather_code"] == 61


def test_week_ahead_weather_code_uses_noon_when_day_has_only_zero_pv() -> None:
    build = _load_builder()
    tz = "Europe/Brussels"
    idx = pd.date_range("2026-02-19 00:00", periods=24, freq="h", tz=tz)

    pv_p50 = pd.Series(0.0, index=idx)
    weather = pd.Series(3.0, index=idx)
    weather.loc[pd.Timestamp("2026-02-19 12:00", tz=tz)] = 61.0

    week = build(
        target_date=dt.date(2026, 2, 19),
        tz=tz,
        hourly_pv_p50=pv_p50,
        hourly_pv_p10=None,
        hourly_pv_p90=None,
        weather_code_series=weather,
    )

    assert week[0]["weather_code"] == 61
