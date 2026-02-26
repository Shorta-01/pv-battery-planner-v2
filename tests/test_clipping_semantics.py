import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def _synthetic_sunny_day(tz: str) -> pd.DataFrame:
    idx = pd.date_range(
        pd.Timestamp(dt.datetime(2026, 6, 21, 0, 0), tz=tz),
        periods=24,
        freq="h",
    )
    ghi = [0.0] * 24
    dni = [0.0] * 24
    dhi = [0.0] * 24
    for hour in range(6, 19):
        if hour <= 12:
            ramp = (hour - 6) / 6.0
        else:
            ramp = (18 - hour) / 6.0
        ghi[hour] = 1000.0 * max(0.0, ramp)
        dni[hour] = 900.0 * max(0.0, ramp)
        dhi[hour] = 200.0 * max(0.0, ramp)

    return pd.DataFrame(
        {
            "ghi_wm2": ghi,
            "dni_wm2": dni,
            "dhi_wm2": dhi,
            "cloud_cover_pct": [0.0] * len(idx),
            "temp_air_c": [20.0] * len(idx),
            "wind_speed_ms": [1.0] * len(idx),
        },
        index=idx,
    )


@pytest.mark.skipif(not core.PVLIB_AVAILABLE, reason="pvlib not installed")
def test_forced_low_ac_limit_exposes_clipping_semantics() -> None:
    cfg = core.load_config_file(core.CONFIG_PATH)
    cfg["pv"]["inverter_ac_kw_limit"] = 1.0

    previous = core.get_effective_config()
    try:
        core.apply_config(cfg)
        tz = core.TIMEZONE
        loc = core.Location("x", core.LATITUDE, core.LONGITUDE)
        out = core.build_pv_forecast(_synthetic_sunny_day(tz), loc, tz=tz)
    finally:
        core.apply_config(previous)

    assert float(out["pv_total_unclipped_kw"].max()) > 1.2
    assert float(out["pv_total_kw"].max()) <= 1.0 + 1e-9
    assert float(out["pv_clipped_kwh"].sum()) > 0.1

    unclipped_sum = float(out["pv_total_unclipped_kwh"].sum())
    clipped_sum = float(out["pv_total_kwh"].sum())
    clipped_energy_sum = float(out["pv_clipped_kwh"].sum())
    assert abs(clipped_energy_sum - (unclipped_sum - clipped_sum)) < 0.05


@pytest.mark.skipif(not core.PVLIB_AVAILABLE, reason="pvlib not installed")
def test_default_ac_limit_has_no_false_clipping_when_unclipped_below_limit() -> None:
    cfg = core.load_config_file(core.CONFIG_PATH)

    with core.applied_config(cfg):
        tz = core.TIMEZONE
        loc = core.Location("x", core.LATITUDE, core.LONGITUDE)
        weather = _synthetic_sunny_day(tz)
        weather["ghi_wm2"] *= 0.15
        weather["dni_wm2"] *= 0.15
        weather["dhi_wm2"] *= 0.15
        out = core.build_pv_forecast(weather, loc, tz=tz)

    assert float(out["pv_total_unclipped_kw"].max()) < float(core.INVERTER_AC_KW_LIMIT)
    assert float(out["pv_clipped_kwh"].sum()) == pytest.approx(0.0, abs=1e-9)
