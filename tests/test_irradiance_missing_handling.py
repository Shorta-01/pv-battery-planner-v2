import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core
import weather_ensemble as we


def _hourly_index() -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp("2026-01-10 00:00:00", tz="Europe/Brussels"), periods=24, freq="h")


def test_finalize_irradiance_derives_from_ghi_when_dni_dhi_missing() -> None:
    idx = _hourly_index()
    ghi = pd.Series([0.0] * 6 + [200.0] * 10 + [0.0] * 8, index=idx)
    dni = pd.Series(np.nan, index=idx)
    dhi = pd.Series(np.nan, index=idx)

    dni_out, dhi_out, missing_vars, derived = we._finalize_irradiance_components(
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        missing_vars=[],
    )

    assert derived is True
    assert dni_out.notna().any()
    assert dhi_out.notna().any()
    assert "direct_normal_irradiance" not in missing_vars
    assert "diffuse_radiation" not in missing_vars


def test_finalize_irradiance_marks_true_missing_when_ghi_unusable() -> None:
    idx = _hourly_index()
    ghi = pd.Series(np.nan, index=idx)
    dni = pd.Series(np.nan, index=idx)
    dhi = pd.Series(np.nan, index=idx)

    dni_out, dhi_out, missing_vars, derived = we._finalize_irradiance_components(
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        loc=core.Location(name="x", latitude=50.8, longitude=4.3),
        tz="Europe/Brussels",
        missing_vars=[],
    )

    assert derived is False
    assert dni_out.isna().all()
    assert dhi_out.isna().all()
    assert "direct_normal_irradiance" in missing_vars
    assert "diffuse_radiation" in missing_vars
