import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_select_week_ahead_models_includes_aifs_excludes_d2() -> None:
    selected = we.select_week_ahead_models(requested_days=7)
    assert "ecmwf_aifs" in selected
    assert "dwd_icon_d2" not in selected


def test_nan_safe_median_preserves_all_nan_hours() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="Europe/Brussels")
    matrix = pd.DataFrame(
        {
            "m1": [np.nan, 1.0],
            "m2": [np.nan, np.nan],
            "m3": [np.nan, 3.0],
        },
        index=idx,
    )
    out = we._nan_safe_hourly_median(matrix)

    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 2.0
