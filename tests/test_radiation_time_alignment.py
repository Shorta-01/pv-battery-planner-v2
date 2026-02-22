import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_aggregate_minutely_15_buckets_to_hour_end() -> None:
    payload = {
        "time": [
            "2026-01-01T09:15",
            "2026-01-01T09:30",
            "2026-01-01T09:45",
            "2026-01-01T10:00",
        ],
        "shortwave_radiation": [100, 100, 100, 100],
        "direct_normal_irradiance": [200, 200, 200, 200],
        "diffuse_radiation": [50, 50, 50, 50],
    }

    out = we._aggregate_minutely_15_to_hourly(payload, tz="Europe/Brussels")

    expected_idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01T10:00", tz="Europe/Brussels")])
    assert out.index.equals(expected_idx)
    assert float(out.loc[expected_idx[0], "ghi_wm2"]) == 100.0
    assert float(out.loc[expected_idx[0], "dni_wm2"]) == 200.0
    assert float(out.loc[expected_idx[0], "dhi_wm2"]) == 50.0


def test_align_backward_hourly_mean_to_hour_start() -> None:
    idx = pd.date_range("2026-01-01T00:00", periods=3, freq="h", tz="Europe/Brussels")
    src = pd.Series([0.0, 10.0, 20.0], index=idx)

    out = we._align_backward_hourly_mean_to_hour_start(src)

    expected = pd.Series([10.0, 20.0, float("nan")], index=idx)
    pd.testing.assert_series_equal(out, expected)
