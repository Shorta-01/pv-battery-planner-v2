import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planner_core import local_day_hourly_index, normalize_hourly_forecast_index


def _synthetic_hourly_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ghi_wm2": [100.0] * len(index),
            "dni_wm2": [80.0] * len(index),
            "dhi_wm2": [20.0] * len(index),
            "cloud_cover_pct": [40.0] * len(index),
            "temp_air_c": [12.0] * len(index),
            "wind_speed_ms": [2.0] * len(index),
        },
        index=index,
    )


def test_local_day_hourly_index_dst_row_counts() -> None:
    tzname = "Europe/Brussels"

    assert len(local_day_hourly_index(dt.date(2026, 3, 29), tzname)) == 23
    assert len(local_day_hourly_index(dt.date(2026, 10, 25), tzname)) == 25
    assert len(local_day_hourly_index(dt.date(2026, 6, 21), tzname)) == 24


def test_normalization_preserves_dst_aware_row_counts() -> None:
    tzname = "Europe/Brussels"
    for day, expected in [
        (dt.date(2026, 3, 29), 23),
        (dt.date(2026, 10, 25), 25),
        (dt.date(2026, 6, 21), 24),
    ]:
        idx = local_day_hourly_index(day, tzname)
        df = _synthetic_hourly_df(idx)
        normalized = normalize_hourly_forecast_index(df, day, tzname)

        assert len(normalized) == expected
        assert normalized.index.equals(idx)
