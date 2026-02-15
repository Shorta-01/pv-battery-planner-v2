import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planner_core import normalize_hourly_forecast_index


@pytest.mark.parametrize(
    ("day", "expected_len"),
    [
        (dt.date(2026, 3, 29), 23),  # spring forward
        (dt.date(2026, 10, 25), 25),  # fall back
        (dt.date(2026, 2, 15), 24),  # normal day
    ],
)
@pytest.mark.parametrize("input_mode", ["aware_utc", "naive_local_like"])
def test_normalize_hourly_forecast_index_dst_lengths(day: dt.date, expected_len: int, input_mode: str) -> None:
    tz = "Europe/Brussels"
    day_start = pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=tz)
    next_day = pd.Timestamp(dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 0)), tz=tz)
    local_index = pd.date_range(day_start, next_day, freq="h", inclusive="left")

    if input_mode == "aware_utc":
        input_index = local_index.tz_convert("UTC")
    else:
        # Simulate local-like naive timestamps, including duplicated local hour on fall-back day.
        input_index = local_index.tz_localize(None)

    df = pd.DataFrame(
        {
            "ghi_wm2": range(len(input_index)),
            "dni_wm2": range(len(input_index)),
            "dhi_wm2": range(len(input_index)),
            "cloud_cover_pct": [25.0] * len(input_index),
            "temp_air_c": [10.0] * len(input_index),
            "wind_speed_ms": [1.0] * len(input_index),
        },
        index=input_index,
    )

    normalized = normalize_hourly_forecast_index(df, day, tz)

    assert len(normalized) == expected_len
    assert str(normalized.index.tz) == tz


def test_normalize_hourly_forecast_index_handles_nonexistent_naive_hour() -> None:
    tz = "Europe/Brussels"
    day = dt.date(2026, 3, 29)  # spring forward in EU

    # Includes local 02:00, which does not exist on this date in Europe/Brussels.
    naive_hours = pd.date_range(
        pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0))),
        periods=24,
        freq="h",
    )

    df = pd.DataFrame(
        {
            "ghi_wm2": range(24),
            "dni_wm2": range(24),
            "dhi_wm2": range(24),
            "cloud_cover_pct": [50.0] * 24,
            "temp_air_c": [12.0] * 24,
            "wind_speed_ms": [2.0] * 24,
        },
        index=naive_hours,
    )

    normalized = normalize_hourly_forecast_index(df, day, tz)

    assert len(normalized) == 23
    assert normalized.index[0] == pd.Timestamp("2026-03-29 00:00:00", tz=tz)
    assert normalized.index[-1] == pd.Timestamp("2026-03-29 23:00:00", tz=tz)
