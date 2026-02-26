import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weather_ensemble import local_day_hourly_index


def test_local_day_hourly_index_dst_row_counts_and_tz_awareness() -> None:
    tzname = "Europe/Brussels"

    spring = local_day_hourly_index(dt.date(2026, 3, 29), tzname)
    fall = local_day_hourly_index(dt.date(2026, 10, 25), tzname)
    normal = local_day_hourly_index(dt.date(2026, 6, 21), tzname)

    assert spring.tz is not None
    assert fall.tz is not None
    assert normal.tz is not None

    assert len(spring) == 23
    assert len(fall) == 25
    assert len(normal) == 24


def test_local_day_hourly_index_keeps_both_fall_back_2am_hours() -> None:
    idx = local_day_hourly_index(dt.date(2026, 10, 25), "Europe/Brussels")

    two_am = idx[idx.hour == 2]
    assert len(two_am) == 2
    assert two_am[0].utcoffset() != two_am[1].utcoffset()
