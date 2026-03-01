import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_compute_charging_window_all_day_charge_date_transition_to_overnight_target() -> None:
    cfg = {
        "offpeak_windows_by_dow": [
            [["22:00", "07:00"]],  # Monday
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["22:00", "07:00"]],
            [["00:00", "24:00"]],  # Sunday
        ]
    }
    target_date = dt.date(2026, 3, 2)  # Monday

    start, end = core.compute_charging_window_for_target_date(target_date, cfg)

    assert start == core.pd.Timestamp("2026-03-01 22:00", tz=core.TIMEZONE)
    assert end == core.pd.Timestamp("2026-03-02 07:00", tz=core.TIMEZONE)
