import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_cycle_load_is_normalized_once() -> None:
    target_date = dt.date(2026, 1, 6)
    total_kwh = 18.0

    cycle_load = core.build_cycle_hourly_load_series(target_date, total_kwh)

    assert float(cycle_load.sum()) == pytest.approx(total_kwh, abs=1e-6)


def test_night_plus_day_load_equals_total_once() -> None:
    target_date = dt.date(2026, 1, 6)
    total_kwh = 18.0
    tariff_cfg = core.DEFAULT_CONFIG["tariff"]

    cycle_load = core.build_cycle_hourly_load_series(target_date, total_kwh, tariff_cfg=tariff_cfg)
    _, window_end = core.compute_charging_window_for_target_date(target_date, tariff_cfg)
    tomorrow_start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=core.TIMEZONE)
    tomorrow_end = tomorrow_start + dt.timedelta(days=1)

    night_load = cycle_load[cycle_load.index < window_end]
    day_load = cycle_load[(cycle_load.index >= window_end) & (cycle_load.index < tomorrow_end)]

    assert float(night_load.sum() + day_load.sum()) == pytest.approx(total_kwh, abs=1e-6)


def test_simulate_full_day_soc_no_duplicate_index() -> None:
    target_date = dt.date(2026, 1, 6)
    tomorrow_start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=core.TIMEZONE)
    tomorrow_idx = pd.date_range(tomorrow_start, tomorrow_start + dt.timedelta(days=1), freq="h", inclusive="left")

    pv_df = pd.DataFrame({"pv_total_kwh": [0.0] * len(tomorrow_idx)}, index=tomorrow_idx)

    _, flows_df = core.simulate_full_day_soc(
        df=pv_df,
        total_consumption_kwh=18.0,
        soc_at_22=0.5,
        charge_kw=2.0,
        cutoff_soc=0.8,
        tomorrow_date=target_date,
        tariff_cfg=core.DEFAULT_CONFIG["tariff"],
    )

    assert flows_df.index.is_unique
    assert len(flows_df.index) == 24
