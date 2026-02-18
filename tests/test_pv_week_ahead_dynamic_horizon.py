import datetime as dt

import pandas as pd

from weather_ensemble import _weighted_ensemble


def test_weighted_ensemble_dynamic_horizon_renormalizes_per_timestamp() -> None:
    tz = "Europe/Brussels"
    start = pd.Timestamp(dt.datetime(2025, 1, 1, 0, 0), tz=tz)
    idx_7d = pd.date_range(start, start + dt.timedelta(days=7), freq="h", inclusive="left")
    idx_2d = pd.date_range(start, start + dt.timedelta(days=2), freq="h", inclusive="left")
    idx_4d = pd.date_range(start, start + dt.timedelta(days=4), freq="h", inclusive="left")

    series_map = {
        "knmi_harmonie_arome": pd.Series(10.0, index=idx_7d),
        "dwd_icon_d2": pd.Series(30.0, index=idx_2d),
        "ecmwf_ifs": pd.Series(20.0, index=idx_4d),
    }

    out, _ = _weighted_ensemble(series_map, list(series_map.keys()))

    # day 1 uses A+B+C with default weights 0.45, 0.35, 0.20
    assert abs(float(out.loc[start + dt.timedelta(hours=12)]) - 19.0) < 1e-9

    # day 3 uses A+C only, renormalized to 0.45/(0.45+0.20), 0.20/(0.45+0.20)
    d3 = start + dt.timedelta(days=2, hours=12)
    expected_d3 = (10.0 * 0.45 + 20.0 * 0.20) / (0.45 + 0.20)
    assert abs(float(out.loc[d3]) - expected_d3) < 1e-9

    # day 6 uses A only
    d6 = start + dt.timedelta(days=5, hours=12)
    assert abs(float(out.loc[d6]) - 10.0) < 1e-9

    # daily totals must equal sum of hourly ensemble outputs
    daily = out.resample("D").sum(min_count=1)
    for day_start, total in daily.items():
        day_end = day_start + dt.timedelta(days=1)
        slice_sum = out[(out.index >= day_start) & (out.index < day_end)].sum()
        assert abs(float(total) - float(slice_sum)) < 1e-9
