import datetime as dt
import pandas as pd
import weather_ensemble as we


def test_local_day_index_dst_counts():
    assert len(we.local_day_hourly_index(dt.date(2026, 1, 10), "Europe/Brussels")) == 24
    assert len(we.local_day_hourly_index(dt.date(2026, 3, 29), "Europe/Brussels")) == 23
    assert len(we.local_day_hourly_index(dt.date(2026, 10, 25), "Europe/Brussels")) == 25


def test_quantile_sanity_no_nans():
    idx = we.local_day_hourly_index(dt.date(2026, 1, 10), "Europe/Brussels")
    matrix = pd.DataFrame({"a": [0.0]*8 + [0.3]*8 + [0.0]*8, "b": [0.0]*8 + [0.5]*8 + [0.0]*8, "c": [0.0]*8 + [0.7]*8 + [0.0]*8}, index=idx)
    p10 = matrix.quantile(0.10, axis=1)
    p50 = matrix.quantile(0.50, axis=1)
    p90 = matrix.quantile(0.90, axis=1)
    p10s, p90s = we._apply_uncertainty_day_type_scaling(p10, p50, p90, "variable_cloudy")
    assert (p10s <= p50).all()
    assert (p50 <= p90s).all()
    assert p10s.notna().all() and p50.notna().all() and p90s.notna().all()
