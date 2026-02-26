import pandas as pd
import weather_ensemble as we


def test_uncertainty_scaling_monotonic_and_nonnegative():
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="Europe/Brussels")
    p10 = pd.Series([0.1, 0.2, 0.3], index=idx)
    p50 = pd.Series([0.2, 0.4, 0.6], index=idx)
    p90 = pd.Series([0.4, 0.8, 1.0], index=idx)

    n10, n90 = we._apply_uncertainty_day_type_scaling(p10, p50, p90, "stable_clear")
    assert (n10 >= 0).all()
    assert (n10 <= p50).all()
    assert (n90 >= p50).all()
    assert ((n90 - n10) < (p90 - p10)).all()

    w10, w90 = we._apply_uncertainty_day_type_scaling(p10, p50, p90, "fronty_wet")
    assert ((w90 - w10) > (p90 - p10)).all()
