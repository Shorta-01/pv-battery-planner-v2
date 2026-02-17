import pandas as pd

import weather_ensemble as we


def test_weighted_ensemble_falls_back_to_mean_when_no_weighted_models():
    idx = pd.date_range("2026-01-01", periods=2, freq="h")
    s1 = pd.Series([1.0, 3.0], index=idx)
    s2 = pd.Series([5.0, 7.0], index=idx)

    out, weights = we._weighted_ensemble({"unknown_a": s1, "unknown_b": s2}, ["unknown_a", "unknown_b"])

    assert weights is None
    assert out.iloc[0] == 3.0
    assert out.iloc[1] == 5.0
