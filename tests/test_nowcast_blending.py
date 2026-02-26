import pandas as pd
import weather_ensemble as we


def test_nowcast_blending_linear_transition():
    idx = pd.date_range("2026-01-01", periods=10, freq="h", tz="Europe/Brussels")
    nowcast = pd.DataFrame({"shortwave_radiation": [100.0] * 10, "direct_normal_irradiance": [100.0] * 10, "diffuse_radiation": [100.0] * 10}, index=idx)
    nwp = pd.DataFrame({"shortwave_radiation": [200.0] * 10, "direct_normal_irradiance": [200.0] * 10, "diffuse_radiation": [200.0] * 10}, index=idx)
    blended = we.blend_nowcast_with_nwp(nowcast, nwp, horizon_hours=6, blend_hours=2)
    assert blended.iloc[0]["shortwave_radiation"] == 100.0
    assert blended.iloc[3]["shortwave_radiation"] == 100.0
    assert blended.iloc[6]["shortwave_radiation"] == 200.0
    assert 100.0 < blended.iloc[4]["shortwave_radiation"] < 200.0
    assert blended.notna().all().all()
