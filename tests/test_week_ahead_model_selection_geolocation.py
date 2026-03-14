import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_week_ahead_uses_geolocation_and_horizon() -> None:
    eu = we.select_week_ahead_models(requested_days=7, lat=50.85, lon=4.35)
    au = we.select_week_ahead_models(requested_days=7, lat=-33.86, lon=151.21)
    assert eu
    assert au
    assert "dwd_icon_d2" not in eu
    assert "dwd_icon_d2" not in au
    assert any(m in eu for m in ["dwd_icon_eu", "meteofrance_seamless", "ecmwf_ifs"])
    assert au[0] in {"ecmwf_ifs", "ecmwf_aifs", "gfs"}
