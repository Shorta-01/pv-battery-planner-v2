import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weather_ensemble as we


def test_forecast_quality_tier_low_for_global_only_and_high_repair() -> None:
    tier = we._forecast_quality_tier(
        selected_models=["ecmwf_ifs", "gfs"],
        quality_weight_factors_by_model={"ecmwf_ifs": 0.65, "gfs": 0.62},
        derived_irradiance_hours_by_model={"ecmwf_ifs": 12, "gfs": 10},
        missing_vars_by_model={"ecmwf_ifs": ["rain"], "gfs": ["precipitation"]},
    )
    assert tier == "low"


def test_forecast_quality_tier_high_for_strong_regional_coverage() -> None:
    tier = we._forecast_quality_tier(
        selected_models=["knmi_harmonie_arome", "dwd_icon_d2", "dwd_icon_eu"],
        quality_weight_factors_by_model={"knmi_harmonie_arome": 0.95, "dwd_icon_d2": 0.92, "dwd_icon_eu": 0.90},
        derived_irradiance_hours_by_model={"knmi_harmonie_arome": 1, "dwd_icon_d2": 0, "dwd_icon_eu": 2},
        missing_vars_by_model={"knmi_harmonie_arome": [], "dwd_icon_d2": [], "dwd_icon_eu": []},
    )
    assert tier == "high"
