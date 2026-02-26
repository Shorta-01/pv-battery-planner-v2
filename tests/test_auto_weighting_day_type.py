import pandas as pd
import weather_ensemble as we


def test_classify_day_types():
    stable_cloud = pd.Series([10, 12, 11, 10, 9, 11])
    stable_code = pd.Series([0, 1, 1, 0, 1, 0])
    assert we.classify_day_type(stable_cloud, stable_code) == "stable_clear"

    variable_cloud = pd.Series([5, 80, 20, 90, 10, 85])
    variable_code = pd.Series([2, 3, 2, 3, 2, 3])
    assert we.classify_day_type(variable_cloud, variable_code) == "variable_cloudy"

    wet_cloud = pd.Series([90, 95, 88, 92, 96, 91])
    wet_code = pd.Series([61, 63, 65, 80, 81, 82])
    assert we.classify_day_type(wet_cloud, wet_code) == "fronty_wet"


def test_weights_sum_and_bias():
    base = {"knmi_harmonie_arome": 0.4, "dwd_icon_d2": 0.3, "gfs": 0.3}
    models = list(base)
    stable = we._weights_for_day_type(base, models, day_type="stable_clear", expert_mode=False)
    assert abs(sum(stable.values()) - 1.0) < 1e-9
    assert stable["knmi_harmonie_arome"] > base["knmi_harmonie_arome"]

    expert = we._weights_for_day_type(base, models, day_type="fronty_wet", expert_mode=True)
    assert abs(sum(expert.values()) - 1.0) < 1e-9
    assert expert["knmi_harmonie_arome"] == base["knmi_harmonie_arome"]
