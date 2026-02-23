import weather_ensemble as we


def test_weather_models_uses_ecmwf_aifs025_for_open_meteo_models_param():
    assert we.WEATHER_MODELS['ecmwf_aifs']['params']['models'] == 'ecmwf_aifs025'


def test_historical_forecast_params_uses_ecmwf_aifs025():
    _, params = we.historical_forecast_params('ecmwf_aifs')
    assert params['models'] == 'ecmwf_aifs025'
