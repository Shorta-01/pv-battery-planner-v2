import unittest
import weather_ensemble as we


class TestOpenMeteoModelIds(unittest.TestCase):
    def test_gfs_forecast_models_param_is_seamless(self):
        self.assertIn("gfs", we.WEATHER_MODELS)
        self.assertIn("params", we.WEATHER_MODELS["gfs"])
        self.assertEqual(we.WEATHER_MODELS["gfs"]["params"].get("models"), "gfs_seamless")

    def test_gfs_historical_model_mapping_is_seamless(self):
        self.assertIn("gfs", we.HISTORICAL_FORECAST_MODEL_PARAMS)
        self.assertEqual(we.HISTORICAL_FORECAST_MODEL_PARAMS["gfs"], "gfs_seamless")


if __name__ == "__main__":
    unittest.main()
