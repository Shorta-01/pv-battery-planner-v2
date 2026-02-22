import datetime as dt
import unittest

import pandas as pd

import planner_core as core
import scoring
import weather_ensemble


class PriorityAccuracyFixesTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "tariff": {
                "offpeak_windows_by_dow": [[("22:00", "07:00")]] * 7,
                "peak_grid_price_eur_per_kwh": 0.3,
                "offpeak_grid_price_eur_per_kwh": 0.2,
            }
        }

    def test_offpeak_clock_mask(self):
        day = dt.date(2026, 1, 7)
        idx = pd.date_range(pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=core.TIMEZONE), periods=24, freq="h")
        mask = core.get_offpeak_mask(idx, self.cfg["tariff"])
        self.assertEqual(int(mask.sum()), 9)
        self.assertTrue(bool(mask.loc[idx[1]]))
        self.assertTrue(bool(mask.loc[idx[23]]))
        self.assertFalse(bool(mask.loc[idx[12]]))

    def test_charge_session_respects_cfg(self):
        charge_date = dt.date(2026, 1, 7)
        session = core.get_charge_session_index(charge_date, self.cfg["tariff"])
        self.assertEqual(len(session), 9)
        self.assertIn(pd.Timestamp(dt.datetime.combine(charge_date, dt.time(22, 0)), tz=core.TIMEZONE), session)
        self.assertIn(pd.Timestamp(dt.datetime.combine(charge_date + dt.timedelta(days=1), dt.time(6, 0)), tz=core.TIMEZONE), session)

    def test_import_price_respects_cfg(self):
        day = dt.date(2026, 1, 7)
        self.assertEqual(core.import_price_eur_per_kwh(pd.Timestamp(dt.datetime.combine(day, dt.time(1, 0)), tz=core.TIMEZONE), self.cfg["tariff"]), 0.2)
        self.assertEqual(core.import_price_eur_per_kwh(pd.Timestamp(dt.datetime.combine(day, dt.time(12, 0)), tz=core.TIMEZONE), self.cfg["tariff"]), 0.3)

    def test_pv_quality_ratio_when_loc_present(self):
        if not core.PVLIB_AVAILABLE:
            self.skipTest("pvlib not installed")
        idx = pd.date_range("2026-06-01 08:00", periods=4, freq="h", tz=core.TIMEZONE)
        pv_df = pd.DataFrame({"pv_total_kwh": [0.5, 0.8, 0.9, 0.6]}, index=idx)
        weather_df = pd.DataFrame({"temp_air_c": [15, 16, 17, 16], "wind_speed_ms": [2, 2, 2, 2]}, index=idx)
        loc = core.Location(name="X", latitude=50.85, longitude=4.35)
        out = scoring.compute_pv_quality_score(pv_df, weather_df, dt.date(2026, 6, 1), core.TIMEZONE, loc=loc)
        self.assertIsNotNone(out.get("ratio"))
        self.assertTrue(0 <= int(out.get("score", -1)) <= 100)
        self.assertFalse(bool(out.get("is_fallback", True)))

    def test_missing_irradiance_behavior(self):
        if not core.PVLIB_AVAILABLE:
            self.skipTest("pvlib not installed")
        idx = pd.date_range("2026-06-01 10:00", periods=3, freq="h", tz=core.TIMEZONE)
        loc = core.Location(name="X", latitude=50.85, longitude=4.35)

        a = pd.DataFrame({"ghi_wm2": [None, None, None], "cloud_cover_pct": [20, 20, 20], "temp_air_c": [15, 15, 15], "wind_speed_ms": [1, 1, 1]}, index=idx)
        pv_a = core.build_pv_forecast(a, loc, tz=core.TIMEZONE)
        self.assertTrue((pd.to_numeric(pv_a["pv_total_kwh"], errors="coerce").fillna(0.0) > 0).any())

        b = pd.DataFrame({"ghi_wm2": [None, None, None], "cloud_cover_pct": [None, None, None], "temp_air_c": [15, 15, 15], "wind_speed_ms": [1, 1, 1]}, index=idx)
        pv_b = core.build_pv_forecast(b, loc, tz=core.TIMEZONE)
        self.assertTrue(pd.to_numeric(pv_b["pv_total_kwh"], errors="coerce").isna().any())

    def test_satellite_volatility_heuristic(self):
        stable = pd.DataFrame({"cloud_cover_pct": [20, 20, 20, 20, 20]})
        volatile = pd.DataFrame({"cloud_cover_pct": [10, 70, 20, 80, 15]})
        self.assertFalse(weather_ensemble._nowcast_sky_is_volatile(stable))
        self.assertTrue(weather_ensemble._nowcast_sky_is_volatile(volatile))


if __name__ == "__main__":
    unittest.main()
