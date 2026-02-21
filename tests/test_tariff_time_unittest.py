import unittest

from tariff_time import compute_offpeak_segments, parse_hhmm


class TariffTimeTests(unittest.TestCase):
    def test_parse_hhmm_requires_strict_format(self):
        with self.assertRaises(ValueError):
            parse_hhmm("7:00")
        with self.assertRaises(ValueError):
            parse_hhmm("07:0")

    def test_parse_hhmm_24h_only_allowed_as_end(self):
        with self.assertRaises(ValueError):
            parse_hhmm("24:00", allow_24_end=False)
        self.assertEqual(parse_hhmm("24:00", allow_24_end=True), 1440)

    def test_compute_offpeak_segments_allows_overnight(self):
        self.assertEqual(
            compute_offpeak_segments(parse_hhmm("22:00"), parse_hhmm("07:00", allow_24_end=True)),
            [(1320, 1440), (0, 420)],
        )

    def test_compute_offpeak_segments_equal_invalid_except_all_day(self):
        with self.assertRaises(ValueError):
            compute_offpeak_segments(600, 600)
        self.assertEqual(compute_offpeak_segments(0, 1440), [(0, 1440)])


if __name__ == "__main__":
    unittest.main()
