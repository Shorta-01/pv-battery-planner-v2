import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


SCOPED_FILES = [
    Path("planner_core.py"),
    Path("weather_ensemble.py"),
]


def test_effective_elevation_helper_prefers_resolved_value() -> None:
    assert core.effective_elevation_m(145.5) == 145.5
    assert core.effective_elevation_m(None) == core.PVLIB_LOCATION_ALTITUDE_FALLBACK_M


def test_no_leftover_altitude_zero_or_implicit_or0_paths_in_scoped_chain() -> None:
    for path in SCOPED_FILES:
        text = path.read_text()
        assert "altitude=0.0" not in text
        assert "elevation_m or 0.0" not in text


def test_pvlib_location_paths_use_effective_elevation_helper() -> None:
    for path in SCOPED_FILES:
        text = path.read_text()
        if "pvlib.location.Location(" in text:
            assert "effective_elevation_m(" in text or "PVLIB_LOCATION_ALTITUDE_FALLBACK_M" in text
