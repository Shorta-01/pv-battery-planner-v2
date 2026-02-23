from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_utils import weather_code_to_icon


def test_weather_code_to_icon_common_codes_are_non_empty() -> None:
    for code in [0, 1, 2, 3, 45, 51, 61, 71, 80, 95]:
        icon = weather_code_to_icon(code)
        assert isinstance(icon, str)
        assert icon.strip()
        assert icon != "❓"


def test_weather_code_to_icon_none_and_unknown() -> None:
    assert weather_code_to_icon(None) == "❓"
    assert weather_code_to_icon(12345) == "❓"
    assert weather_code_to_icon("0") == "☀️"
    assert weather_code_to_icon(1.0) == "🌤️"


def test_weather_code_to_icon_symbolic_strings() -> None:
    assert weather_code_to_icon("rain") == "🌧️"
    assert weather_code_to_icon("partly_cloudy") == "⛅"
    assert weather_code_to_icon("unknown") == "❓"
