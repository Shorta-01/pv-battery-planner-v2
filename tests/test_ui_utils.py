from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_utils import resolve_pv_outlook_savings, weather_code_to_icon


def test_weather_code_to_icon_known_numeric_codes() -> None:
    expected = {
        0: "☀️",
        1: "🌤️",
        2: "⛅",
        3: "☁️",
        45: "🌫️",
        51: "🌦️",
        61: "🌧️",
        71: "🌨️",
        80: "🌦️",
        85: "🌨️",
        95: "⛈️",
    }
    for code, icon in expected.items():
        assert weather_code_to_icon(code) == icon


def test_weather_code_to_icon_string_labels() -> None:
    expected = {
        "clear": "☀️",
        "mainly_clear": "🌤️",
        "partly_cloudy": "⛅",
        "fog": "🌫️",
        "rain": "🌧️",
        "thunderstorm": "⛈️",
    }
    for label, icon in expected.items():
        result = weather_code_to_icon(label)
        assert result == icon
        assert result.strip()


def test_weather_code_to_icon_never_blank() -> None:
    for value in (None, 999, "", "???", "mystery_weather"):
        result = weather_code_to_icon(value)
        assert isinstance(result, str)
        assert result != ""
        assert result != "️"


def test_resolve_pv_outlook_savings_prefers_tomorrow_and_reconciles() -> None:
    payload = {
        "baseline_cost_eur_total": 10,
        "plan_cost_eur_total": 20,
        "savings_eur_total": -10,
        "baseline_cost_eur_tomorrow": 7.62,
        "plan_cost_eur_tomorrow": 7.94,
        "savings_eur_tomorrow": -0.32,
        "hourly_savings_eur_tomorrow": [0.0] * 24,
    }

    out = resolve_pv_outlook_savings(payload)

    assert out["base_cost"] == 7.62
    assert out["plan_cost"] == 7.94
    assert out["savings"] == 7.62 - 7.94
    assert isinstance(out["hourly"], list)
    assert len(out["hourly"]) == 24


def test_resolve_pv_outlook_savings_corrects_bad_savings_field() -> None:
    payload = {
        "baseline_cost_eur_tomorrow": 7.62,
        "plan_cost_eur_tomorrow": 7.94,
        "savings_eur_tomorrow": -4.17,
        "hourly_savings_eur_tomorrow": [0.0] * 24,
    }

    out = resolve_pv_outlook_savings(payload)

    assert out["savings"] == 7.62 - 7.94
