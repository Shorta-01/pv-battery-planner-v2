from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_utils import (
    format_ev_datetime,
    format_ev_km,
    format_ev_kw,
    format_ev_time_to_full_minutes,
    is_app_debug_enabled,
    resolve_pv_outlook_savings,
    summarize_cardata_readiness,
    summarize_ev_provider_state,
    weather_code_to_icon,
)


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


def test_is_app_debug_enabled_reads_only_app_debug() -> None:
    assert is_app_debug_enabled({"APP_DEBUG": "1"}) is True
    assert is_app_debug_enabled({"APP_DEBUG": "true"}) is True
    assert is_app_debug_enabled({"APP_DEBUG": "0", "DEBUG": "1"}) is False


def test_summarize_ev_provider_state_auth_required() -> None:
    state = summarize_ev_provider_state(
        {"provider_status": "auth_required", "data_status": "stale"},
        has_vehicle=False,
        soc_available=False,
        vehicle_freshness_seconds=None,
    )
    assert "Auth required" in state["chips"]
    assert "authorization required" in state["fallback"].lower()


def test_summarize_ev_provider_state_no_vehicle_and_stale_vehicle_data() -> None:
    no_vehicle = summarize_ev_provider_state(
        {"provider_status": "degraded", "data_status": "partial"},
        has_vehicle=False,
        soc_available=False,
        vehicle_freshness_seconds=None,
    )
    assert "No vehicle" in no_vehicle["chips"]

    stale = summarize_ev_provider_state(
        {"provider_status": "healthy", "data_status": "fresh"},
        has_vehicle=True,
        soc_available=True,
        vehicle_freshness_seconds=2400,
    )
    assert "Connected" in stale["chips"]
    assert "Stale" in stale["chips"]
    assert "last known" in stale["helper"].lower()


def test_ev_format_helpers_support_friendly_unknown_labels() -> None:
    assert format_ev_km(None, unknown_label="Not available") == "Not available"
    assert format_ev_kw(None, unknown_label="Not available") == "Not available"
    assert format_ev_time_to_full_minutes(None, unknown_label="Waiting for BMW data") == "Waiting for BMW data"
    assert format_ev_datetime(None, unknown_label="Waiting for BMW data") == "Waiting for BMW data"


def test_summarize_cardata_readiness_ev_disabled_is_neutral() -> None:
    state = summarize_cardata_readiness(
        ev_enabled=False,
        has_client_id=False,
        provider_status={},
        has_vehicle=False,
        has_device_flow_session=False,
        vehicle_freshness_seconds=None,
    )
    assert state["required"] is False
    assert state["ready"] is True


def test_summarize_cardata_readiness_ready_and_not_ready_states() -> None:
    ready = summarize_cardata_readiness(
        ev_enabled=True,
        has_client_id=True,
        provider_status={"provider_status": "healthy", "data_status": "fresh"},
        has_vehicle=True,
        has_device_flow_session=False,
        vehicle_freshness_seconds=60,
    )
    assert ready["ready"] is True
    assert ready["detail"] == "Ready"

    missing_client = summarize_cardata_readiness(
        ev_enabled=True,
        has_client_id=False,
        provider_status={"provider_status": "healthy", "data_status": "fresh"},
        has_vehicle=True,
        has_device_flow_session=False,
        vehicle_freshness_seconds=60,
    )
    assert missing_client["ready"] is False
    assert missing_client["detail"] == "Missing BMW client id"

    no_vehicle = summarize_cardata_readiness(
        ev_enabled=True,
        has_client_id=True,
        provider_status={"provider_status": "healthy", "data_status": "fresh"},
        has_vehicle=False,
        has_device_flow_session=False,
        vehicle_freshness_seconds=60,
    )
    assert no_vehicle["ready"] is False
    assert no_vehicle["detail"] == "No linked vehicle"

    stale = summarize_cardata_readiness(
        ev_enabled=True,
        has_client_id=True,
        provider_status={"provider_status": "healthy", "data_status": "stale"},
        has_vehicle=True,
        has_device_flow_session=False,
        vehicle_freshness_seconds=2400,
    )
    assert stale["ready"] is False
    assert stale["detail"] == "Stale/degraded data"
