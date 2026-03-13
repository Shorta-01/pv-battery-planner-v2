from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_utils import (
    format_ev_bool,
    format_ev_datetime,
    format_ev_freshness,
    format_ev_km,
    format_ev_kwh,
    format_ev_kw,
    format_ev_time_to_full_minutes,
    summarize_ev_setup_state,
)


def test_ev_format_helpers_basic() -> None:
    assert format_ev_bool(True) == "Yes"
    assert format_ev_bool(False) == "No"
    assert format_ev_bool(None) == "—"
    assert format_ev_kw(7.24) == "7.2 kW"
    assert format_ev_kwh(5.55) == "5.5 kWh"
    assert format_ev_km(312.4) == "312 km"
    assert format_ev_time_to_full_minutes(130) == "2h 10m"
    assert format_ev_freshness(45) == "45s ago"
    assert format_ev_freshness(120) == "2m ago"
    assert format_ev_datetime("2026-03-11T06:30:00+00:00")


def test_app_uses_card_widget_and_canonical_debug_gating() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")

    assert "EV / Car Status" not in app_text
    assert "CAR STATUS" in app_text
    assert "EV diagnostics" in app_text
    assert "APP_DEBUG = is_app_debug_enabled()" in app_text
    assert "bool(os.getenv(\"APP_DEBUG\"))" not in app_text
    assert "os.getenv(\"DEBUG\"" not in app_text
    assert "if APP_DEBUG:" in app_text
    assert "st.write({" not in app_text[app_text.index("def render_ev_car_status_panel"): app_text.index("forecast_mode, selected_models")]


def test_app_widget_uses_unified_ev_status_endpoint() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")
    start = app_text.index("def render_ev_car_status_panel")
    end = app_text.index("forecast_mode, selected_models", start)
    fn_block = app_text[start:end]

    assert "api_get(\"/v1/ev/status\")" in fn_block
    assert "api_get(\"/v1/ev/vehicles\")" not in fn_block
    assert "api_get(\"/v1/ev/provider_status\")" not in fn_block
    for required in ["soc_pct", "is_plugged", "is_charging", "charge_power_kw", "expected_full_charge_ts", "range_km", "deadline_state", "freshness_label"]:
        assert required in fn_block


def test_results_layout_places_car_status_with_fusionsolar_card() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")
    with_top_left = app_text.index("with top_left:")
    render_summary = app_text.index("render_offpeak_plan_summary(", with_top_left)
    render_car_status = app_text.index("render_ev_car_status_panel(st.container())", with_top_left)
    render_pv_week = app_text.index("render_pv_week_ahead_widget(pv_week_ahead_display)", with_top_left)

    assert render_summary < render_car_status < render_pv_week


def test_app_widget_uses_unified_status_labels_and_debug() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")
    start = app_text.index("def render_ev_car_status_panel")
    end = app_text.index("forecast_mode, selected_models", start)
    fn_block = app_text[start:end]

    assert "Not charging" in fn_block
    assert "Waiting for charger data" in fn_block
    assert "Not set" in fn_block
    assert "Not provided by BMW yet" in fn_block
    assert "Power from OCPP" in fn_block
    assert "if APP_DEBUG:" in fn_block


def test_car_status_widget_no_literal_placeholder_tokens() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")
    start = app_text.index("def render_ev_car_status_panel")
    end = app_text.index("forecast_mode, selected_models", start)
    fn_block = app_text[start:end]

    assert ">⏱ {freshness_label}</span>" not in fn_block
    assert ">🚗 {status_chip_text}</span>" not in fn_block


def test_settings_vehicle_data_uses_compact_status_chips_and_refresh_label() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")

    assert "Refresh BMW data" in app_text
    assert "Manual refresh" not in app_text
    assert "status_chip_specs" in app_text
    assert "Provider status:" in app_text
    assert "if APP_DEBUG:" in app_text


def test_summarize_ev_setup_state_missing_client_id_and_reconnect_paths() -> None:
    missing = summarize_ev_setup_state({}, ev_enabled=True, has_client_id=False, vehicle_count=0, has_device_flow_session=False)
    assert missing["title"] == "BMW client ID required"

    reconnect = summarize_ev_setup_state({"provider_status": "auth_required"}, ev_enabled=True, has_client_id=True, vehicle_count=0, has_device_flow_session=False)
    assert reconnect["title"] == "Reconnect required"


def test_summarize_ev_setup_state_device_flow_and_vehicle_outcomes() -> None:
    device_flow = summarize_ev_setup_state({"provider_status": "degraded"}, ev_enabled=True, has_client_id=True, vehicle_count=0, has_device_flow_session=True)
    assert device_flow["title"] == "BMW authorization required"

    no_vehicle = summarize_ev_setup_state({"provider_status": "healthy"}, ev_enabled=True, has_client_id=True, vehicle_count=0, has_device_flow_session=False)
    assert no_vehicle["title"] == "No BMW vehicles found"

    one_vehicle = summarize_ev_setup_state({"provider_status": "healthy"}, ev_enabled=True, has_client_id=True, vehicle_count=1, has_device_flow_session=False)
    assert one_vehicle["title"] == "1 vehicle linked"

    many = summarize_ev_setup_state({"provider_status": "healthy"}, ev_enabled=True, has_client_id=True, vehicle_count=2, has_device_flow_session=False)
    assert many["title"] == "Multiple vehicles found, choose one"


def test_settings_block_keeps_diagnostics_secondary_and_no_raw_payload_dump() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")
    start = app_text.index('st.markdown("#### EV Vehicle Data")')
    end = app_text.index('cfg_load_profile = [float(v) for v in effective_cfg["load_profile"]["load_profile_24h"]]', start)
    block = app_text[start:end]

    assert "if APP_DEBUG:" in block
    assert "st.json({\"provider_status\": provider_status, \"vehicles\": vehicles_payload}" not in block
