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


def test_app_uses_card_widget_and_gates_diagnostics() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")

    assert "EV / Car Status" not in app_text
    assert "CAR STATUS" in app_text
    assert "EV diagnostics" in app_text
    assert "if APP_DEBUG:" in app_text
    assert "st.write({" not in app_text[app_text.index("def render_ev_car_status_panel"): app_text.index("forecast_mode, selected_models")]


def test_app_widget_surfaces_required_bmw_fields_and_deadline() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")
    start = app_text.index("def render_ev_car_status_panel")
    end = app_text.index("forecast_mode, selected_models", start)
    fn_block = app_text[start:end]

    for required in [
        "soc_pct",
        "is_plugged",
        "is_charging",
        "charge_power_kw",
        "expected_full_charge_ts",
        "range_km",
        "energy_needed_kwh",
        "time_to_full_min",
        "planner_status_text",
        "last_update_ts",
        "freshness_seconds",
        "ev_charge_deadline_time",
    ]:
        assert required in fn_block

    assert "api_get(\"/v1/ev/vehicles\")" in fn_block
    assert "api_get(\"/v1/ev/provider_status\")" in fn_block
    assert "ocpp" not in fn_block.lower()
