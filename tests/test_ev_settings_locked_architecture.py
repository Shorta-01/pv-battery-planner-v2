import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_build_effective_config_locks_bmw_as_ev_source_and_migrates_legacy_deadline() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["ev_vehicle_data"].update(
        {
            "enabled": True,
            "source": "manual",
            "bmw_enabled": False,
            "bmw_client_id": "client-123",
            "bmw_stream_enabled": True,
            "charger_max_power_kw": 11.0,
            "petrol_price_eur_per_l": 1.9,
            "petrol_consumption_l_per_100km": 6.5,
            "ready_by_time": "07:00",
        }
    )

    out = core.build_effective_config(cfg)
    ev_cfg = out["ev_vehicle_data"]

    assert ev_cfg["source"] == "bmw_cardata"
    assert ev_cfg["bmw_enabled"] is True
    assert "bmw_stream_enabled" not in ev_cfg
    assert "charger_max_power_kw" not in ev_cfg
    assert ev_cfg["petrol_price_eur_per_l"] == 1.9
    assert ev_cfg["petrol_consumption_l_per_100km"] == 6.5
    assert ev_cfg["ev_charge_deadline_time"] == "07:00"
    assert "ready_by_time" not in ev_cfg


def test_settings_ui_removes_manual_vs_bmw_source_and_shows_economics_and_deadline() -> None:
    text = Path("app.py").read_text(encoding="utf-8")

    assert "Vehicle data source" not in text
    assert "Charger max power (kW)" not in text
    assert "Petrol price (€/L)" in text
    assert "Petrol consumption (L/100 km)" in text
    assert "EV charge deadline (HH:MM)" in text
    assert "Optional ready-by time (HH:MM)" not in text
    assert "ready_by_time" not in text
    assert "Advanced BMW connection details" not in text


def test_validate_config_accepts_optional_economics_and_rejects_invalid_deadline() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["ev_vehicle_data"].update(
        {
            "enabled": True,
            "bmw_client_id": "abc",
            "petrol_price_eur_per_l": "",
            "petrol_consumption_l_per_100km": "",
            "ev_charge_deadline_time": "07:30",
        }
    )
    out = core.build_effective_config(cfg)
    assert out["ev_vehicle_data"]["ev_charge_deadline_time"] == "07:30"

    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["ev_vehicle_data"]["ev_charge_deadline_time"] = "7:99"
    with pytest.raises(ValueError, match="Invalid time"):
        core.build_effective_config(cfg_bad)


def test_validate_config_requires_bmw_client_id_when_ev_enabled() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["ev_vehicle_data"].update({"enabled": True, "bmw_client_id": ""})
    with pytest.raises(ValueError, match="bmw_client_id"):
        core.build_effective_config(cfg)


def test_settings_ui_supports_active_bmw_vehicle_selection_when_multiple_mappings_exist() -> None:
    text = Path("app.py").read_text(encoding="utf-8")

    assert "Active BMW vehicle" in text
    assert "bmw_active_vehicle_id" in text


def test_backend_run_path_no_longer_depends_on_manual_vs_bmw_source_switch() -> None:
    text = Path("backend_api.py").read_text(encoding="utf-8")

    assert 'str(ev_cfg.get("source", "manual")) == "bmw_cardata"' not in text
    assert "def _get_planning_ready_ev_state(self)" in text
    assert "ev_state = self._get_planning_ready_ev_state()" in text


def test_settings_ui_exposes_unified_cardata_setup_actions() -> None:
    text = Path("app.py").read_text(encoding="utf-8")

    assert "Setup CarData connection" in text
    assert "Check connection" in text
    assert "Vehicle data source" not in text
    assert "Manual vs BMW" not in text


def test_settings_ui_keeps_bmw_only_semantics_without_manual_telemetry_path() -> None:
    text = Path("app.py").read_text(encoding="utf-8")

    assert '"source": "bmw_cardata"' in text
    assert '"bmw_enabled": bool(ui.get("cfg_ev_enabled", False))' in text
    assert "manual EV telemetry" not in text


def test_ev_settings_ui_places_petrol_economics_fields_on_one_row() -> None:
    text = Path("app.py").read_text(encoding="utf-8")

    assert 'petrol_col1, petrol_col2 = st.columns(2, gap="large")' in text
    assert 'with petrol_col1:' in text
    assert 'with petrol_col2:' in text
    assert "Petrol price (€/L)" in text
    assert "Petrol consumption (L/100 km)" in text


def test_readiness_strip_includes_cardata_status() -> None:
    text = Path("app.py").read_text(encoding="utf-8")

    assert "summarize_cardata_readiness(" in text
    assert '"CarData": []' in text
    assert '["Inputs", "Location", "Tariffs", "PV", "Battery", "Weather", "CarData"]' in text
