import copy
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_build_effective_config_locks_bmw_as_ev_source_and_prunes_legacy_manual_fields() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["ev_vehicle_data"].update(
        {
            "enabled": True,
            "source": "manual",
            "bmw_enabled": False,
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
    assert "petrol_price_eur_per_l" not in ev_cfg
    assert "petrol_consumption_l_per_100km" not in ev_cfg
    assert "ready_by_time" not in ev_cfg


def test_settings_ui_removes_manual_vs_bmw_source_and_obsolete_ev_fields() -> None:
    text = Path("app.py").read_text(encoding="utf-8")

    assert "Vehicle data source" not in text
    assert "Charger max power (kW)" not in text
    assert "Petrol price (€/L)" not in text
    assert "Petrol consumption (L/100 km)" not in text
    assert "Optional ready-by time (HH:MM)" not in text
    assert "Advanced BMW connection details" not in text


def test_settings_ui_supports_active_bmw_vehicle_selection_when_multiple_mappings_exist() -> None:
    text = Path("app.py").read_text(encoding="utf-8")

    assert "Active BMW vehicle" in text
    assert "bmw_active_vehicle_id" in text


def test_backend_run_path_no_longer_depends_on_manual_vs_bmw_source_switch() -> None:
    text = Path("backend_api.py").read_text(encoding="utf-8")

    assert 'str(ev_cfg.get("source", "manual")) == "bmw_cardata"' not in text
    assert re.search(r"if bool\(ev_cfg.get\(\"enabled\", False\)\):", text)
