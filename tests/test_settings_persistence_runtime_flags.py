from pathlib import Path


def test_app_payload_keeps_runtime_tariff_flags_and_battery_max_soc():
    src = Path("app.py").read_text(encoding="utf-8")
    assert '"optimization_mode": str(' in src
    assert '"night_load_from_battery": bool(' in src
    assert '"battery_max_soc_percent": float(' in src


def test_default_config_declares_runtime_tariff_flags_and_battery_max_soc():
    src = Path("planner_core.py").read_text(encoding="utf-8")
    assert '"optimization_mode": "window_only"' in src
    assert '"night_load_from_battery": False' in src
    assert '"battery_max_soc_percent": BATTERY_MAX_SOC_PERCENT' in src
