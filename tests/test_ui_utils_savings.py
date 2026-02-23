import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_utils import resolve_pv_outlook_savings


def test_prefers_cycle_values_over_tomorrow_values() -> None:
    payload = {
        "baseline_cost_eur_cycle": 12.0,
        "plan_cost_eur_cycle": 9.0,
        "savings_eur_cycle": 3.0,
        "baseline_cost_eur_total": 11.0,
        "plan_cost_eur_total": 10.0,
        "savings_eur_total": 1.0,
        "baseline_cost_eur_tomorrow": 5.0,
        "plan_cost_eur_tomorrow": 4.0,
        "savings_eur_tomorrow": 1.0,
        "hourly_savings_eur_tomorrow": [0.1] * 24,
        "savings_horizon_label": "Cycle (off-peak start → next off-peak start): 22:00 → 22:00",
    }

    result = resolve_pv_outlook_savings(payload)

    assert result["display_scope"] == "cycle"
    assert result["base_cost"] == 12.0
    assert result["plan_cost"] == 9.0
    assert result["savings"] == 3.0
    assert result["hourly"] == [0.1] * 24
    assert "Cycle savings shown" in result["note"]
    assert "tomorrow (00–24)" in result["note"]


def test_fallback_to_tomorrow_values_when_cycle_missing() -> None:
    payload = {
        "baseline_cost_eur_tomorrow": 7.5,
        "plan_cost_eur_tomorrow": 6.0,
        "savings_eur_tomorrow": 1.5,
        "hourly_savings_eur_tomorrow": [0.0] * 24,
    }

    result = resolve_pv_outlook_savings(payload)

    assert result["display_scope"] == "tomorrow"
    assert result["base_cost"] == 7.5
    assert result["plan_cost"] == 6.0
    assert result["savings"] == 1.5
    assert result["hourly"] == [0.0] * 24
    assert "tomorrow only" in result["note"]
