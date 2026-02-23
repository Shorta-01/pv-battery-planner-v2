import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_utils import resolve_pv_outlook_savings


def test_resolver_prefers_cycle_values_over_tomorrow() -> None:
    payload = {
        "baseline_cost_eur_cycle": 13.0,
        "plan_cost_eur_cycle": 8.0,
        "savings_eur_cycle": 5.0,
        "baseline_cost_eur_total": 12.0,
        "plan_cost_eur_total": 9.0,
        "savings_eur_total": 3.0,
        "baseline_cost_eur_tomorrow": 6.0,
        "plan_cost_eur_tomorrow": 5.0,
        "savings_eur_tomorrow": 1.0,
        "hourly_savings_eur_tomorrow": [0.25] * 24,
        "savings_horizon_label": "Cycle (off-peak start → next off-peak start): 22:00 → 22:00",
    }

    out = resolve_pv_outlook_savings(payload)

    assert out["display_scope"] == "cycle"
    assert out["base_cost"] == 13.0
    assert out["plan_cost"] == 8.0
    assert out["savings"] == 5.0
    assert out["hourly"] == [0.25] * 24


def test_resolver_fallback_to_tomorrow_values() -> None:
    payload = {
        "baseline_cost_eur_tomorrow": 7.0,
        "plan_cost_eur_tomorrow": 6.0,
        "savings_eur_tomorrow": 1.0,
        "hourly_savings_eur_tomorrow": [0.0] * 24,
    }

    out = resolve_pv_outlook_savings(payload)

    assert out["display_scope"] == "tomorrow"
    assert out["base_cost"] == 7.0
    assert out["plan_cost"] == 6.0
    assert out["savings"] == 1.0
    assert "tomorrow only" in out["note"]
