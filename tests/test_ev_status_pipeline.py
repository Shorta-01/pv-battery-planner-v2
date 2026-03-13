import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ev_status import build_unified_ev_status


def _vehicle(**kwargs):
    base = {
        "soc_pct": 62.0,
        "is_plugged": True,
        "is_charging": True,
        "charge_power_kw": 3.6,
        "range_km": 260.0,
        "battery_capacity_kwh": 80.0,
        "last_update_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "freshness_seconds": 20,
        "effective_charge_power_limit_kw": 7.4,
    }
    base.update(kwargs)
    return {"veh": base}


def test_unified_status_uses_bmw_only_for_plugged_charging_power() -> None:
    out = build_unified_ev_status(
        ev_cfg={"enabled": True, "bmw_healthcheck_seconds": 300, "ev_charge_deadline_time": "07:00"},
        runtime_ev_cfg={"charger_max_power_kw": 11.0},
        bmw_provider_status={"provider_status": "healthy"},
        bmw_vehicles=_vehicle(is_plugged=False, is_charging=False, charge_power_kw=2.0),
        evse_status={"connected": True, "enabled": True, "is_plugged": True, "is_charging": True, "power_kw": 5.2},
    )
    assert out["soc_source"] == "bmw"
    assert out["is_plugged"] is False and out["plugged_source"] == "bmw"
    assert out["is_charging"] is False and out["charging_source"] == "bmw"
    assert out["charge_power_kw"] == 2.0 and out["charge_power_source"] == "bmw"


def test_charge_power_waiting_when_charging_without_bmw_power() -> None:
    out = build_unified_ev_status(
        ev_cfg={"enabled": True},
        runtime_ev_cfg={"charger_max_power_kw": 7.4},
        bmw_provider_status={},
        bmw_vehicles=_vehicle(charge_power_kw=None),
        evse_status={"connected": True, "enabled": True, "is_plugged": True, "is_charging": True, "power_kw": 7.0},
    )
    assert out["field_states"]["charge_power_kw"] == "waiting_for_bmw_power"


def test_deadline_not_configured_and_range_missing_states() -> None:
    out = build_unified_ev_status(
        ev_cfg={"enabled": True, "ev_charge_deadline_time": ""},
        runtime_ev_cfg={"charger_max_power_kw": 7.4},
        bmw_provider_status={},
        bmw_vehicles=_vehicle(range_km=None),
        evse_status={"connected": False, "enabled": True},
    )
    assert out["deadline_state"] == "not_configured"
    assert out["range_source"] == "bmw_missing"
    assert out["field_states"]["range_km"] == "bmw_missing"


def test_full_charge_eta_uses_vehicle_expected_ts_when_available() -> None:
    ts = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=45)).replace(microsecond=0).isoformat()
    out = build_unified_ev_status(
        ev_cfg={"enabled": True},
        runtime_ev_cfg={"charger_max_power_kw": 11.0},
        bmw_provider_status={},
        bmw_vehicles=_vehicle(expected_full_charge_ts=ts),
        evse_status={"connected": False, "enabled": False},
    )
    assert out["expected_full_charge_ts"] == ts
    assert out["expected_full_charge_source"] == "bmw"
    assert out["field_states"]["expected_full_charge_ts"] == "ready"


def test_full_charge_eta_uses_bmw_time_to_full_minutes_when_needed() -> None:
    before = dt.datetime.now(dt.timezone.utc)
    out = build_unified_ev_status(
        ev_cfg={"enabled": True},
        runtime_ev_cfg={"charger_max_power_kw": None},
        bmw_provider_status={},
        bmw_vehicles=_vehicle(expected_full_charge_ts=None, time_to_full_min=30, soc_pct=None, battery_capacity_kwh=None, charge_power_kw=None),
        evse_status={"connected": False, "enabled": False},
    )
    after = dt.datetime.now(dt.timezone.utc)
    eta = dt.datetime.fromisoformat(out["expected_full_charge_ts"])
    assert out["expected_full_charge_source"] == "bmw_time_to_full"
    assert out["field_states"]["expected_full_charge_ts"] == "ready"
    assert out["time_to_full_min"] == 30
    assert before + dt.timedelta(minutes=29) <= eta <= after + dt.timedelta(minutes=31)


def test_full_charge_eta_reports_reason_when_impossible() -> None:
    out = build_unified_ev_status(
        ev_cfg={"enabled": True},
        runtime_ev_cfg={"charger_max_power_kw": None},
        bmw_provider_status={},
        bmw_vehicles=_vehicle(soc_pct=None, charge_power_kw=None, time_to_full_min=None),
        evse_status={"connected": True, "enabled": True, "is_plugged": True, "is_charging": True, "power_kw": None},
    )
    assert out["expected_full_charge_ts"] is None
    assert out["field_states"]["expected_full_charge_ts"] in {"waiting_for_soc", "waiting_for_bmw_eta"}
