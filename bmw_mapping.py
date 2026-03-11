from __future__ import annotations

import datetime as dt
from typing import Any

from bmw_models import NormalizedVehicleState, parse_dt, utcnow


def _as_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    f = _as_float(v)
    return None if f is None else int(f)


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in {"true", "1", "yes", "plugged", "charging", "connected"}:
            return True
        if v.lower() in {"false", "0", "no", "unplugged", "disconnected"}:
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def freshness_bucket(last_update_ts: dt.datetime | None, now: dt.datetime | None = None) -> tuple[str, int | None]:
    if last_update_ts is None:
        return "error", None
    now = now or utcnow()
    age = max(0, int((now - last_update_ts).total_seconds()))
    if age < 120:
        return "fresh", age
    if age < 600:
        return "aging", age
    if age < 1800:
        return "stale", age
    return "error", age


def map_bmw_payload_to_vehicle_states(payload: dict[str, Any]) -> list[NormalizedVehicleState]:
    vehicles = payload.get("vehicles") if isinstance(payload.get("vehicles"), list) else [payload]
    out: list[NormalizedVehicleState] = []
    for row in vehicles:
        if not isinstance(row, dict):
            continue
        vehicle_id = str(row.get("vehicle_id") or row.get("vin") or row.get("id") or "").strip()
        if not vehicle_id:
            continue
        ts = parse_dt(row.get("last_update_ts") or row.get("timestamp") or row.get("event_time"))
        soc = _as_float(row.get("soc_pct") or row.get("soc") or row.get("battery", {}).get("soc"))
        battery_capacity = _as_float(row.get("battery_capacity_kwh") or row.get("battery", {}).get("capacity_kwh"))
        is_plugged = _as_bool(row.get("is_plugged") or row.get("plugged") or row.get("plug_status"))
        is_charging = _as_bool(row.get("is_charging") or row.get("charging") or row.get("charging_status"))
        charge_error = row.get("charge_error_raw") or row.get("charging_error")
        status, age = freshness_bucket(ts)

        state = NormalizedVehicleState(
            vehicle_id=vehicle_id,
            display_name=row.get("display_name") or row.get("model") or row.get("vehicle_name"),
            data_status=status,
            last_update_ts=ts,
            freshness_seconds=age,
            soc_pct=soc,
            is_plugged=is_plugged,
            is_charging=is_charging,
            range_km=_as_float(row.get("range_km") or row.get("remaining_range_km")),
            time_to_full_min=_as_int(row.get("time_to_full_min") or row.get("remaining_to_full_min")),
            charge_power_kw=_as_float(row.get("charge_power_kw") or row.get("charging_power_kw")),
            ac_current_limit_a=_as_float(row.get("ac_current_limit_a") or row.get("ac_limit_a")),
            battery_capacity_kwh=battery_capacity,
            charging_mode=row.get("charging_mode"),
            optimized_charging_preference=row.get("optimized_charging_preference"),
            charge_window_start=row.get("charge_window_start"),
            charge_window_end=row.get("charge_window_end"),
            odometer_km=_as_float(row.get("odometer_km")),
            travelled_distance_km=_as_float(row.get("travelled_distance_km")),
            plug_status_raw=row.get("plug_status_raw") or row.get("plug_status"),
            flap_lock_status_raw=row.get("flap_lock_status_raw") or row.get("flap_status"),
            charge_error_raw=charge_error,
            raw_fields=row,
        )
        state.charge_session_active = bool(is_plugged and is_charging and not charge_error) if None not in (is_plugged, is_charging) else None
        state.energy_needed_kwh = (battery_capacity * (100 - soc) / 100) if battery_capacity is not None and soc is not None else None
        out.append(state)
    return out


def apply_planner_derivations(state: NormalizedVehicleState, *, petrol_price_eur_per_l: float | None, petrol_consumption_l_per_100km: float | None, charger_max_power_kw: float | None) -> NormalizedVehicleState:
    missing: list[str] = []
    if state.ac_current_limit_a is not None:
        car_kw = state.ac_current_limit_a * 230.0 / 1000.0
    else:
        car_kw = None
    limits = [v for v in [car_kw, charger_max_power_kw] if v is not None and v > 0]
    state.effective_charge_power_limit_kw = min(limits) if limits else None
    blocking_fault = bool(state.charge_error_raw)
    state.planner_demand_active = bool(state.is_plugged and state.data_status not in {"stale", "error"} and not blocking_fault and (state.soc_pct is None or state.soc_pct < 100))

    if state.energy_needed_kwh is None and state.battery_capacity_kwh is not None and state.soc_pct is not None:
        state.energy_needed_kwh = state.battery_capacity_kwh * (100 - state.soc_pct) / 100
    if state.energy_needed_kwh is not None:
        state.planned_energy_kwh = max(0.0, state.energy_needed_kwh)
    if state.planned_energy_kwh is not None and state.effective_charge_power_limit_kw and state.effective_charge_power_limit_kw > 0 and state.last_update_ts:
        hours = state.planned_energy_kwh / state.effective_charge_power_limit_kw
        state.expected_full_charge_ts = state.last_update_ts + dt.timedelta(hours=hours)
        state.max_deliverable_kwh = state.effective_charge_power_limit_kw * 8.0
        if state.battery_capacity_kwh and state.soc_pct is not None:
            gained_pct = (state.max_deliverable_kwh / state.battery_capacity_kwh) * 100
            state.max_reachable_soc_pct = min(100.0, state.soc_pct + gained_pct)

    if petrol_price_eur_per_l is None or petrol_consumption_l_per_100km is None:
        missing.append("petrol inputs")
    if state.range_km is None:
        missing.append("range")
    if state.planned_energy_kwh is not None and state.effective_charge_power_limit_kw is not None:
        state.planned_charge_cost_eur = state.planned_energy_kwh * 0.25
    if not missing and state.range_km is not None:
        state.avoided_petrol_cost_eur = (state.range_km / 100.0) * petrol_consumption_l_per_100km * petrol_price_eur_per_l
    if state.avoided_petrol_cost_eur is not None and state.planned_charge_cost_eur is not None:
        state.net_economic_benefit_eur = state.avoided_petrol_cost_eur - state.planned_charge_cost_eur
    state.planner_priority = "high" if state.planner_demand_active else "normal"
    state.planner_status_text = "Missing prerequisites: " + ", ".join(sorted(set(missing))) if missing else "Planner inputs complete"
    return state
