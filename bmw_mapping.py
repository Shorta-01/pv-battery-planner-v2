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




def _extract_endpoint_payload(payload: dict[str, Any], path: str) -> dict[str, Any] | None:
    node = payload.get(path)
    if isinstance(node, dict):
        return node
    if isinstance(payload.get("endpoint"), str) and payload.get("endpoint") == path and isinstance(payload.get("payload"), dict):
        return payload.get("payload")
    return None


def _extract_vehicles_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    vehicles_resp = _extract_endpoint_payload(payload, "/v1/vehicles")
    if isinstance(vehicles_resp, dict) and isinstance(vehicles_resp.get("vehicles"), list):
        return [row for row in vehicles_resp.get("vehicles") if isinstance(row, dict)]
    if isinstance(payload.get("vehicles"), list):
        return [row for row in payload.get("vehicles") if isinstance(row, dict)]
    return []

def _vehicle_mapping_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _extract_endpoint_payload(payload, "/v1/vehicle-mappings")
    entries = rows.get("vehicleMappings") if isinstance(rows, dict) else None
    if not isinstance(entries, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        vin = str(entry.get("vin") or "").strip()
        if vin:
            out[vin] = entry
    return out


def map_bmw_payload_to_vehicle_states(payload: dict[str, Any]) -> list[NormalizedVehicleState]:
    mappings_by_vin = _vehicle_mapping_index(payload)
    vehicles = _extract_vehicles_list(payload)
    if not vehicles:
        return []

    out: list[NormalizedVehicleState] = []
    for row in vehicles:
        vehicle_id = str(row.get("vin") or "").strip()
        if not vehicle_id:
            continue

        charging = row.get("charging") if isinstance(row.get("charging"), dict) else {}
        battery = row.get("battery") if isinstance(row.get("battery"), dict) else {}
        range_data = row.get("range") if isinstance(row.get("range"), dict) else {}
        charge_settings = row.get("chargeSettings") if isinstance(row.get("chargeSettings"), dict) else {}
        mapped = mappings_by_vin.get(vehicle_id, {})

        plug_status_raw = charging.get("plugConnectionState")
        is_plugged = True if plug_status_raw in {"CONNECTED", "PLUGGED"} else False if plug_status_raw in {"DISCONNECTED", "UNPLUGGED"} else None

        charging_state = charging.get("chargingState")
        is_charging = True if charging_state in {"CHARGING", "ACTIVE"} else False if charging_state in {"NOT_CHARGING", "COMPLETED", "ERROR"} else None

        ts = parse_dt(row.get("lastUpdatedAt"))
        status, age = freshness_bucket(ts)

        state = NormalizedVehicleState(
            vehicle_id=vehicle_id,
            display_name=mapped.get("displayName") or mapped.get("name") or row.get("model"),
            data_status=status,
            last_update_ts=ts,
            freshness_seconds=age,
            soc_pct=_as_float(battery.get("socPercent")),
            is_plugged=is_plugged,
            is_charging=is_charging,
            range_km=_as_float(range_data.get("electricKm")),
            time_to_full_min=_as_int(charging.get("remainingTimeToFullMinutes")),
            charge_power_kw=_as_float(charging.get("chargePowerKw")),
            ac_current_limit_a=_as_float(charge_settings.get("acCurrentLimitA")),
            battery_capacity_kwh=_as_float(battery.get("capacityKwh")),
            charging_mode=charge_settings.get("chargingMode"),
            optimized_charging_preference=charge_settings.get("optimizedChargingPreference"),
            charge_window_start=charge_settings.get("chargeWindowStart"),
            charge_window_end=charge_settings.get("chargeWindowEnd"),
            odometer_km=_as_float(row.get("odometerKm")),
            travelled_distance_km=_as_float(row.get("travelledDistanceKm")),
            plug_status_raw=str(plug_status_raw) if plug_status_raw is not None else None,
            flap_lock_status_raw=charging.get("flapLockStatus"),
            charge_error_raw=charging.get("chargeError") or charging.get("errorCode"),
            raw_fields=row,
        )
        state.charge_session_active = bool(is_plugged and is_charging and not state.charge_error_raw) if None not in (is_plugged, is_charging) else None
        if state.battery_capacity_kwh is not None and state.soc_pct is not None:
            state.energy_needed_kwh = state.battery_capacity_kwh * (100 - state.soc_pct) / 100
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
