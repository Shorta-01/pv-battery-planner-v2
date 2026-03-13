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




def _dig(node: Any, *path: str) -> Any:
    cur = node
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _first(*values: Any) -> Any:
    for v in values:
        if v is not None and v != "":
            return v
    return None


def _as_bool_from_states(value: Any, *, true_values: set[str], false_values: set[str]) -> bool | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in true_values:
        return True
    if s in false_values:
        return False
    return None


def _descriptor_value(telematic: dict[str, Any], key: str, field: str = "value") -> Any:
    node = telematic.get(key)
    if isinstance(node, dict):
        return node.get(field)
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


def _extract_endpoint_payload(payload: dict[str, Any], path: str) -> dict[str, Any] | None:
    node = payload.get(path)
    if node is None:
        node = payload.get(f"GET {path}")
    if isinstance(node, dict):
        return node
    if isinstance(payload.get("endpoint"), str) and payload.get("endpoint") == path and isinstance(payload.get("payload"), dict):
        return payload.get("payload")
    return None


def _path_vin(path: str) -> str | None:
    chunks = [c for c in path.split("/") if c]
    if len(chunks) >= 3 and chunks[0] == "customers" and chunks[1] == "vehicles":
        return None if chunks[2] == "mappings" else chunks[2]
    return None


def _vehicle_mapping_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    for path in ("/customers/vehicles/mappings",):
        rows = _extract_endpoint_payload(payload, path)
        entries = rows.get("vehicleMappings") if isinstance(rows, dict) else rows.get("vehicles") if isinstance(rows, dict) else None
        if isinstance(entries, list):
            out: dict[str, dict[str, Any]] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                vid = str(entry.get("vin") or entry.get("vehicleId") or entry.get("id") or "").strip()
                if vid:
                    out[vid] = entry
            return out
    return {}




def _normalize_endpoint_key(key: str) -> str:
    k = key.strip()
    if " " in k and k.split(" ", 1)[0] in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        return k.split(" ", 1)[1]
    return k
def _extract_vehicles_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    for key, node in payload.items():
        if not isinstance(key, str) or not isinstance(node, dict) or "_error" in node:
            continue
        endpoint_key = _normalize_endpoint_key(key)
        vin = _path_vin(endpoint_key)
        if not vin:
            continue
        merged = out.get(vin, {"vin": vin, "vehicleId": vin})
        endpoint_no_query = endpoint_key.split("?", 1)[0]
        if endpoint_no_query.endswith("/basicData"):
            merged["basicData"] = node
        elif endpoint_no_query.endswith("/telematicData"):
            merged["telematicData"] = node
        elif endpoint_no_query.endswith("/chargingprofile"):
            merged["chargingProfile"] = node
        elif endpoint_no_query.endswith("/charging"):
            merged["charging"] = node
        else:
            merged.update(node)
        out[vin] = merged

    if isinstance(payload.get("endpoint"), str) and isinstance(payload.get("payload"), dict):
        endpoint = str(payload.get("endpoint"))
        endpoint_no_query = endpoint.split("?", 1)[0]
        wrapped_payload = dict(payload.get("payload") or {})
        vin = _path_vin(endpoint_no_query)
        if vin:
            merged = out.get(vin, {"vin": vin, "vehicleId": vin})
            if endpoint_no_query.endswith("/basicData"):
                if isinstance(wrapped_payload.get("vehicles"), list) and wrapped_payload.get("vehicles"):
                    first = wrapped_payload.get("vehicles")[0]
                    merged["basicData"] = first if isinstance(first, dict) else wrapped_payload
                else:
                    merged["basicData"] = wrapped_payload
            elif endpoint_no_query.endswith("/telematicData"):
                merged["telematicData"] = wrapped_payload
            elif endpoint_no_query.endswith("/chargingprofile"):
                merged["chargingProfile"] = wrapped_payload
            else:
                merged.update(wrapped_payload)
            out[vin] = merged

    return list(out.values())


def map_bmw_payload_to_vehicle_states(payload: dict[str, Any]) -> list[NormalizedVehicleState]:
    mappings_by_vin = _vehicle_mapping_index(payload)
    vehicles = _extract_vehicles_list(payload)
    if not vehicles:
        return []

    out: list[NormalizedVehicleState] = []
    for row in vehicles:
        vehicle_id = str(row.get("vin") or row.get("vehicleId") or row.get("id") or "").strip()
        if not vehicle_id:
            continue

        basic = row.get("basicData") if isinstance(row.get("basicData"), dict) else row
        mapped = mappings_by_vin.get(vehicle_id, row.get("_mapping") if isinstance(row.get("_mapping"), dict) else {})
        telematic = row.get("telematicData") if isinstance(row.get("telematicData"), dict) else {}
        if isinstance(telematic.get("telematicData"), dict):
            telematic = telematic.get("telematicData")
        charging_profile = row.get("chargingProfile") if isinstance(row.get("chargingProfile"), dict) else {}

        charging = basic.get("charging") if isinstance(basic.get("charging"), dict) else {}
        battery = basic.get("battery") if isinstance(basic.get("battery"), dict) else {}
        range_data = basic.get("range") if isinstance(basic.get("range"), dict) else {}
        charge_settings = basic.get("chargeSettings") if isinstance(basic.get("chargeSettings"), dict) else {}

        tele_charging = telematic.get("charging") if isinstance(telematic.get("charging"), dict) else {}
        tele_battery = telematic.get("battery") if isinstance(telematic.get("battery"), dict) else {}
        tele_range = telematic.get("range") if isinstance(telematic.get("range"), dict) else {}
        tele_charge_settings = telematic.get("chargeSettings") if isinstance(telematic.get("chargeSettings"), dict) else {}

        if not charging and tele_charging:
            charging = tele_charging
        if not charging and isinstance(charging_profile.get("charging"), dict):
            charging = charging_profile.get("charging")
        if not charge_settings and tele_charge_settings:
            charge_settings = tele_charge_settings
        if not charge_settings and isinstance(charging_profile.get("chargeSettings"), dict):
            charge_settings = charging_profile.get("chargeSettings")

        charging_status_raw = _first(
            _descriptor_value(telematic, "vehicle.drivetrain.electricEngine.charging.status"),
            _dig(telematic, "chargingStatus"),
            _dig(telematic, "chargingState"),
            _dig(tele_charging, "chargingState"),
            charging.get("chargingState"),
        )
        is_charging = _as_bool_from_states(
            charging_status_raw,
            true_values={"CHARGING", "CHARGINGACTIVE", "ACTIVE", "IN_PROGRESS"},
            false_values={"NOT_CHARGING", "CHARGINGINACTIVE", "INACTIVE", "COMPLETED", "ERROR", "IDLE"},
        )
        if is_charging is None:
            is_charging = _as_bool_from_states(_first(_dig(telematic, "isCharging"), _dig(tele_charging, "isCharging")), true_values={"TRUE", "1"}, false_values={"FALSE", "0"})

        plug_status_raw = _first(
            _descriptor_value(telematic, "vehicle.body.chargingPort.status"),
            _dig(telematic, "plugStatus"),
            _dig(telematic, "plugConnectionState"),
            _dig(tele_charging, "plugConnectionState"),
            _dig(tele_charging, "plugState"),
            charging.get("plugConnectionState"),
        )
        is_plugged = _as_bool_from_states(
            plug_status_raw,
            true_values={"CONNECTED", "PLUGGED", "PLUGGED_IN"},
            false_values={"DISCONNECTED", "UNPLUGGED", "NOT_PLUGGED", "OPEN", "NOT_CONNECTED"},
        )
        if is_plugged is None:
            is_plugged = _as_bool_from_states(_first(_dig(telematic, "isPlugged"), _dig(tele_charging, "isPlugged")), true_values={"TRUE", "1"}, false_values={"FALSE", "0"})

        descriptor_timestamps = [
            parse_dt(v.get("timestamp"))
            for v in telematic.values()
            if isinstance(v, dict) and isinstance(v.get("timestamp"), str)
        ]
        descriptor_timestamps = [t for t in descriptor_timestamps if t is not None]
        ts = max(descriptor_timestamps) if descriptor_timestamps else parse_dt(_first(
            _dig(telematic, "lastUpdatedAt"),
            _dig(telematic, "statusUpdatedAt"),
            _dig(telematic, "updateTime"),
            _dig(telematic, "timestamp"),
            basic.get("lastUpdatedAt"),
            basic.get("statusUpdatedAt"),
            basic.get("updateTime"),
        ))
        status, age = freshness_bucket(ts) if ts else ("partial", None)
        if telematic and status == "partial":
            status = "live_partial"

        ac_current_limit_a = _as_float(_first(
            _descriptor_value(telematic, "vehicle.powertrain.electric.battery.charging.acLimit.selected"),
            _dig(telematic, "acCurrentLimitA"),
            _dig(tele_charge_settings, "acCurrentLimitA"),
            charge_settings.get("acCurrentLimitA"),
        ))

        state = NormalizedVehicleState(
            vehicle_id=vehicle_id,
            display_name=_first(basic.get("modelName"), basic.get("model"), mapped.get("displayName"), mapped.get("name"), basic.get("displayName")),
            data_status=status,
            last_update_ts=ts,
            freshness_seconds=age,
            soc_pct=_as_float(_first(_dig(telematic, "socPct"), _dig(telematic, "soc"), _dig(tele_battery, "socPercent"), _dig(tele_battery, "socPct"), battery.get("socPercent"))),
            is_plugged=is_plugged,
            is_charging=is_charging,
            range_km=_as_float(_first(_dig(telematic, "rangeKm"), _dig(telematic, "remainingRangeKm"), _dig(tele_range, "electricKm"), range_data.get("electricKm"))),
            time_to_full_min=_as_int(_first(_dig(telematic, "timeToFullMin"), _dig(telematic, "remainingTimeToFullMinutes"), _dig(tele_charging, "remainingTimeToFullMinutes"), charging.get("remainingTimeToFullMinutes"))),
            charge_power_kw=_as_float(_first(_dig(telematic, "chargePowerKw"), _dig(tele_charging, "chargePowerKw"), charging.get("chargePowerKw"))),
            ac_current_limit_a=ac_current_limit_a,
            battery_capacity_kwh=_as_float(_first(_dig(tele_battery, "capacityKwh"), battery.get("capacityKwh"))),
            charging_mode=_first(charge_settings.get("chargingMode"), str(charging_status_raw) if charging_status_raw is not None else None),
            optimized_charging_preference=charge_settings.get("optimizedChargingPreference"),
            charge_window_start=charge_settings.get("chargeWindowStart"),
            charge_window_end=charge_settings.get("chargeWindowEnd"),
            odometer_km=_as_float(basic.get("odometerKm")),
            travelled_distance_km=_as_float(basic.get("travelledDistanceKm")),
            plug_status_raw=str(plug_status_raw) if plug_status_raw is not None else None,
            flap_lock_status_raw=charging.get("flapLockStatus"),
            charge_error_raw=_first(_dig(telematic, "chargeError"), _dig(telematic, "errorCode"), _dig(tele_charging, "chargeError"), charging.get("chargeError"), charging.get("errorCode")),
            raw_fields=row,
        )
        state.charge_session_active = bool(is_plugged and is_charging and not state.charge_error_raw) if None not in (is_plugged, is_charging) else None
        state.field_availability = {
            "charging_status": charging_status_raw is not None,
            "plug_status": plug_status_raw is not None,
            "ac_current_limit_a": ac_current_limit_a is not None,
            "last_update_ts": ts is not None,
        }
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
