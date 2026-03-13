from __future__ import annotations

import datetime as dt
from typing import Any


BMW_FRESH_DEFAULT_SECONDS = 300


def _parse_ts(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    num = _to_float(value)
    if num is None:
        return None
    return int(num)


def _bmw_threshold_seconds(ev_cfg: dict[str, Any]) -> int:
    raw = ev_cfg.get("bmw_healthcheck_seconds") if isinstance(ev_cfg, dict) else None
    try:
        out = int(raw)
        if out > 0:
            return out
    except (TypeError, ValueError):
        pass
    return BMW_FRESH_DEFAULT_SECONDS


def _freshness_seconds(vehicle: dict[str, Any]) -> int | None:
    raw = vehicle.get("freshness_seconds")
    if raw is not None:
        parsed = _to_int(raw)
        if parsed is not None:
            return max(0, parsed)
    ts = _parse_ts(vehicle.get("last_update_ts"))
    if ts is None:
        return None
    return max(0, int((dt.datetime.now(dt.timezone.utc) - ts.astimezone(dt.timezone.utc)).total_seconds()))


def _is_bmw_soc_usable(vehicle: dict[str, Any], threshold_seconds: int) -> tuple[bool, str, int | None]:
    soc = _to_float(vehicle.get("soc_pct"))
    freshness = _freshness_seconds(vehicle)
    if soc is None:
        return False, "unavailable", freshness
    if freshness is None:
        return False, "bmw_stale", None
    if freshness <= threshold_seconds:
        return True, "bmw", freshness
    return False, "bmw_stale", freshness


def build_unified_ev_status(
    *,
    ev_cfg: dict[str, Any],
    runtime_ev_cfg: dict[str, Any],
    bmw_provider_status: dict[str, Any],
    bmw_vehicles: dict[str, dict[str, Any]],
    evse_status: dict[str, Any],
) -> dict[str, Any]:
    enabled = bool((ev_cfg or {}).get("enabled", False))
    vehicle = next(iter((bmw_vehicles or {}).values()), {}) if isinstance(bmw_vehicles, dict) else {}
    threshold_seconds = _bmw_threshold_seconds(ev_cfg or {})
    warnings: list[str] = []

    soc_ok, soc_source, freshness_seconds = _is_bmw_soc_usable(vehicle, threshold_seconds)
    soc_pct = _to_float(vehicle.get("soc_pct")) if soc_ok else None

    ocpp_connected = bool((evse_status or {}).get("connected"))
    ocpp_enabled = bool((evse_status or {}).get("enabled"))

    if ocpp_connected:
        is_plugged = evse_status.get("is_plugged")
        plugged_source = "ocpp"
    else:
        is_plugged = vehicle.get("is_plugged")
        plugged_source = "bmw" if is_plugged is not None else "unavailable"

    if ocpp_connected:
        is_charging = evse_status.get("is_charging")
        charging_source = "ocpp"
    else:
        is_charging = vehicle.get("is_charging")
        charging_source = "bmw" if is_charging is not None else "unavailable"

    ocpp_power = _to_float(evse_status.get("power_kw"))
    bmw_power = _to_float(vehicle.get("charge_power_kw"))
    if ocpp_connected and ocpp_power is not None:
        charge_power_kw = ocpp_power
        charge_power_source = "ocpp"
    elif bmw_power is not None:
        charge_power_kw = bmw_power
        charge_power_source = "bmw"
    else:
        charge_power_kw = None
        charge_power_source = "unavailable"

    range_km = _to_float(vehicle.get("range_km"))
    range_source = "bmw" if range_km is not None else "bmw_missing"

    deadline_raw = str((ev_cfg or {}).get("ev_charge_deadline_time") or "").strip()
    deadline_time = deadline_raw or None
    deadline_state = "configured" if deadline_time else "not_configured"

    battery_capacity_kwh = _to_float(vehicle.get("battery_capacity_kwh"))
    energy_needed_kwh = _to_float(vehicle.get("energy_needed_kwh"))
    if energy_needed_kwh is None and battery_capacity_kwh is not None and soc_pct is not None:
        energy_needed_kwh = max(0.0, battery_capacity_kwh * (100.0 - soc_pct) / 100.0)

    limit_candidates = [
        _to_float(vehicle.get("effective_charge_power_limit_kw")),
        _to_float(runtime_ev_cfg.get("charger_max_power_kw")),
    ]
    limit_candidates = [v for v in limit_candidates if v is not None and v > 0]
    effective_limit_kw = min(limit_candidates) if limit_candidates else None

    full_charge_state = "unavailable"
    expected_full_charge_ts = None
    expected_full_charge_source = "unavailable"
    if not bool(is_plugged):
        full_charge_state = "not_plugged"
    elif not bool(is_charging):
        full_charge_state = "not_charging"
    elif soc_pct is None or energy_needed_kwh is None:
        full_charge_state = "waiting_for_soc"
    else:
        power_for_eta = charge_power_kw if charge_power_kw and charge_power_kw > 0 else effective_limit_kw
        if power_for_eta and power_for_eta > 0:
            eta_hours = energy_needed_kwh / power_for_eta
            expected_full_charge_ts = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=eta_hours)).replace(microsecond=0).isoformat()
            expected_full_charge_source = "ocpp_power" if charge_power_source == "ocpp" else "power_limit"
            full_charge_state = "ready"
        else:
            full_charge_state = "waiting_for_power_limit"

    freshness_label = "BMW freshness unknown"
    if freshness_seconds is not None:
        if freshness_seconds < 60:
            freshness_label = f"{freshness_seconds}s ago"
        elif freshness_seconds < 3600:
            freshness_label = f"{int(round(freshness_seconds / 60.0))}m ago"
        else:
            freshness_label = f"{freshness_seconds / 3600.0:.1f}h ago"

    if soc_source == "bmw_stale":
        warnings.append("Using last known BMW data; SOC is stale.")

    if bool(is_charging) and charge_power_kw is None:
        charge_power_state = "waiting_for_charger_data"
    elif bool(is_charging) is False:
        charge_power_state = "not_charging"
    elif charge_power_kw is not None:
        charge_power_state = "available"
    else:
        charge_power_state = "unavailable"

    out = {
        "enabled": enabled,
        "provider": {
            "bmw": bmw_provider_status or {},
            "ocpp": evse_status or {},
            "ocpp_connected": ocpp_connected,
            "ocpp_enabled": ocpp_enabled,
        },
        "sources": {
            "soc_pct": soc_source,
            "is_plugged": plugged_source,
            "is_charging": charging_source,
            "charge_power_kw": charge_power_source,
            "range_km": range_source,
            "expected_full_charge_ts": expected_full_charge_source,
            "deadline_time": "config",
        },
        "soc_pct": soc_pct,
        "soc_source": soc_source,
        "is_plugged": is_plugged,
        "plugged_source": plugged_source,
        "is_charging": is_charging,
        "charging_source": charging_source,
        "charge_power_kw": charge_power_kw,
        "charge_power_source": charge_power_source,
        "range_km": range_km,
        "range_source": range_source,
        "deadline_time": deadline_time,
        "deadline_state": deadline_state,
        "expected_full_charge_ts": expected_full_charge_ts,
        "expected_full_charge_source": expected_full_charge_source,
        "freshness_seconds": freshness_seconds,
        "freshness_label": freshness_label,
        "warnings": warnings,
        "field_states": {
            "soc_pct": "available" if soc_pct is not None else ("stale" if soc_source == "bmw_stale" else "unavailable"),
            "charge_power_kw": charge_power_state,
            "range_km": "available" if range_km is not None else "bmw_missing",
            "deadline_time": deadline_state,
            "expected_full_charge_ts": full_charge_state,
        },
        "derived": {
            "energy_needed_kwh": energy_needed_kwh,
            "effective_charge_power_limit_kw": effective_limit_kw,
        },
    }
    return out
