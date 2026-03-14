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
    return int(num) if num is not None else None


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


def _bmw_diag_field(diag: dict[str, Any], field_name: str) -> dict[str, Any]:
    raw = diag.get("raw_field_evidence") if isinstance(diag.get("raw_field_evidence"), dict) else {}
    node = raw.get(f"{field_name}_evidence")
    return node if isinstance(node, dict) else {}


def _setup_state_for_field(diag: dict[str, Any], field_name: str) -> str:
    field = _bmw_diag_field(diag, field_name)
    if field.get("descriptor_active") is False:
        return "descriptor_inactive"
    missing = diag.get("missing_critical_descriptors") if isinstance(diag.get("missing_critical_descriptors"), list) else []
    if field.get("descriptor") in set(str(x) for x in missing):
        return "descriptor_inactive"
    if diag.get("reprobe_triggered") and missing:
        return "reprobe_pending"
    return "ready"


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
    bmw_diag = bmw_provider_status.get("bmw_ev_diagnostics") if isinstance(bmw_provider_status, dict) and isinstance(bmw_provider_status.get("bmw_ev_diagnostics"), dict) else {}

    soc_ok, soc_source, freshness_seconds = _is_bmw_soc_usable(vehicle, threshold_seconds)
    soc_pct = _to_float(vehicle.get("soc_pct")) if soc_ok else None

    ocpp_connected = bool((evse_status or {}).get("connected"))
    ocpp_enabled = bool((evse_status or {}).get("enabled"))

    is_plugged = vehicle.get("is_plugged")
    plugged_source = "bmw" if is_plugged is not None else "unavailable"

    is_charging = vehicle.get("is_charging")
    charging_setup_state = _setup_state_for_field(bmw_diag, "is_charging")
    charging_evidence = _bmw_diag_field(bmw_diag, "is_charging")
    charging_source = "bmw" if is_charging is not None else ("setup_incomplete" if charging_setup_state != "ready" else "bmw_missing" if charging_evidence.get("raw_value_present") is False else "unavailable")

    bmw_power = _to_float(vehicle.get("charge_power_kw"))
    charge_power_setup_state = _setup_state_for_field(bmw_diag, "charge_power_kw")
    charge_power_evidence = _bmw_diag_field(bmw_diag, "charge_power_kw")
    if bmw_power is not None:
        charge_power_kw = bmw_power
        charge_power_source = "bmw"
    else:
        charge_power_kw = None
        charge_power_source = "setup_incomplete" if charge_power_setup_state != "ready" else "bmw_missing" if charge_power_evidence.get("raw_value_present") is False else "unavailable"

    range_km = _to_float(vehicle.get("range_km"))
    range_setup_state = _setup_state_for_field(bmw_diag, "range_km")
    range_evidence = _bmw_diag_field(bmw_diag, "range_km")
    if range_km is not None:
        range_source = "bmw"
    elif range_setup_state != "ready":
        range_source = "setup_incomplete"
    elif range_evidence.get("raw_value_present") is False or not range_evidence:
        range_source = "bmw_missing"
    else:
        range_source = "unavailable"

    deadline_raw = str((ev_cfg or {}).get("ev_charge_deadline_time") or "").strip()
    deadline_time = deadline_raw or None
    deadline_state = "configured" if deadline_time else "not_configured"

    battery_capacity_kwh = _to_float(vehicle.get("battery_capacity_kwh"))
    energy_needed_kwh = _to_float(vehicle.get("energy_needed_kwh"))
    if energy_needed_kwh is None and battery_capacity_kwh is not None and soc_pct is not None:
        energy_needed_kwh = max(0.0, battery_capacity_kwh * (100.0 - soc_pct) / 100.0)

    limit_candidates = [_to_float(vehicle.get("effective_charge_power_limit_kw")), _to_float(runtime_ev_cfg.get("charger_max_power_kw"))]
    limit_candidates = [v for v in limit_candidates if v is not None and v > 0]
    effective_limit_kw = min(limit_candidates) if limit_candidates else None

    bmw_expected_full_charge_ts = _parse_ts(vehicle.get("expected_full_charge_ts"))
    bmw_time_to_full_min = _to_int(vehicle.get("time_to_full_min"))
    eta_setup_state = _setup_state_for_field(bmw_diag, "time_to_full_min")

    full_charge_state = "unavailable"
    expected_full_charge_ts = None
    expected_full_charge_source = "unavailable"
    if is_plugged is False:
        full_charge_state = "not_plugged"
    elif is_charging is False:
        full_charge_state = "not_charging"
    elif is_plugged is None or is_charging is None:
        full_charge_state = "waiting_for_bmw_status"
    elif bmw_expected_full_charge_ts is not None:
        expected_full_charge_ts = bmw_expected_full_charge_ts.replace(microsecond=0).isoformat()
        expected_full_charge_source = "bmw"
        full_charge_state = "ready"
    elif bmw_time_to_full_min is not None and bmw_time_to_full_min >= 0:
        expected_full_charge_ts = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=bmw_time_to_full_min)).replace(microsecond=0).isoformat()
        expected_full_charge_source = "bmw_time_to_full"
        full_charge_state = "ready"
    elif soc_pct is None or energy_needed_kwh is None:
        full_charge_state = "waiting_for_soc"
    else:
        power_for_eta = charge_power_kw if charge_power_kw and charge_power_kw > 0 else effective_limit_kw
        if power_for_eta and power_for_eta > 0:
            eta_hours = energy_needed_kwh / power_for_eta
            expected_full_charge_ts = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=eta_hours)).replace(microsecond=0).isoformat()
            expected_full_charge_source = "power_limit"
            full_charge_state = "ready"
        else:
            full_charge_state = "waiting_for_bmw_eta" if is_charging is True else "waiting_for_power_limit"

    if eta_setup_state != "ready" and expected_full_charge_ts is None and full_charge_state in {"waiting_for_bmw_eta", "waiting_for_bmw_status", "unavailable"}:
        full_charge_state = "setup_incomplete"

    freshness_label = "BMW freshness unknown"
    if freshness_seconds is not None:
        freshness_label = f"{freshness_seconds}s ago" if freshness_seconds < 60 else f"{int(round(freshness_seconds / 60.0))}m ago" if freshness_seconds < 3600 else f"{freshness_seconds / 3600.0:.1f}h ago"

    if soc_source == "bmw_stale":
        warnings.append("Using last known BMW data; SOC is stale.")

    if charge_power_setup_state != "ready":
        charge_power_state = "setup_incomplete"
    elif is_charging is True and charge_power_kw is None:
        charge_power_state = "waiting_for_bmw_power"
    elif is_charging is False:
        charge_power_state = "not_charging"
    elif is_charging is None:
        charge_power_state = "waiting_for_bmw_status"
    elif charge_power_kw is not None:
        charge_power_state = "available"
    else:
        charge_power_state = "unavailable"

    return {
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
            "bmw_ev_diagnostics": "bmw",
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
        "time_to_full_min": bmw_time_to_full_min,
        "freshness_seconds": freshness_seconds,
        "freshness_label": freshness_label,
        "warnings": warnings,
        "field_states": {
            "soc_pct": "available" if soc_pct is not None else ("stale" if soc_source == "bmw_stale" else "unavailable"),
            "range_setup": range_setup_state,
            "charging_setup": charging_setup_state,
            "charge_power_setup": charge_power_setup_state,
            "charge_power_kw": charge_power_state,
            "range_km": "available" if range_km is not None else ("setup_incomplete" if range_setup_state != "ready" else "bmw_missing" if range_evidence.get("raw_value_present") is False or not range_evidence else "unavailable"),
            "deadline_time": deadline_state,
            "expected_full_charge_ts": full_charge_state,
        },
        "bmw_ev_diagnostics": bmw_diag,
        "derived": {
            "energy_needed_kwh": energy_needed_kwh,
            "effective_charge_power_limit_kw": effective_limit_kw,
        },
    }
