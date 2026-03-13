from __future__ import annotations

import datetime as dt
import os
from typing import Any


def is_app_debug_enabled(env: dict[str, str] | None = None) -> bool:
    """Return True when APP_DEBUG is explicitly enabled."""
    source = env if env is not None else os.environ
    raw = str(source.get("APP_DEBUG", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def summarize_ev_provider_state(
    provider_status: dict[str, Any] | None,
    *,
    has_vehicle: bool,
    soc_available: bool,
    vehicle_freshness_seconds: Any,
) -> dict[str, Any]:
    """Map backend provider_status into concise user-facing EV widget state."""
    status = provider_status if isinstance(provider_status, dict) else {}
    provider = str(status.get("provider_status") or "").strip().lower()
    data_status = str(status.get("data_status") or "").strip().lower()
    freshness = _to_float_or_none(vehicle_freshness_seconds)
    stale_data = data_status in {"stale", "partial", "error"} or (freshness is not None and freshness >= 1800.0)

    chips: list[str] = []
    helper = ""
    fallback = ""
    headline_override = ""

    if not status:
        chips = ["Waiting for BMW data"]
        fallback = "BMW data temporarily unavailable. Please try again shortly."
    elif provider == "disabled":
        chips = ["BMW disabled"]
        fallback = "EV integration is disabled. Enable BMW in settings to see car status."
    elif provider == "auth_required":
        chips = ["Auth required"]
        fallback = "BMW authorization required. Connect your BMW account to load vehicle status."
    elif not has_vehicle:
        if provider == "degraded":
            chips = ["No vehicle", "Degraded"]
            fallback = "No BMW vehicles found yet. Check vehicle mapping and refresh BMW data."
        else:
            chips = ["Waiting for BMW data"]
            fallback = "BMW connected. Waiting for vehicle data."
    else:
        if provider in {"healthy", "ready", "polling"}:
            chips.append("Connected")
        elif provider == "degraded":
            chips.append("Degraded")
        else:
            chips.append("BMW status unknown")

        if stale_data:
            chips.append("Stale")
            helper = "Using last known BMW vehicle data."
        if not soc_available:
            headline_override = "Vehicle data ready · battery level pending"

    return {
        "chips": chips[:2],
        "helper": helper,
        "fallback": fallback,
        "headline_override": headline_override,
        "provider_status": provider,
        "data_status": data_status,
    }


def summarize_ev_setup_state(
    provider_status: dict[str, Any] | None,
    *,
    ev_enabled: bool,
    has_client_id: bool,
    vehicle_count: int,
    has_device_flow_session: bool,
) -> dict[str, Any]:
    """Return user-friendly setup guidance for BMW CarData connection flow."""
    status = provider_status if isinstance(provider_status, dict) else {}
    provider = str(status.get("provider_status") or "").strip().lower()

    if not ev_enabled:
        return {"level": "info", "title": "BMW not connected", "detail": "Enable EV integration to set up BMW CarData."}
    if not has_client_id:
        return {
            "level": "warning",
            "title": "BMW client ID required",
            "detail": "Enter your BMW client id, then click Setup CarData connection.",
        }
    if has_device_flow_session:
        return {
            "level": "info",
            "title": "BMW authorization required",
            "detail": "Open the BMW page and enter this code, then click Check connection.",
        }
    if provider == "auth_required":
        return {
            "level": "warning",
            "title": "Reconnect required",
            "detail": "BMW authorization has expired or is missing. Click Setup CarData connection.",
        }
    if vehicle_count <= 0:
        return {"level": "warning", "title": "No BMW vehicles found", "detail": "Connect BMW and re-check your linked vehicles."}
    if vehicle_count == 1:
        return {"level": "success", "title": "1 vehicle linked", "detail": "Vehicle data ready."}
    return {"level": "info", "title": "Multiple vehicles found, choose one", "detail": "Select the active BMW vehicle for forecasts."}


def summarize_cardata_readiness(
    *,
    ev_enabled: bool,
    has_client_id: bool,
    provider_status: dict[str, Any] | None,
    has_vehicle: bool,
    has_device_flow_session: bool,
    vehicle_freshness_seconds: Any,
) -> dict[str, str | bool]:
    """Return compact readiness state for the bottom status strip."""
    if not ev_enabled:
        return {"required": False, "ready": True, "label": "CarData", "detail": "EV integration off"}

    if not has_client_id:
        return {"required": True, "ready": False, "label": "CarData", "detail": "Missing BMW client id"}

    if has_device_flow_session:
        return {"required": True, "ready": False, "label": "CarData", "detail": "Auth/setup required"}

    status = provider_status if isinstance(provider_status, dict) else {}
    provider = str(status.get("provider_status") or "").strip().lower()
    data_status = str(status.get("data_status") or "").strip().lower()
    freshness = _to_float_or_none(vehicle_freshness_seconds)
    is_stale_or_degraded = provider == "degraded" or data_status in {"stale", "partial", "error"} or (freshness is not None and freshness >= 1800.0)

    if provider == "auth_required":
        return {"required": True, "ready": False, "label": "CarData", "detail": "Auth/setup required"}

    if not has_vehicle:
        return {"required": True, "ready": False, "label": "CarData", "detail": "No linked vehicle"}

    if is_stale_or_degraded:
        return {"required": True, "ready": False, "label": "CarData", "detail": "Stale/degraded data"}

    return {"required": True, "ready": True, "label": "CarData", "detail": "Ready"}


def format_ev_bool(value: Any, true_label: str = "Yes", false_label: str = "No", unknown_label: str = "—") -> str:
    if value is None:
        return unknown_label
    return true_label if bool(value) else false_label


def format_ev_datetime(value: Any, unknown_label: str = "—") -> str:
    if value is None:
        return unknown_label
    raw = str(value).strip()
    if not raw:
        return unknown_label
    try:
        normalized = raw.replace("Z", "+00:00")
        ts = dt.datetime.fromisoformat(normalized)
        if ts.tzinfo is not None:
            ts = ts.astimezone()
        return ts.strftime("%H:%M")
    except Exception:
        return raw


def format_ev_freshness(seconds: Any, unknown_label: str = "—") -> str:
    age = _to_float_or_none(seconds)
    if age is None:
        return unknown_label
    age = max(0.0, age)
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(round(age / 60.0))}m ago"
    return f"{age / 3600.0:.1f}h ago"


def format_ev_kw(value: Any, unknown_label: str = "—") -> str:
    num = _to_float_or_none(value)
    if num is None:
        return unknown_label
    return f"{num:.1f} kW"


def format_ev_kwh(value: Any, unknown_label: str = "—") -> str:
    num = _to_float_or_none(value)
    if num is None:
        return unknown_label
    return f"{num:.1f} kWh"


def format_ev_km(value: Any, unknown_label: str = "—") -> str:
    num = _to_float_or_none(value)
    if num is None:
        return unknown_label
    return f"{num:.0f} km"


def format_ev_time_to_full_minutes(value: Any, unknown_label: str = "—") -> str:
    mins = _to_float_or_none(value)
    if mins is None:
        return unknown_label
    mins = max(0.0, mins)
    if mins < 60:
        return f"{int(round(mins))} min"
    hours = int(mins // 60)
    rem = int(round(mins - (hours * 60)))
    if rem == 0:
        return f"{hours}h"
    return f"{hours}h {rem}m"


def weather_code_to_icon(weather_code: int | float | str | None) -> str:
    """Map WMO weather codes or common labels to visible emoji icons.

    Explicitly keep common WMO codes on fully visible glyphs (no blank variation selectors).
    """
    if weather_code is None:
        return "❓"

    if isinstance(weather_code, str):
        raw = weather_code.strip()
        if not raw:
            return "❓"

        try:
            return weather_code_to_icon(int(float(raw)))
        except Exception:
            pass

        key = raw.lower()
        label_map = {
            "clear": "☀️",
            "sunny": "☀️",
            "mainly_clear": "🌤️",
            "partly_cloudy": "⛅",
            "cloudy": "☁️",
            "overcast": "☁️",
            "fog": "🌫️",
            "mist": "🌫️",
            "drizzle": "🌦️",
            "rain": "🌧️",
            "showers": "🌦️",
            "rain_showers": "🌦️",
            "snow": "🌨️",
            "snowfall": "🌨️",
            "snow_showers": "🌨️",
            "sleet": "🌨️",
            "thunderstorm": "⛈️",
        }
        return label_map.get(key, "❓")

    try:
        code = int(weather_code)
    except Exception:
        return "❓"

    if code == 0:
        return "☀️"
    if code == 1:
        return "🌤️"
    if code == 2:
        return "⛅"
    if code == 3:
        return "☁️"
    if code in (45, 48):
        return "🌫️"
    if 51 <= code <= 57:
        return "🌦️"
    if 61 <= code <= 67:
        return "🌧️"
    if 71 <= code <= 77:
        return "🌨️"
    if 80 <= code <= 82:
        return "🌦️"
    if code in (85, 86):
        return "🌨️"
    if code in (95, 96, 99):
        return "⛈️"
    return "❓"


def weather_code_to_label(weather_code: int | float | str | None) -> str:
    """Return a readable label for weather code values."""
    if weather_code is None:
        return "Unknown"

    if isinstance(weather_code, str):
        raw = weather_code.strip()
        if not raw:
            return "Unknown"
        try:
            code = int(float(raw))
        except Exception:
            key = raw.lower()
            label_map = {
                "clear": "Clear sky",
                "sunny": "Clear sky",
                "mainly_clear": "Mainly clear",
                "partly_cloudy": "Partly cloudy",
                "cloudy": "Cloudy",
                "overcast": "Overcast",
                "fog": "Fog",
                "mist": "Fog",
                "drizzle": "Drizzle",
                "rain": "Rain",
                "showers": "Rain showers",
                "rain_showers": "Rain showers",
                "snow": "Snow",
                "snowfall": "Snow",
                "snow_showers": "Snow showers",
                "sleet": "Snow showers",
                "thunderstorm": "Thunderstorm",
            }
            return label_map.get(key, "Unknown")
    else:
        try:
            code = int(weather_code)
        except Exception:
            return "Unknown"

    if code == 0:
        return "Clear sky"
    if code == 1:
        return "Mainly clear"
    if code == 2:
        return "Partly cloudy"
    if code == 3:
        return "Overcast"
    if code in (45, 48):
        return "Fog"
    if 51 <= code <= 57:
        return "Drizzle"
    if 61 <= code <= 67:
        return "Rain"
    if 71 <= code <= 77:
        return "Snow"
    if 80 <= code <= 82:
        return "Rain showers"
    if code in (85, 86):
        return "Snow showers"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return "Unknown"


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def resolve_pv_outlook_savings(pv_quality_dict: dict | None) -> dict[str, Any]:
    """Resolve consistent PV OUTLOOK savings display values."""
    data = pv_quality_dict or {}

    cycle_grid_only = _to_float_or_none(data.get("grid_only_cost_eur_cycle"))
    cycle_isystem = _to_float_or_none(data.get("isystem_cost_eur_cycle"))
    cycle_benefit = _to_float_or_none(data.get("benefit_vs_grid_only_eur_cycle"))

    cycle_base = _to_float_or_none(data.get("baseline_cost_eur_cycle"))
    cycle_plan = _to_float_or_none(data.get("plan_cost_eur_cycle"))
    cycle_savings = _to_float_or_none(data.get("savings_eur_cycle"))

    tomorrow_base = _to_float_or_none(data.get("baseline_cost_eur_tomorrow"))
    tomorrow_plan = _to_float_or_none(data.get("plan_cost_eur_tomorrow"))
    tomorrow_savings = _to_float_or_none(data.get("savings_eur_tomorrow"))

    total_base = _to_float_or_none(data.get("baseline_cost_eur_total"))
    total_plan = _to_float_or_none(data.get("plan_cost_eur_total"))
    total_savings = _to_float_or_none(data.get("savings_eur_total"))

    preferred_scope = str(data.get("savings_preferred_scope") or "").strip().lower() or None

    has_new_cycle = cycle_benefit is not None
    has_cycle = (
        cycle_benefit is not None
        or cycle_base is not None
        or cycle_plan is not None
        or cycle_savings is not None
        or cycle_grid_only is not None
        or cycle_isystem is not None
    )
    has_tomorrow = tomorrow_base is not None or tomorrow_plan is not None or tomorrow_savings is not None
    has_total = total_base is not None or total_plan is not None or total_savings is not None

    if preferred_scope == "tomorrow" and has_tomorrow:
        display_scope = "tomorrow"
        base_cost = tomorrow_base
        plan_cost = tomorrow_plan
        reported_savings = tomorrow_savings
    elif has_cycle:
        display_scope = "cycle"
        if has_new_cycle:
            base_cost = cycle_grid_only
            plan_cost = cycle_isystem
            reported_savings = cycle_benefit
        else:
            base_cost = cycle_base
            plan_cost = cycle_plan
            reported_savings = cycle_savings
    elif has_tomorrow:
        display_scope = "tomorrow"
        base_cost = tomorrow_base
        plan_cost = tomorrow_plan
        reported_savings = tomorrow_savings
    else:
        display_scope = "total"
        if has_total:
            base_cost = total_base
            plan_cost = total_plan
            reported_savings = total_savings
        else:
            base_cost = None
            plan_cost = None
            reported_savings = None

    if base_cost is not None and plan_cost is not None:
        savings = base_cost - plan_cost
    else:
        savings = reported_savings

    def _convert_hourly(raw: Any) -> list[float] | None:
        if not (isinstance(raw, list) and len(raw) == 24):
            return None
        converted: list[float] = []
        for val in raw:
            num = _to_float_or_none(val)
            if num is None:
                return None
            converted.append(num)
        return converted

    hourly_cycle = _convert_hourly(data.get("hourly_benefit_vs_grid_only_eur_cycle_cash"))
    if hourly_cycle is None:
        hourly_cycle = _convert_hourly(data.get("hourly_savings_eur_cycle"))

    hourly_tomorrow = _convert_hourly(data.get("hourly_benefit_vs_grid_only_eur_tomorrow_cash"))
    if hourly_tomorrow is None:
        hourly_tomorrow = _convert_hourly(data.get("hourly_savings_eur_tomorrow"))

    hourly_labels_raw = data.get("hourly_benefit_cycle_hour_labels")
    if not (isinstance(hourly_labels_raw, list) and len(hourly_labels_raw) == 24):
        hourly_labels_raw = data.get("hourly_savings_cycle_hour_labels")
    hourly_cycle_labels = (
        [str(x) for x in hourly_labels_raw]
        if isinstance(hourly_labels_raw, list) and len(hourly_labels_raw) == 24
        else None
    )

    bars_scope = "cycle"
    hourly: list[float] | None = hourly_cycle
    hourly_labels = [f"{h:02d}:00" for h in range(24)]
    if hourly is None:
        bars_scope = "tomorrow"
        hourly = hourly_tomorrow
    if bars_scope == "cycle" and hourly_cycle_labels is not None:
        hourly_labels = hourly_cycle_labels

    horizon_label = str(data.get("savings_horizon_label") or "").strip() or None
    if bars_scope == "tomorrow":
        detail_note = "⏱️ Bars: tomorrow (00–24)"
    else:
        detail_note = "⏱️ Bars: cycle"

    if preferred_scope == "tomorrow":
        note = "Cycle savings unavailable due to missing PV forecast coverage; showing tomorrow-only savings."
    elif display_scope == "cycle":
        if bars_scope == "tomorrow":
            note = (
                "Grid only vs iSystem Cycle savings shown; bars: tomorrow (00–24). "
                "(24u vanaf start daluren (vast 24u venster))."
            )
        else:
            note = "Grid only vs iSystem Cycle savings shown (24u vanaf start daluren (vast 24u venster))."
    elif display_scope == "tomorrow":
        note = "PV OUTLOOK savings shown (tomorrow only)."
    else:
        note = "PV OUTLOOK savings shown."

    return {
        "base_cost": base_cost,
        "plan_cost": plan_cost,
        "savings": savings,
        "hourly": hourly,
        "hourly_labels": hourly_labels,
        "bars_scope": bars_scope,
        "display_scope": display_scope,
        "terminal_battery_value_eur_cycle": _to_float_or_none(data.get("terminal_battery_value_eur_cycle")),
        "plan_cost_eur_cycle_cash": _to_float_or_none(data.get("plan_cost_eur_cycle_cash")),
        "cycle_start_soc_pct": _to_float_or_none(data.get("cycle_start_soc_pct")),
        "cycle_end_soc_pct": _to_float_or_none(data.get("cycle_end_soc_pct")),
        "savings_cycle_terminal_value_applied": bool(data.get("savings_cycle_terminal_value_applied", False)),
        "horizon_label": horizon_label,
        "horizon_start_iso": str(data.get("savings_horizon_start_iso") or "").strip() or None,
        "horizon_end_iso": str(data.get("savings_horizon_end_iso") or "").strip() or None,
        "detail_note": detail_note,
        "note": note,
    }
