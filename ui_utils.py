from __future__ import annotations

from typing import Any


def weather_code_to_icon(weather_code: int | float | str | None) -> str:
    """Map WMO weather codes or common labels to visible emoji icons.

    Explicitly keep common WMO codes on fully visible glyphs (no blank variation selectors).
    """
    if weather_code is None:
        return "☁️"

    if isinstance(weather_code, str):
        raw = weather_code.strip()
        if not raw:
            return "☁️"

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
        return label_map.get(key, "☁️")

    try:
        code = int(weather_code)
    except Exception:
        return "☁️"

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
    return "☁️"


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

    cycle_base = _to_float_or_none(data.get("baseline_cost_eur_cycle"))
    cycle_plan = _to_float_or_none(data.get("plan_cost_eur_cycle"))
    cycle_savings = _to_float_or_none(data.get("savings_eur_cycle"))

    tomorrow_base = _to_float_or_none(data.get("baseline_cost_eur_tomorrow"))
    tomorrow_plan = _to_float_or_none(data.get("plan_cost_eur_tomorrow"))
    tomorrow_savings = _to_float_or_none(data.get("savings_eur_tomorrow"))

    total_base = _to_float_or_none(data.get("baseline_cost_eur_total"))
    total_plan = _to_float_or_none(data.get("plan_cost_eur_total"))
    total_savings = _to_float_or_none(data.get("savings_eur_total"))

    display_scope = "tomorrow"
    if cycle_base is not None or cycle_plan is not None or cycle_savings is not None:
        display_scope = "cycle"
    elif total_base is not None or total_plan is not None or total_savings is not None:
        display_scope = "total"

    if display_scope == "cycle":
        base_cost = cycle_base
        plan_cost = cycle_plan
        reported_savings = cycle_savings
    elif display_scope == "total":
        base_cost = total_base
        plan_cost = total_plan
        reported_savings = total_savings
    else:
        base_cost = tomorrow_base
        plan_cost = tomorrow_plan
        reported_savings = tomorrow_savings

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

    hourly_cycle = _convert_hourly(data.get("hourly_savings_eur_cycle"))
    hourly_tomorrow = _convert_hourly(data.get("hourly_savings_eur_tomorrow"))
    hourly_labels_raw = data.get("hourly_savings_cycle_hour_labels")
    hourly_cycle_labels = (
        [str(x) for x in hourly_labels_raw]
        if isinstance(hourly_labels_raw, list) and len(hourly_labels_raw) == 24
        else None
    )

    bars_scope = "tomorrow"
    hourly: list[float] | None = hourly_tomorrow
    hourly_labels = [f"{h:02d}:00" for h in range(24)]
    if display_scope == "cycle" and hourly_cycle is not None:
        bars_scope = "cycle"
        hourly = hourly_cycle
        if hourly_cycle_labels is not None:
            hourly_labels = hourly_cycle_labels

    horizon_label = str(data.get("savings_horizon_label") or "").strip() or None
    detail_note = "⏱️ Bars: tomorrow (00–24)"
    if display_scope == "cycle" and bars_scope == "cycle":
        note = "Cycle savings shown (off-peak start → next off-peak start). Hourly bars align to cycle hours."
        detail_note = "⏱️ Bars: cycle (off-peak start → next off-peak start)"
    elif display_scope in {"cycle", "total"} and hourly is not None:
        note = "Cycle savings shown (off-peak start → next off-peak start). Hourly bars show tomorrow (00–24)."
    elif display_scope in {"cycle", "total"}:
        note = "Cycle savings shown (off-peak start → next off-peak start)."
    else:
        note = "Values shown are for tomorrow only."

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
        "detail_note": detail_note,
        "note": note,
    }
