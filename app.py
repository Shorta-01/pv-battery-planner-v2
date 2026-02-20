from __future__ import annotations

import datetime as dt
import html
import inspect
import json
import os
import time
import traceback
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import planner_core as core
from tariff_time import compute_offpeak_segments, make_summary_lines, parse_hhmm
from weather_ensemble import auto_select_models_for_location, should_use_satellite_nowcast_auto

PLOTLY_DARK = "plotly_dark"

INPUT_TOOLTIPS = {
    "soc_percent": "This is your battery level at 22:00. It matters because charging need is based on how full the battery already is. Example: 35 means the battery starts at 35%.",
    "yesterday_kwh": "This is your total home usage yesterday. It matters because the app uses it to estimate tomorrow's hourly load. Example: if yesterday was 18 kWh, tomorrow's hourly load profile scales to 18 kWh.",
    "buffer_percent": "This adds a safety margin to the target SOC. It matters when forecasts are uncertain. Example: 3% means the target cutoff SOC is increased by 3 percentage points.",
    "performance_ratio": "Overall PV real-world efficiency after losses. Recommended starting point: 0.82. Typical working range: 0.70–0.85. Example: 0.82 means you expect about 82% of ideal output.",
    "inverter_eff": "DC→AC inverter efficiency (only used in split loss mode). Recommended: 0.97. Typical range: 0.95–0.98 for modern inverters.",
    "pv_loss_model": "Choose how PV losses are applied before inverter modeling: split = performance ratio then inverter efficiency/model, combined = performance ratio only.",
    "iam_model": "IAM (Incidence Angle Modifier) models reflection losses when sunlight hits at steep angles (morning/evening, especially east/west). none: ignore angle losses. ashrae: apply AOI losses using IAM ASHRAE b.",
    "iam_ashrae_b": "ASHRAE IAM coefficient b (only used when IAM model = ashrae). Recommended: 0.05. Typical range: 0.03–0.08. Higher b reduces early/late power more.",
    "albedo": "Ground reflectance. Recommended default: 0.20 (grass). Asphalt ~0.10–0.15. Snow can be 0.60+. Only change if your ground conditions are unusual.",
    "albedo_enabled": "Enable only if you want to override albedo manually. Leave OFF for normal use. Turn ON for unusual ground reflectance (snow, very bright surfaces).",
    "inverter_ac_model": "How DC power becomes AC power. linear: AC = DC × inverter efficiency, then clip at the inverter AC limit (simple). pvwatts: part-load inverter behavior (more realistic at low power). Example: 2.0 kW DC with 0.97 → ~1.94 kW AC in linear mode before clipping.",
    "pv_calibration_factor": "Global PV tuning factor. Start at 1.00. Effective east = global × east(relative). Effective south = global × south(relative). Use only after comparing forecast vs actual.",
    "pv_calibration_factor_east": "East relative tuning multiplied by global. Start at 1.00. Example: global 0.95 and east 1.02 → effective east 0.969.",
    "pv_calibration_factor_south": "South relative tuning multiplied by global. Start at 1.00. Example: global 0.95 and south 0.98 → effective south 0.931.",
    "max_ac_user_cap": "The app computes a recommended AC charge power from required energy and off-peak window hours. This field is your safety cap: final used value is min(recommended, your cap, inverter/battery limits).",
}


INPUT_TOOLTIPS.update({
    "latitude": "Your home latitude. It matters for sun angle and PV timing. Example: 50.85 for Brussels.",
    "longitude": "Your home longitude. It matters for local solar time. Example: 4.35 for Brussels.",
    "timezone": "IANA timezone used for all hourly planning. It matters for tariff windows. Example: Europe/Brussels.",
    "tariff_from": "Start of cheap window in 24h HH:MM. It matters for charging hours. Example: 22:00.",
    "tariff_to": "End of cheap window in 24h HH:MM. It matters for charging hours. Example: 06:00.",
    "peak_price": "Grid import price during expensive hours. It affects charge strategy. Example: 0.34 €/kWh.",
    "offpeak_price": "Grid import price during cheap hours. It drives night charging. Example: 0.22 €/kWh.",
    "injection_price": "Price you get for export. It affects export value. Example: 0.05 €/kWh.",
    "panel_wp": "Rated power per panel. It scales PV forecast. Example: 420 Wp.",
    "array_panels": "Number of panels on this roof face. It scales energy for that face. Example: 8 panels.",
    "tilt": "Roof tilt angle from horizontal. It changes seasonal yield. Example: 35°.",
    "azimuth": "Panel compass direction (N=0, E=90, S=180). It shifts production timing. Example: 180° for south.",
    "inverter_ac_kw_limit": "Maximum AC output your inverter can deliver. It caps PV power. Example: 5.0 kW.",
    "battery_kwh": "Usable battery energy capacity. It sets storage size. Example: 10.0 kWh.",
    "min_soc": "Minimum SOC reserve to protect battery. It limits discharge depth. Example: 15%.",
    "cutoff_soc": "Target stop SOC for grid charging. It controls overnight charge level. Example: 75%.",
    "battery_max_charge_kw": "Maximum battery charging power. It caps charging speed. Example: 3.0 kW.",
    "battery_max_discharge_kw": "Maximum battery discharge power. It caps support to load. Example: 3.0 kW.",
    "max_ac_charge_kw_hard_limit": "Hard AC charging cap. It protects wiring/inverter. Example: 2.5 kW.",
    "forecast_mode": "Auto lets the system pick models and nowcast behavior. Expert lets you choose models and nowcast yourself. Example: use Auto for normal operation.",
    "sat_nowcast": "Adds satellite radiation for near-term (0–6h). It can improve short-term cloud timing. Example: ON during daytime, OFF for week-ahead.",
    "address_query": "Search text for location lookup. It helps fill coordinates/timezone. Example: Main Street 10, Brussels.",
})

def get_help(key: str, fallback: str = "") -> str:
    tip = INPUT_TOOLTIPS.get(key)
    if isinstance(tip, str) and tip.strip():
        return tip
    return fallback

METRIC_TOOLTIPS = {
    "Allowed AC charge power (kW)": "Final AC charge power used by the planner after all limits. It matters because this is the value to configure in FusionSolar. It never exceeds your safety cap, inverter limits, or battery limits.",
    "AC charge cutoff SOC (%)": "Battery SOC where FusionSolar should stop charging from the grid. Set this value as the 'AC charge cutoff SOC'.",
    "Forecast total PV (kWh)": "Estimated total PV energy produced tomorrow (after inverter AC limit).",
    "Forecast total load (kWh)": "Estimated total consumption for tomorrow. This is based on yesterday's total and a default hourly profile.",
    "Estimated grid import (expensive h)": "Estimated energy you may still buy from the grid during expensive tariff hours after using PV and the battery.",
    "Estimated export/curtailment (kWh)": "Estimated PV energy that cannot be used or stored and may be exported to the grid (or clipped/curtailed).",
}

CHART_TOOLTIPS = {
    "PV production vs Load (estimated) (hourly)": "This chart shows hourly energy from your PV and your estimated home load, plus battery SOC on a secondary axis. Load is estimated from yesterday total plus a profile and can differ from real usage (EV charging, heat pump cycles, weekends).",
    "Surplus vs Deficit (hourly)": "This chart compares PV surplus and load deficit each hour. It matters because surplus can charge the battery, while deficit means battery discharge or grid import is needed.",
    "Grid import/export + curtailment": "These bars show hourly grid import, grid export, and curtailed PV energy. Positive bars are import from grid. Negative bars are energy sent out or lost due to limits.",
}

TABLE_TOOLTIPS = {
    "Weather inputs used": "This table shows the weather data used for the forecast, hour by hour. It matters because PV results depend directly on these values.",
    "Hourly planning output": "This table combines hourly PV, load, battery SOC, and grid flows. It helps you inspect exactly what the planner expects each hour.",
    "History log": "By default this table shows the latest run per date. Enable \"Show all runs\" to view every run.",
}

WEATHER_MODEL_ORDER = [
    "knmi_harmonie_arome",
    "dwd_icon_d2",
    "ecmwf_ifs",
    "dwd_icon_eu",
    "meteofrance_seamless",
]

WEATHER_MODEL_DEFAULT = {"knmi_harmonie_arome", "dwd_icon_d2", "ecmwf_ifs"}
FORECAST_MODE_OPTIONS = {"Auto (System picks the best models)": "auto", "Auto (Recommended)": "auto", "Expert": "expert"}

WEATHER_MODEL_HOVERTEXT = {
    "knmi_harmonie_arome": "High-resolution KNMI regional model for Benelux. Strong for short-term local cloud and wind changes.",
    "dwd_icon_d2": "Very high-resolution DWD model (Germany region). Often captures fast cloud transitions that impact PV.",
    "ecmwf_ifs": "ECMWF global model. Very reliable for fronts and the overall weather pattern, good stable baseline.",
    "dwd_icon_eu": "European ICON model. Useful secondary view when the high-res model is noisy or inconsistent.",
    "meteofrance_seamless": "Météo-France seamless blend. Helpful extra perspective for Western Europe cloud patterns.",
}

BADGE_META = {
    "🏅": {"label": "CORE", "tip": "Core model. Recommended default for Belgium."},
    "📡": {"label": "HI-RES", "tip": "High-resolution (local). Best for short-term local cloud timing, which often improves PV ramps and hour-to-hour changes."},
    "🇪🇺": {"label": "EU", "tip": "Regional (Europe-scale). A good second opinion that is usually smoother and more stable than high-resolution models."},
    "🌐": {"label": "GLOBAL", "tip": "Global. Stable big-picture baseline for fronts and the overall weather pattern."},
    "☀": {"label": "SOLAR+", "tip": "Best PV inputs. This model provides the main solar irradiance fields directly, which usually improves PV accuracy."},
    "∑": {"label": "SOLAR∼", "tip": "Derived PV inputs. Some solar irradiance fields are missing, so we estimate them. PV still works, but accuracy can drop on difficult cloud days."},
    "⏱": {"label": "15m", "tip": "Uses 15-minute solar radiation (then aggregated to hourly). Can improve the PV curve shape when clouds change quickly."},
}

BADGE_ALIASES = {
    "⭐": "🏅",
    "🔎": "📡",
    "🗺": "🇪🇺",
    "🌍": "🌐",
    "🟩": "☀",
    "🧩": "∑",
    "☀️": "☀",
    "⏱️": "⏱",
}


UI_PROGRESS_BAR_HEIGHT_PX = 8

PV_RECO_PR = 0.82
PV_RECO_INVERTER_EFF = 0.97
PV_RECO_INVERTER_AC_MODEL = "pvwatts"
PV_RECO_IAM_MODEL = "ashrae"
PV_RECO_IAM_B = 0.05
PV_RECO_ALBEDO = 0.20
PV_RECO_CAL_GLOBAL = 1.00
PV_RECO_CAL_EAST = 1.00
PV_RECO_CAL_SOUTH = 1.00


def _esc(s: str) -> str:
    return html.escape(str(s or ""), quote=True)


def _safe_float(value: object, fallback: float) -> float | None:
    try:
        if value is None:
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def get_selected_weather_models(valid_model_ids: set[str]) -> list[str]:
    selected: list[str] = []
    for mid in WEATHER_MODEL_ORDER:
        if mid in valid_model_ids and bool(st.session_state.get(f"wm_{mid}", False)):
            selected.append(mid)
    return selected


def normalize_effective_cfg_to_payload(effective_cfg: dict, valid_model_ids: set[str]) -> dict:
    saved_selected = effective_cfg.get("weather_models_selected")
    if isinstance(saved_selected, list):
        selected_models = [mid for mid in saved_selected if isinstance(mid, str) and mid in valid_model_ids]
    else:
        selected_models = []
    if not selected_models:
        selected_models = sorted(list(WEATHER_MODEL_DEFAULT & valid_model_ids)) or sorted(list(valid_model_ids))

    forecast_mode_value = str(effective_cfg.get("forecast_mode", "auto")).strip().lower()
    forecast_mode_to_save = forecast_mode_value if forecast_mode_value in {"auto", "expert"} else "auto"

    location_cfg = effective_cfg.get("location", {}) if isinstance(effective_cfg, dict) else {}
    tariff_cfg = effective_cfg.get("tariff", {}) if isinstance(effective_cfg, dict) else {}
    pv_cfg = effective_cfg.get("pv", {}) if isinstance(effective_cfg, dict) else {}
    battery_cfg = effective_cfg.get("battery", {}) if isinstance(effective_cfg, dict) else {}
    weather_cfg = effective_cfg.get("weather", {}) if isinstance(effective_cfg, dict) else {}
    load_profile_cfg = effective_cfg.get("load_profile", {}) if isinstance(effective_cfg, dict) else {}

    return {
        "location": {
            "use_geocoding": False,
            "address_query": str(location_cfg.get("address_query", "")),
            "address_structured": {
                "street": str((location_cfg.get("address_structured", {}) or {}).get("street", "")),
                "house_number": str((location_cfg.get("address_structured", {}) or {}).get("house_number", "")),
                "postal_code": str((location_cfg.get("address_structured", {}) or {}).get("postal_code", "")),
                "city": str((location_cfg.get("address_structured", {}) or {}).get("city", "")),
                "country": str((location_cfg.get("address_structured", {}) or {}).get("country", "")),
            },
            "latitude": float(location_cfg.get("latitude", core.LATITUDE)),
            "longitude": float(location_cfg.get("longitude", core.LONGITUDE)),
            "timezone": str(location_cfg.get("timezone", core.TIMEZONE)),
        },
        "tariff": {
            "peak_grid_price_eur_per_kwh": float(tariff_cfg.get("peak_grid_price_eur_per_kwh", core.DEFAULT_CONFIG["tariff"]["peak_grid_price_eur_per_kwh"])),
            "offpeak_grid_price_eur_per_kwh": float(tariff_cfg.get("offpeak_grid_price_eur_per_kwh", core.DEFAULT_CONFIG["tariff"]["offpeak_grid_price_eur_per_kwh"])),
            "injection_grid_price_eur_per_kwh": float(tariff_cfg.get("injection_grid_price_eur_per_kwh", core.DEFAULT_CONFIG["tariff"]["injection_grid_price_eur_per_kwh"])),
            "offpeak_windows_by_dow": tariff_cfg.get("offpeak_windows_by_dow", core.DEFAULT_CONFIG["tariff"]["offpeak_windows_by_dow"]),
        },
        "pv": dict(pv_cfg),
        "battery": dict(battery_cfg),
        "load_profile": {
            "load_profile_24h": [float(v) for v in load_profile_cfg.get("load_profile_24h", core.DEFAULT_CONFIG["load_profile"]["load_profile_24h"])],
        },
        "weather": {
            **weather_cfg,
            "use_satellite_nowcast_0_6h": bool(weather_cfg.get("use_satellite_nowcast_0_6h", False)),
        },
        "weather_models_selected": selected_models,
        "forecast_mode": forecast_mode_to_save,
    }


def build_settings_payload(effective_cfg: dict, valid_model_ids: set[str]) -> tuple[dict | None, str | None]:
    ui = st.session_state.get("_cfg_ui_snapshot")
    if not isinstance(ui, dict):
        return normalize_effective_cfg_to_payload(effective_cfg, valid_model_ids), None

    for day_idx, (from_value, to_value) in enumerate(ui.get("tariff_inputs", [])):
        try:
            start_min = parse_hhmm(from_value, allow_24_end=False)
            end_min = parse_hhmm(to_value, allow_24_end=True)
            compute_offpeak_segments(start_min, end_min)
        except ValueError as exc:
            day_names = ui.get("day_names", [])
            day_label = day_names[day_idx] if day_idx < len(day_names) else f"Day {day_idx + 1}"
            return None, f"Tariff settings error for {day_label}: {exc}"

    existing_wm_keys = any(f"wm_{mid}" in st.session_state for mid in valid_model_ids)
    if existing_wm_keys:
        selected_to_save = get_selected_weather_models(valid_model_ids)
    else:
        saved_selected = effective_cfg.get("weather_models_selected")
        selected_to_save = [mid for mid in saved_selected if isinstance(mid, str) and mid in valid_model_ids] if isinstance(saved_selected, list) else []
    if not selected_to_save:
        selected_to_save = sorted(list(WEATHER_MODEL_DEFAULT & valid_model_ids)) or sorted(list(valid_model_ids))

    forecast_mode_value = str(st.session_state.get("forecast_mode_select", "Auto (System picks the best models)"))
    forecast_mode_to_save = FORECAST_MODE_OPTIONS.get(forecast_mode_value, "auto")
    user_sat_setting = bool(st.session_state.get("use_sat_nowcast_expert", ui.get("saved_sat", False)))

    new_cfg = {
        "location": {
            "use_geocoding": False,
            "address_query": str(st.session_state.get("loc_address_query_display", "")),
            "address_structured": {
                "street": str(st.session_state.get("loc_street", "")),
                "house_number": str(st.session_state.get("loc_house_number", "")),
                "postal_code": str(st.session_state.get("loc_postal_code", "")),
                "city": str(st.session_state.get("loc_city", "")),
                "country": str(st.session_state.get("loc_country", "")),
            },
            "latitude": float(ui["cfg_latitude"]),
            "longitude": float(ui["cfg_longitude"]),
            "timezone": str(st.session_state.get("loc_timezone", core.TIMEZONE)),
        },
        "tariff": {
            "peak_grid_price_eur_per_kwh": float(ui["cfg_peak_price_input"]),
            "offpeak_grid_price_eur_per_kwh": float(ui["cfg_offpeak_price_input"]),
            "injection_grid_price_eur_per_kwh": float(ui["cfg_injection_price_input"]),
            "offpeak_windows_by_dow": [
                [[from_value, to_value], *[[w_start, w_end] for (w_start, w_end) in ui["tariff_by_day"].get(day_idx, [])[1:]]] if ui["tariff_by_day"].get(day_idx) else [[from_value, to_value]]
                for day_idx, (from_value, to_value) in enumerate(ui["tariff_inputs"])
            ],
        },
        "pv": {
            "panel_wp": int(ui["cfg_panel_wp"]),
            "array_south_panels": int(ui["cfg_array_south_panels"]),
            "array_east_panels": int(ui["cfg_array_east_panels"]),
            "tilt_east_deg": float(ui["cfg_tilt_east_deg"]),
            "tilt_south_deg": float(ui["cfg_tilt_south_deg"]),
            "azimuth_east_deg": float(ui["cfg_azimuth_east_deg"]),
            "azimuth_south_deg": float(ui["cfg_azimuth_south_deg"]),
            "performance_ratio": float(ui["cfg_performance_ratio"]),
            "inverter_eff": float(ui["cfg_inverter_eff"]),
            "pv_loss_model": str(ui["cfg_pv_loss_model"]),
            "iam_model": str(ui["cfg_iam_model"]),
            "iam_ashrae_b": float(ui["cfg_iam_ashrae_b"]),
            "albedo": (float(ui["cfg_albedo"]) if ui["cfg_albedo_enabled"] else None),
            "inverter_ac_model": str(ui["cfg_inverter_ac_model"]),
            "pv_calibration_factor": float(ui["cfg_pv_calibration_factor"]),
            "pv_calibration_factor_east": float(ui["cfg_pv_calibration_factor_east"]),
            "pv_calibration_factor_south": float(ui["cfg_pv_calibration_factor_south"]),
            "inverter_ac_kw_limit": float(ui["cfg_inverter_ac_kw_limit"]),
        },
        "battery": {
            "battery_kwh": float(ui["cfg_battery_kwh"]),
            "min_soc_percent": float(ui["cfg_min_soc_percent"]),
            "max_cutoff_soc_percent": float(ui["cfg_max_cutoff_soc_percent"]),
            "battery_max_charge_kw": float(ui["cfg_battery_max_charge_kw"]),
            "battery_max_discharge_kw": float(ui["cfg_battery_max_discharge_kw"]),
            "max_ac_charge_kw_hard_limit": float(ui["cfg_max_ac_charge_kw_hard_limit"]),
        },
        "load_profile": {
            "load_profile_24h": [float(v) for v in ui["cfg_load_profile"]],
        },
        "weather": {
            **((effective_cfg.get("weather", {}) if isinstance(effective_cfg, dict) else {})),
            "use_satellite_nowcast_0_6h": user_sat_setting,
        },
        "weather_models_selected": selected_to_save,
        "forecast_mode": forecast_mode_to_save,
    }
    return new_cfg, None


def validate_sidebar_readiness(ui: dict, *, yesterday_kwh: float, forecast_mode: str, selected_models: list[str]) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {
        "Inputs": [],
        "Location": [],
        "Tariffs": [],
        "PV": [],
        "Battery": [],
        "Weather": [],
    }

    if yesterday_kwh < 2.0 or yesterday_kwh > 60.0:
        issues["Inputs"].append("Yesterday usage must be between 2.0 and 60.0 kWh.")

    lat = float(ui["cfg_latitude"])
    lon = float(ui["cfg_longitude"])
    tz_name = str(st.session_state.get("loc_timezone", core.TIMEZONE)).strip()
    if not (-90.0 <= lat <= 90.0):
        issues["Location"].append("Latitude must be between -90 and 90.")
    if not (-180.0 <= lon <= 180.0):
        issues["Location"].append("Longitude must be between -180 and 180.")
    try:
        ZoneInfo(tz_name)
    except Exception:
        issues["Location"].append("Timezone must be a valid IANA name (example: Europe/Brussels).")

    day_names = ui.get("day_names", [])
    for day_idx, (from_value, to_value) in enumerate(ui.get("tariff_inputs", [])):
        try:
            start_min = parse_hhmm(from_value, allow_24_end=False)
            end_min = parse_hhmm(to_value, allow_24_end=True)
            compute_offpeak_segments(start_min, end_min)
        except ValueError as exc:
            day = day_names[day_idx] if day_idx < len(day_names) else f"Day {day_idx + 1}"
            issues["Tariffs"].append(f"{day}: {exc}")

    if float(ui["cfg_panel_wp"]) <= 0:
        issues["PV"].append("Panel power must be > 0.")
    if int(ui["cfg_array_south_panels"]) + int(ui["cfg_array_east_panels"]) <= 0:
        issues["PV"].append("Total east + south panels must be > 0.")
    if float(ui["cfg_inverter_ac_kw_limit"]) <= 0:
        issues["PV"].append("Inverter AC limit must be > 0.")

    if float(ui["cfg_battery_kwh"]) <= 0:
        issues["Battery"].append("Battery capacity must be > 0.")
    min_soc = float(ui["cfg_min_soc_percent"])
    cutoff_soc = float(ui["cfg_max_cutoff_soc_percent"])
    if not (0 <= min_soc <= 100):
        issues["Battery"].append("Min SOC must be between 0 and 100.")
    if not (0 <= cutoff_soc <= 100):
        issues["Battery"].append("Cutoff SOC must be between 0 and 100.")
    if cutoff_soc < min_soc:
        issues["Battery"].append("Cutoff SOC must be greater than or equal to Min SOC.")
    if float(ui["cfg_battery_max_charge_kw"]) <= 0:
        issues["Battery"].append("Max charge power must be > 0.")
    if float(ui["cfg_battery_max_discharge_kw"]) <= 0:
        issues["Battery"].append("Max discharge power must be > 0.")
    if float(ui["cfg_max_ac_charge_kw_hard_limit"]) <= 0:
        issues["Battery"].append("AC charge hard limit must be > 0.")

    if forecast_mode == "expert" and not selected_models:
        issues["Weather"].append("Select at least one weather model in Expert mode.")

    return issues


def save_settings_payload(new_cfg: dict, *, rerun: bool = True) -> bool:
    try:
        updated = api_put(
            "/v1/settings",
            {
                "config": new_cfg,
                "nightly_run_time": backend_settings.get("nightly_run_time", "22:00"),
                "timezone": str(new_cfg["location"].get("timezone", backend_settings.get("timezone", "Europe/Brussels"))),
                "max_ac_charge_power_kw_default": backend_settings.get("max_ac_charge_power_kw_default", 5.0),
            },
        )
        st.cache_data.clear()
        st.session_state["_pending_location_state"] = updated["config"]["location"]
        st.session_state["_settings_flash"] = "Saved settings to backend"
        if rerun:
            st.rerun()
    except Exception as exc:
        st.error(f"Could not save settings: {exc}")
        return False
    return True

LOCAL_STATE_DIR = Path("local_state")
API_BASE_URL = os.getenv("PVBP_BACKEND_URL", "http://127.0.0.1:8787")
API_TOKEN_FILE = LOCAL_STATE_DIR / "api_token.txt"
APP_DEBUG = os.getenv("DEBUG", "").strip() in ("1", "true", "True", "yes", "YES")

st.session_state.setdefault("history_all_runs", False)
st.session_state.setdefault("history_show_run_at", False)
st.session_state.setdefault("history_debug_columns", False)

def apply_pending_location_state() -> None:
    pending = st.session_state.pop("_pending_location_state", None)
    if not isinstance(pending, dict):
        return
    structured = pending.get("address_structured", {}) if isinstance(pending.get("address_structured"), dict) else {}
    st.session_state["loc_address_query_display"] = str(pending.get("address_query", ""))
    st.session_state["loc_latitude"] = _safe_float(pending.get("latitude"), core.LATITUDE)
    st.session_state["loc_longitude"] = _safe_float(pending.get("longitude"), core.LONGITUDE)
    st.session_state["loc_timezone"] = str(pending.get("timezone", core.TIMEZONE))
    st.session_state["loc_street"] = str(structured.get("street", ""))
    st.session_state["loc_house_number"] = str(structured.get("house_number", ""))
    st.session_state["loc_postal_code"] = str(structured.get("postal_code", ""))
    st.session_state["loc_city"] = str(structured.get("city", ""))
    st.session_state["loc_country"] = str(structured.get("country", ""))


def apply_location_lookup_result(cfg: dict) -> None:
    if not st.session_state.get("loc_apply_lookup"):
        return
    res = st.session_state.get("loc_lookup_result")
    if not isinstance(res, dict):
        st.session_state["loc_apply_lookup"] = False
        return

    cfg.setdefault("location", {})
    cfg["location"]["address_query"] = str(res.get("address_query", ""))
    cfg["location"]["latitude"] = _safe_float(res.get("latitude"), core.LATITUDE)
    cfg["location"]["longitude"] = _safe_float(res.get("longitude"), core.LONGITUDE)
    cfg["location"]["timezone"] = str(res.get("timezone", core.TIMEZONE))
    cfg["location"]["address_structured"] = res.get("address_structured", {})

    st.session_state["loc_address_query_display"] = str(res.get("address_query", ""))
    st.session_state["loc_latitude"] = _safe_float(res.get("latitude"), core.LATITUDE)
    st.session_state["loc_longitude"] = _safe_float(res.get("longitude"), core.LONGITUDE)
    st.session_state["loc_timezone"] = str(res.get("timezone", core.TIMEZONE))

    structured = res.get("address_structured", {}) if isinstance(res.get("address_structured"), dict) else {}
    fallback_query = str(res.get("address_query", ""))
    st.session_state["loc_street"] = str(structured.get("street", "")) or fallback_query
    st.session_state["loc_house_number"] = str(structured.get("house_number", ""))
    st.session_state["loc_postal_code"] = str(structured.get("postal_code", ""))
    st.session_state["loc_city"] = str(structured.get("city", ""))
    st.session_state["loc_country"] = str(structured.get("country", ""))

    st.session_state["loc_apply_lookup"] = False


def submit_structured_lookup() -> None:
    try:
        result = core.resolve_location_from_structured_address(
            st.session_state.get("loc_street", ""),
            st.session_state.get("loc_house_number", ""),
            st.session_state.get("loc_postal_code", ""),
            st.session_state.get("loc_city", ""),
            st.session_state.get("loc_country", ""),
        )
    except Exception as exc:
        st.session_state["_geo_error"] = str(exc)
        st.session_state.pop("_geo_success", None)
        return

    st.session_state["loc_lookup_result"] = result
    st.session_state["loc_apply_lookup"] = True
    st.session_state["loc_lookup_open"] = False
    st.session_state["_geo_success"] = (
        f"Resolved {result['address_query']}: {result['latitude']:.5f}, {result['longitude']:.5f}"
    )
    st.session_state.pop("_geo_error", None)


def close_lookup() -> None:
    st.session_state["loc_lookup_open"] = False


def open_lookup(loc_cfg: dict) -> None:
    pending_structured = {
        "street": "",
        "house_number": "",
        "postal_code": "",
        "city": "",
        "country": "",
    }
    st.session_state["_pending_location_state"] = {
        "address_query": str(st.session_state.get("loc_address_query_display", "")),
        "latitude": _safe_float(st.session_state.get("loc_latitude"), _safe_float(loc_cfg.get("latitude"), core.LATITUDE)),
        "longitude": _safe_float(st.session_state.get("loc_longitude"), _safe_float(loc_cfg.get("longitude"), core.LONGITUDE)),
        "timezone": str(st.session_state.get("loc_timezone", loc_cfg.get("timezone", core.TIMEZONE))),
        "address_structured": pending_structured,
    }
    st.session_state["loc_street"] = ""
    st.session_state["loc_house_number"] = ""
    st.session_state["loc_postal_code"] = ""
    st.session_state["loc_city"] = ""
    st.session_state["loc_country"] = ""
    st.session_state.pop("_geo_error", None)
    st.session_state.pop("_geo_success", None)
    st.session_state["loc_lookup_open"] = True


def _render_lookup_form_contents() -> None:
    st.text_input("Street", key="loc_street")
    st.text_input("House Number", key="loc_house_number")
    st.text_input("ZIP", key="loc_postal_code")
    st.text_input("City", key="loc_city")
    st.text_input("Country", key="loc_country")


if hasattr(st, "dialog"):
    @st.dialog("Lookup location")
    def lookup_location_dialog() -> None:
        with st.form("lookup_location_form"):
            _render_lookup_form_contents()
            if st.session_state.get("_geo_error"):
                st.error(st.session_state["_geo_error"])
            if st.session_state.get("_geo_success"):
                st.success(st.session_state["_geo_success"])
            cancel_col, ok_col = st.columns(2)
            with cancel_col:
                cancel = st.form_submit_button("Cancel")
            with ok_col:
                ok = st.form_submit_button("OK", type="primary")
            if cancel:
                close_lookup()
                st.session_state.pop("_geo_error", None)
                st.session_state.pop("_geo_success", None)
                st.rerun()
            if ok:
                submit_structured_lookup()
                st.rerun()
else:
    def lookup_location_dialog() -> None:
        with st.container(border=True):
            st.markdown("#### Lookup location")
            with st.form("lookup_location_form_fallback"):
                _render_lookup_form_contents()
                if st.session_state.get("_geo_error"):
                    st.error(st.session_state["_geo_error"])
                if st.session_state.get("_geo_success"):
                    st.success(st.session_state["_geo_success"])
                cancel_col, ok_col = st.columns(2)
                with cancel_col:
                    cancel = st.form_submit_button("Cancel")
                with ok_col:
                    ok = st.form_submit_button("OK", type="primary")
                if cancel:
                    close_lookup()
                    st.session_state.pop("_geo_error", None)
                    st.session_state.pop("_geo_success", None)
                    st.rerun()
                if ok:
                    submit_structured_lookup()
                    st.rerun()


def format_hour_from_index(index: pd.Index, fmt: str) -> pd.Series:
    dt_index = pd.to_datetime(index, errors="coerce")
    if isinstance(dt_index, pd.DatetimeIndex):
        return pd.Series(dt_index.strftime(fmt), index=index)
    return pd.Series(index.astype(str), index=index)


def inject_tooltip_css() -> None:
    st.markdown(
        """
        <style>
        .info-tooltip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-left: 0.35rem;
            width: 1rem;
            height: 1rem;
            border-radius: 50%;
            border: 1px solid rgba(255,255,255,0.25);
            color: #e8eaed;
            font-size: 0.75rem;
            cursor: help;
            line-height: 1rem;
            vertical-align: middle;
        }
        .tooltip-heading {
            margin-top: 0.7rem;
            margin-bottom: 0.2rem;
            color: #fafafa;
            font-size: 1.05rem;
            font-weight: 600;
        }
        .pvbp-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            border-radius: 0;
            border: none;
            background: transparent;
            box-shadow: none;
            outline: none;
            font-size: 12px;
            line-height: 1;
            margin-left: 6px;
            cursor: help;
            white-space: nowrap;
        }
        .pvbp-badge-icon {
            font-size: 14px;
            line-height: 1;
        }
        .stButton>button {
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _normalize_badge_icon(icon: str) -> str:
    return BADGE_ALIASES.get(icon, icon)


def badge_chip(icon: str, tip: str) -> str:
    title_attr = f' title="{html.escape(tip)}"' if tip else ""
    return (
        f'<span class="pvbp-badge"{title_attr}>'
        f'<span class="pvbp-badge-icon">{html.escape(icon)}</span>'
        "</span>"
    )


def weather_model_option_help(model: dict) -> str:
    badges_raw = list(model.get("badges", []) or [])
    normalized_badges: list[str] = []
    for badge in badges_raw:
        if not str(badge).strip():
            continue
        normalized = _normalize_badge_icon(str(badge))
        if normalized in BADGE_META:
            normalized_badges.append(normalized)

    unique_badges = list(dict.fromkeys(normalized_badges))
    badge_summary = ", ".join(
        f"{icon} {BADGE_META[icon]['label'].lower()}" for icon in unique_badges
    )
    notes = str(model.get("notes") or model.get("capability", {}).get("notes") or "")
    if notes and badge_summary:
        return f"{notes}\n\nLegend: {badge_summary}."
    if notes:
        return notes
    if badge_summary:
        return f"Legend: {badge_summary}."
    return ""


def last_run_status_badge(model_id: str) -> tuple[str | None, str]:
    dbg = st.session_state.get("last_weather_ensemble_debug") or {}
    if not isinstance(dbg, dict) or not dbg:
        return (None, "")

    failed = set(dbg.get("failed_models") or [])
    reasons = dbg.get("failed_model_reasons") or {}
    missing_by = dbg.get("missing_vars_by_model") or {}
    derived_by = dbg.get("derived_irradiance_by_model") or {}

    if model_id in failed:
        r = reasons.get(model_id) or {}
        category = str(r.get("category") or "unknown").strip() if isinstance(r, dict) else "unknown"
        status = r.get("status") if isinstance(r, dict) else None
        message = str(r.get("message") or "Unknown error").strip() if isinstance(r, dict) else str(r).strip() or "Unknown error"
        parts = ["Failed in last run. This model was not used."]
        parts.append(f"Reason: {message}")
        parts.append(f"Category: {category}")
        if status is not None:
            parts.append(f"HTTP status: {status}")
        return ("❌", " ".join(parts))

    missing = list(missing_by.get(model_id) or [])
    derived = bool(derived_by.get(model_id))

    if missing or derived:
        parts = ["Worked in last run, but PV inputs were incomplete or estimated."]
        if missing:
            parts.append("Missing: " + ", ".join(missing) + ".")
        if derived:
            parts.append("Solar irradiance components were derived/approximated.")
        return ("⚠", " ".join(parts))

    return (None, "")


def render_weather_models(
    weather_models_catalog: list[dict],
    default_selected: set[str],
    *,
    widget_key_prefix: str = "wm",
    disabled: bool = False,
    used_models: set[str] | None = None,
    auto_locked_models: set[str] | None = None,
    show_capability_badges: bool = True,
) -> list[str]:
    model_options = {m.get("id"): m for m in weather_models_catalog if isinstance(m.get("id"), str)}
    selected_models: list[str] = []

    st.markdown(
        "<style>.wm-name{cursor:help}.wm-badges{display:flex;align-items:center;justify-content:flex-end;flex-wrap:wrap;row-gap:4px}.wm-lock{font-size:1rem;opacity:0.85;display:flex;align-items:center;justify-content:center;height:100%;line-height:1}</style>",
        unsafe_allow_html=True,
    )

    for model_id in WEATHER_MODEL_ORDER:
        model = model_options.get(model_id)
        if not model:
            continue

        cols = st.columns([0.35, 3.2, 1.3], vertical_alignment="center")

        with cols[0]:
            if disabled:
                lock_icon = "🔒" if (auto_locked_models and model_id in auto_locked_models) else ""
                st.markdown(f"<div class='wm-lock'>{_esc(lock_icon)}</div>", unsafe_allow_html=True)
                checked = bool(auto_locked_models and model_id in auto_locked_models)
            else:
                checked = st.checkbox(
                    "enabled",
                    value=(model_id in default_selected),
                    key=f"{widget_key_prefix}_{model_id}",
                    label_visibility="collapsed",
                    disabled=False,
                )

        with cols[1]:
            label = str(model.get("label") or model_id)
            tip = WEATHER_MODEL_HOVERTEXT.get(model_id, "")
            st.markdown(
                f"<span class='wm-name' title='{_esc(tip)}'><b>{_esc(label)}</b></span>",
                unsafe_allow_html=True,
            )

        with cols[2]:
            static_badges = list(model.get("badges") or [])
            status_icon, status_tip = last_run_status_badge(model_id)

            badge_html: list[str] = []
            if status_icon:
                badge_html.append(
                    f"<span class='pvbp-badge' title='{_esc(status_tip)}'><span class='pvbp-badge-icon'>{_esc(status_icon)}</span></span>"
                )

            if used_models and model_id in used_models and status_icon != "❌":
                badge_html.append(badge_chip("✅", "Used in the last run."))

            if show_capability_badges:
                for badge in static_badges:
                    icon = _normalize_badge_icon(str(badge))
                    meta = BADGE_META.get(icon)
                    if not meta:
                        continue
                    badge_html.append(badge_chip(icon=icon, tip=str(meta.get("tip") or "")))

            st.markdown(f"<div class='wm-badges'>{''.join(badge_html)}</div>", unsafe_allow_html=True)

        if checked:
            selected_models.append(model_id)

    if (not selected_models) and (not disabled):
        st.error("Select at least one weather model.")

    debug_ui = bool(os.getenv("APP_DEBUG")) and st.session_state.get("history_mode", "Simple") == "Debug"
    if debug_ui:
        dbg = st.session_state.get("last_weather_ensemble_debug") or {}
        with st.expander("Advanced: last run model debug", expanded=False):
            st.caption("Raw per-model debug from the last Run forecast. Copy/paste this into Codex when reporting issues.")
            if not dbg:
                st.info("No debug data yet. Click Run forecast once to populate this.")
            else:
                dbg_json = json.dumps(dbg, indent=2, ensure_ascii=False)
                st.text_area(
                    "Weather ensemble debug JSON (copy/paste)",
                    value=dbg_json,
                    height=280,
                    key=f"{widget_key_prefix}_weather_ensemble_debug_json_text_area",
                )
                st.download_button(
                    "Download debug JSON",
                    data=dbg_json,
                    file_name="weather_ensemble_debug.json",
                    mime="application/json",
                    key=f"{widget_key_prefix}_weather_ensemble_debug_json_download_button",
                )

    return selected_models


def tooltip_heading(label: str, help_text: str) -> None:
    safe_help = help_text.replace('"', "&quot;")
    st.markdown(
        f"<div class='tooltip-heading'>{label}<span class='info-tooltip' title=\"{safe_help}\">ⓘ</span></div>",
        unsafe_allow_html=True,
    )


def build_column_config(df: pd.DataFrame, candidates: dict) -> dict:
    if df is None or df.empty:
        return {}
    return {k: v for k, v in candidates.items() if k in df.columns}


def make_column_config(df: pd.DataFrame, units_and_help_map: dict[str, dict[str, str]]) -> dict:
    if df is None or df.empty:
        return {}
    config: dict[str, object] = {}
    for col in df.columns:
        meta = units_and_help_map.get(col)
        if not isinstance(meta, dict):
            continue
        label = str(meta.get("label") or col)
        help_text = str(meta.get("help") or "")
        fmt = str(meta.get("format") or "")
        if not fmt:
            if col.endswith("_pct"):
                fmt = "%.1f"
            elif col.endswith("_kwh") or col.endswith("_kw"):
                fmt = "%.2f"
            else:
                fmt = "%.2f"
        config[col] = st.column_config.NumberColumn(label=label, format=fmt, help=help_text)
    return config


@st.cache_data(show_spinner=False)
def compute_residual_kwh(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    for col in [
        "pv_total_kwh",
        "grid_import_kwh",
        "batt_discharge_kwh",
        "load_kwh",
        "batt_charge_kwh",
        "grid_export_kwh",
        "pv_curtailed_kwh",
    ]:
        if col not in out.columns:
            out[col] = 0.0
    lhs = (
        pd.to_numeric(out["pv_total_kwh"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["grid_import_kwh"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["batt_discharge_kwh"], errors="coerce").fillna(0.0)
    )
    rhs = (
        pd.to_numeric(out["load_kwh"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["batt_charge_kwh"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["grid_export_kwh"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["pv_curtailed_kwh"], errors="coerce").fillna(0.0)
    )
    out["residual_kwh"] = (lhs - rhs).round(3)
    return out


@st.cache_data(show_spinner=False)
def get_preset_columns(columns: tuple[str, ...], preset: str, table_kind: str) -> list[str]:
    available = list(columns)
    if table_kind == "weather":
        core_cols = ["hour", "temperature_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation", "ghi"]
        pv_plus = [
            "dni",
            "dhi",
            "direct_normal_irradiance",
            "diffuse_radiation",
            "shortwave_radiation_min",
            "shortwave_radiation_max",
        ]
        wanted = core_cols if preset == "Core" else (core_cols + pv_plus if preset == "PV-relevant" else available)
    else:
        core_cols = [
            "hour",
            "pv_total_kwh",
            "load_kwh",
            "soc_end_pct",
            "grid_import_kwh",
            "grid_export_kwh",
            "batt_charge_kwh",
            "batt_discharge_kwh",
            "charge_kw",
            "cutoff_soc_pct",
        ]
        bal_plus = [
            "pv_clipped_kwh",
            "pv_ac_limited_kwh",
            "pv_curtailed_kwh",
            "pv_dc_available_kwh",
            "pv_surplus_kwh",
            "pv_deficit_kwh",
            "residual_kwh",
        ]
        wanted = core_cols if preset == "Core" else (core_cols + bal_plus if preset == "Energy balance" else available)
    selected = [c for c in wanted if c in available]
    if table_kind == "weather" and "shortwave_radiation" not in selected and "ghi" in available and "ghi" not in selected:
        selected.append("ghi")
    return selected or available


@st.cache_data(show_spinner=False)


def weather_code_to_icon(weather_code: int | float | str | None) -> str:
    """
    Map Open-Meteo/WMO weather codes to clear, modern emojis.
    Accepts int codes or string labels; returns an emoji.
    """
    if weather_code is None:
        return "🌥️"

    # Support string labels (defensive)
    if isinstance(weather_code, str):
        raw = weather_code.strip()
        if raw.isdigit() or (raw.startswith('-') and raw[1:].isdigit()):
            try:
                return weather_code_to_icon(int(raw))
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
            "snow": "❄️",
            "snowfall": "❄️",
            "snow_showers": "🌨️",
            "sleet": "🌨️",
            "thunderstorm": "⛈️",
        }
        return label_map.get(key, "🌥️")

    # Numeric WMO mapping
    try:
        code = int(weather_code)
    except Exception:
        return "🌥️"

    if code == 0:
        return "☀️"      # Clear sky
    if code == 1:
        return "🌤️"     # Mainly clear
    if code == 2:
        return "⛅"      # Partly cloudy
    if code == 3:
        return "☁️"      # Overcast

    if code in (45, 48):
        return "🌫️"     # Fog / depositing rime fog

    if 51 <= code <= 57:
        return "🌦️"     # Drizzle / freezing drizzle (light→dense)

    if 61 <= code <= 67:
        return "🌧️"     # Rain / freezing rain (light→heavy)

    if 71 <= code <= 77:
        return "❄️"     # Snow fall / snow grains

    if 80 <= code <= 82:
        return "🌦️"     # Rain showers (slight/moderate/violent)

    if code in (85, 86):
        return "🌨️"     # Snow showers

    if code in (95, 96, 99):
        return "⛈️"     # Thunderstorm (slight/heavy w hail)

    return "🌥️"


def weather_code_to_label(weather_code):
    if weather_code is None:
        return "Unknown"
    try:
        code = int(weather_code)
    except Exception:
        return str(weather_code)

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


def render_pv_week_ahead_widget(items: list[dict]) -> None:
    st.markdown("### PV Week Ahead")
    st.caption(
        "Shows the 6 days after tomorrow. Week-ahead PV is less certain after day 3–4. "
        "Short-range models cover the first days; long-range models drive later days."
    )

    cols = st.columns(6, gap="small")
    for idx, col in enumerate(cols):
        item = items[idx] if idx < len(items) and isinstance(items[idx], dict) else {}
        date_raw = item.get("date")
        label = "—"
        date_label = ""
        try:
            d = dt.date.fromisoformat(str(date_raw))
            label = d.strftime("%a")
            date_label = d.strftime("%d %b")
        except Exception:
            date_label = str(date_raw or "")

        pv_candidates = pd.to_numeric(pd.Series([item.get("p50_kwh"), item.get("pv_p50_kwh")]), errors="coerce").dropna()
        pv_p50 = float(pv_candidates.iloc[0]) if not pv_candidates.empty else None

        pv_p10_raw = pd.to_numeric(pd.Series([item.get("p10_kwh"), item.get("pv_p10_kwh")]), errors="coerce").dropna()
        pv_p90_raw = pd.to_numeric(pd.Series([item.get("p90_kwh"), item.get("pv_p90_kwh")]), errors="coerce").dropna()
        pv_range = ""
        if pv_p50 is not None and not pv_p10_raw.empty and not pv_p90_raw.empty:
            pv_range = f"{float(pv_p10_raw.iloc[0]):.1f}–{float(pv_p90_raw.iloc[0]):.1f} kWh"

        code_value = item.get("weather_code")
        source_label = item.get("weather_code_source_model_label")
        source_days = item.get("weather_code_source_max_days")
        best_of_day = item.get("weather_best_of_day")

        label_txt = weather_code_to_label(code_value)
        icon = weather_code_to_icon(code_value) or "❔"

        icon_tooltip = (
            f"WMO: {code_value} ({label_txt})"
            + (f" | Source: {source_label}" if source_label else " | Source: n/a")
            + (f" ({source_days}d)" if source_days else "")
            + (" | Best-of-day (08–18)" if best_of_day else "")
        )
        icon_tooltip = html.escape(icon_tooltip, quote=True)

        with col:
            st.markdown(
                (
                    "<div style='border:1px solid rgba(255,255,255,0.12);border-radius:14px;padding:0.5rem;"
                    "background:linear-gradient(140deg, rgba(43,48,58,0.9), rgba(20,24,31,0.85));text-align:center;'>"
                    f"<div style='font-size:0.72rem;opacity:0.8;'>{label}</div>"
                    f"<div style='font-size:0.7rem;opacity:0.75;margin-top:0.05rem;'>{date_label}</div>"
                    f"<div title=\"{icon_tooltip}\" style='font-size:1.2rem;margin-top:0.25rem;cursor:help;'>{icon}</div>"
                    f"<div style='font-size:1rem;font-weight:700;margin-top:0.2rem;'>{(f'{pv_p50:.1f} kWh' if pv_p50 is not None else '—')}</div>"
                    f"<div style='font-size:0.68rem;opacity:0.75;margin-top:0.12rem;min-height:1.1em;'>{pv_range}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def resolve_forecast_summary_pv_kwh(
    pv_quality_dict: dict | None,
    pv_week_ahead: list[dict] | list[float] | None,
    pv_df: pd.DataFrame,
    result: dict | None,
    metrics: dict | None,
    weather_ensemble: dict | None,
) -> float | None:
    def _pick_float_value(*candidate_values: object) -> float | None:
        for candidate in candidate_values:
            value = pd.to_numeric(pd.Series([candidate]), errors="coerce").iloc[0]
            if not pd.isna(value):
                return float(value)
        return None

    week_day1 = None
    if isinstance(pv_week_ahead, list) and pv_week_ahead:
        first = pv_week_ahead[0]
        if isinstance(first, dict):
            week_day1 = first.get("p50_kwh") or first.get("pv_p50_kwh")
        else:
            week_day1 = first

    pv_total_kwh = _pick_float_value(
        result.get("pv_totals_kwh", {}).get("p50") if isinstance(result, dict) and isinstance(result.get("pv_totals_kwh"), dict) else None,
        pv_quality_dict.get("pv_total_kwh") if isinstance(pv_quality_dict, dict) else None,
        week_day1,
        pv_df["pv_total_kwh"].sum(min_count=1) if "pv_total_kwh" in pv_df.columns else None,
        result.get("pv_kwh_p50") if isinstance(result, dict) else None,
        result.get("pv_p50_kwh") if isinstance(result, dict) else None,
        metrics.get("pv_kwh_p50") if isinstance(metrics, dict) else None,
        metrics.get("pv_p50_kwh") if isinstance(metrics, dict) else None,
        weather_ensemble.get("pv_totals_kwh", {}).get("p50")
        if isinstance(weather_ensemble, dict) and isinstance(weather_ensemble.get("pv_totals_kwh"), dict)
        else None,
    )
    return float(pv_total_kwh) if pv_total_kwh is not None else None


def resolve_tomorrow_pv_low_high_kwh(
    result: dict | None,
    weather_ensemble: dict | None,
    tomorrow_p50_kwh: float | None = None,
) -> tuple[float | None, float | None]:
    range_payload = result.get("pv_tomorrow_low_high_kwh") if isinstance(result, dict) else None
    if not isinstance(range_payload, dict) and isinstance(weather_ensemble, dict):
        range_payload = weather_ensemble.get("pv_tomorrow_low_high_kwh")

    if isinstance(range_payload, dict):
        low_raw = pd.to_numeric(pd.Series([range_payload.get("low")]), errors="coerce").iloc[0]
        high_raw = pd.to_numeric(pd.Series([range_payload.get("high")]), errors="coerce").iloc[0]
        valid_models_raw = pd.to_numeric(pd.Series([range_payload.get("valid_models")]), errors="coerce").iloc[0]
        valid_models = int(valid_models_raw) if not pd.isna(valid_models_raw) else 0
        if valid_models >= 2 and not pd.isna(low_raw) and not pd.isna(high_raw):
            low = float(low_raw)
            high = float(high_raw)
            if low > high or low < 0 or high < 0:
                return None, None
            if tomorrow_p50_kwh is not None:
                p50_raw = pd.to_numeric(pd.Series([tomorrow_p50_kwh]), errors="coerce").iloc[0]
                if not pd.isna(p50_raw):
                    p50 = float(p50_raw)
                    if p50 < (low - 0.01) or p50 > (high + 0.01):
                        return None, None
            return low, high

    return None, None


def resolve_week_ahead_total_pv_kwh(pv_week_ahead: list[dict] | list[float] | None) -> float | None:
    if not isinstance(pv_week_ahead, list) or not pv_week_ahead:
        return 0.0

    pv_totals = 0.0
    for item in pv_week_ahead[:7]:
        value = None
        if isinstance(item, dict):
            value = pd.to_numeric(
                pd.Series([
                    item.get("p50_kwh"),
                    item.get("pv_p50_kwh"),
                    item.get("pv_total_kwh"),
                    item.get("pv_kwh"),
                ]),
                errors="coerce",
            ).dropna()
            if not value.empty:
                pv_totals += float(value.iloc[0])
            continue

        scalar_value = pd.to_numeric(pd.Series([item]), errors="coerce").iloc[0]
        if not pd.isna(scalar_value):
            pv_totals += float(scalar_value)

    return float(pv_totals)


def summarize_model_diagnostics(weather_ensemble: dict) -> dict:
    if not isinstance(weather_ensemble, dict):
        return {"selected": 0, "ok": 0, "failed": 0, "failed_models": [], "derived_models": [], "missing_important": []}
    selected = [str(v) for v in weather_ensemble.get("selected_models", []) if isinstance(v, str)]
    failed = [str(v) for v in weather_ensemble.get("failed_models", []) if isinstance(v, str)]
    derived_map = weather_ensemble.get("derived_irradiance_by_model", {}) if isinstance(weather_ensemble.get("derived_irradiance_by_model"), dict) else {}
    missing_map = weather_ensemble.get("missing_vars_by_model", {}) if isinstance(weather_ensemble.get("missing_vars_by_model"), dict) else {}
    derived_models = sorted([model for model, used in derived_map.items() if bool(used)])
    missing_important: list[str] = []
    for model, vars_list in missing_map.items():
        if not isinstance(vars_list, list) or not vars_list:
            continue
        missing_important.append(f"{model}: {', '.join(str(v) for v in vars_list)}")
    return {
        "selected": len(selected),
        "ok": max(len(selected) - len(failed), 0),
        "failed": len(failed),
        "failed_models": failed,
        "derived_models": derived_models,
        "missing_important": missing_important,
    }


def render_modern_table(df: pd.DataFrame, column_config: dict | None = None) -> None:
    if df is None or df.empty:
        st.info("No data available.")
        return

    # Streamlit/pyarrow rejects duplicate labels; backend payloads can
    # occasionally include accidental duplicate fields after schema drift.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Keep column formatting config in sync with the final rendered dataframe.
    if column_config:
        column_config = {k: v for k, v in column_config.items() if k in df.columns}

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )


def render_selectable_table(
    df: pd.DataFrame,
    *,
    key: str,
    column_config: dict | None = None,
) -> int | None:
    if df is None or df.empty:
        st.info("No data available.")
        return None

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    if column_config:
        column_config = {k: v for k, v in column_config.items() if k in df.columns}

    dataframe_kwargs = {
        "use_container_width": True,
        "hide_index": True,
        "column_config": column_config,
        "key": key,
    }

    if "on_select" not in inspect.signature(st.dataframe).parameters:
        st.dataframe(df, **dataframe_kwargs)
        return None

    event = st.dataframe(
        df,
        on_select="rerun",
        selection_mode="single-row",
        **dataframe_kwargs,
    )

    selection_rows: list[int] = []
    if isinstance(event, dict):
        selection_rows = event.get("selection", {}).get("rows", []) or []
    else:
        selection = getattr(event, "selection", None)
        selection_rows = getattr(selection, "rows", []) if selection is not None else []

    if not selection_rows:
        return None

    try:
        return int(selection_rows[0])
    except Exception:
        return None


def metric_with_help(container, label: str, value: str) -> None:
    if "help" in inspect.signature(st.metric).parameters:
        container.metric(label, value, help=METRIC_TOOLTIPS[label])
        return
    safe_help = METRIC_TOOLTIPS[label].replace('"', "&quot;")
    container.markdown(
        f'**{label}** <span class="info-tooltip" title="{safe_help}">ⓘ</span>',
        unsafe_allow_html=True,
    )
    container.metric(label="", value=value)


def _esc_attr(s: object) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _pv_quality_flag_html(color: str, tooltip: str) -> str:
    col = (color or "").strip() or "#94a3b8"
    tip = _esc_attr(tooltip)
    return (
        f"<span title=\"{tip}\" "
        "style=\"display:inline-block;width:16px;height:12px;"
        f"background:{col};"
        "clip-path:polygon(0 0, 90% 0, 100% 50%, 90% 100%, 0 100%);"
        "border-radius:2px;"
        "box-shadow:0 0 0 1px rgba(255,255,255,0.18) inset;"
        "flex:0 0 auto;\">"
        "</span>"
    )


def _pv_quality_level_from_label(label: str) -> int:
    m = {
        "Excellent": 5,
        "Good": 4,
        "Mixed": 3,
        "Poor": 2,
        "Very low": 1,
    }
    return m.get((label or "").strip(), 3)


def _pv_quality_signal_html(label: str, color: str, tooltip: str) -> str:
    lvl = _pv_quality_level_from_label(label)
    col = (color or "").strip() or "#94a3b8"
    tip = html.escape(tooltip or "", quote=True)

    heights = [6, 8, 10, 12, 14]
    bars: list[str] = []
    for i, h in enumerate(heights, start=1):
        filled = i <= lvl
        bg = col if filled else "rgba(255,255,255,0.16)"
        bars.append(
            f"<span style='display:inline-block;width:4px;height:{h}px;"
            f"background:{bg};border-radius:2px;'></span>"
        )

    return (
        f"<span title=\"{tip}\" style='display:inline-flex;gap:2px;align-items:flex-end;'>"
        + "".join(bars)
        + "</span>"
    )


def render_pv_quality_widget(
    container,
    pv_df: pd.DataFrame,
    pv_quality_dict: dict,
    tomorrow_date: dt.date,
    pv_tomorrow_low_kwh: float | None = None,
    pv_tomorrow_high_kwh: float | None = None,
    tomorrow_weather_code: int | float | str | None = None,
    tomorrow_source_label: str | None = None,
    tomorrow_source_days: int | float | str | None = None,
) -> None:
    _ = pv_df

    score = int(_safe_float((pv_quality_dict or {}).get("score"), 0.0))
    score = min(100, max(0, score))
    ratio_percent = max(0.0, min(_safe_float((pv_quality_dict or {}).get("ratio"), 0.0) * 100.0, 100.0))

    pv_label = str((pv_quality_dict or {}).get("label") or "Mixed")
    pv_color = str((pv_quality_dict or {}).get("color") or "#94a3b8")
    offpeak_windows = core.get_offpeak_windows(tomorrow_date)
    expensive_windows = core.get_expensive_windows(tomorrow_date)
    offpeak_segments = windows_to_segments(offpeak_windows)

    if not offpeak_windows and not expensive_windows:
        timeline_base = "background:rgba(148,163,184,0.30);"
    else:
        timeline_base = "background:rgba(214,40,40,0.85);"

    overlays: list[str] = []
    for start_min, end_min in offpeak_segments:
        left_pct = clamp_pct((start_min / 1440.0) * 100.0)
        width_pct = clamp_pct(((end_min - start_min) / 1440.0) * 100.0)
        radius_bits: list[str] = []
        if start_min == 0:
            radius_bits.append("border-top-left-radius:999px;border-bottom-left-radius:999px;")
        if end_min == 1440:
            radius_bits.append("border-top-right-radius:999px;border-bottom-right-radius:999px;")
        radius_css = "".join(radius_bits)
        overlays.append(
            "<div style='position:absolute;top:0;bottom:0;left:{left:.5f}%;width:{width:.5f}%;"
            "background:#52b788;{radius}'></div>".format(
                left=left_pct,
                width=width_pct,
                radius=radius_css,
            )
        )

    summary_start, summary_end = (offpeak_windows[0] if offpeak_windows else ("00:00", "24:00"))
    offpeak_summary, peak_summary = make_summary_lines(summary_start, summary_end)
    summary_line = offpeak_summary if not peak_summary else f"{offpeak_summary} - {peak_summary}"
    summary_html = f"<div style='margin-top:0.30rem;font-size:0.70rem;opacity:0.92;'>{summary_line}</div>"

    savings_total = pv_quality_dict.get("savings_eur_total")
    hourly = pv_quality_dict.get("hourly_savings_eur_tomorrow")
    base_cost = pv_quality_dict.get("baseline_cost_eur_total")
    plan_cost = pv_quality_dict.get("plan_cost_eur_total")
    if savings_total is not None and isinstance(hourly, list) and len(hourly) == 24 and base_cost is not None and plan_cost is not None:
        s = float(savings_total)
        pill_color = "#52b788" if s >= 0 else "#d62828"
        sign = "+" if s >= 0 else "−"
        pill = f"{sign}€{abs(s):.2f}"
        max_abs = max(0.01, max(abs(float(x)) for x in hourly))
        bars: list[str] = []
        for i, val in enumerate(hourly):
            v = float(val)
            h_pct = (abs(v) / max_abs) * 100.0
            col = "rgba(60,220,150,0.95)" if v >= 0 else "rgba(214,40,40,0.95)"
            hh = f"{i:02d}:00"
            tip = f"{hh}  {('+' if v >= 0 else '−')}€{abs(v):.2f}"
            bars.append(
                f"<div title='{tip}' style='flex:1;height:{h_pct:.1f}%;"
                f"background:{col};border-radius:2px;'></div>"
            )

        savings_html = (
            "<div style='margin-top:0.65rem;padding-top:0.55rem;border-top:1px solid rgba(255,255,255,0.10);'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;'>"
            "<div style='font-size:0.72rem;opacity:0.85;text-transform:uppercase;letter-spacing:0.06em;'>"
            "Savings"
            "</div>"
            f"<div style='font-size:0.95rem;font-weight:800;color:{pill_color};'>{pill}</div>"
            "</div>"
            f"<div style='margin-top:0.22rem;font-size:0.70rem;opacity:0.80;'>"
            f"No battery: €{float(base_cost):.2f} · Battery plan: €{float(plan_cost):.2f}"
            "</div>"
            "<div style='margin-top:0.38rem;height:18px;display:flex;gap:1px;"
            "align-items:flex-end;background:rgba(255,255,255,0.06);"
            "border:1px solid rgba(255,255,255,0.10);border-radius:8px;padding:4px;'>"
            + "".join(bars)
            + "</div>"
            "<div style='margin-top:0.22rem;font-size:0.68rem;opacity:0.75;'>"
            "Hourly savings (00–24). Total includes tonight 22–24 charging."
            "</div>"
            "</div>"
        )
    else:
        savings_html = "<div style='margin-top:0.50rem;font-size:0.70rem;opacity:0.78;'>Run forecast to see € savings.</div>"

    weather_icon = weather_code_to_icon(tomorrow_weather_code)
    weather_label = weather_code_to_label(tomorrow_weather_code)

    w_tip = f"Tomorrow weather (forecast): {weather_label}"
    if tomorrow_weather_code is not None:
        w_tip += f" (WMO {tomorrow_weather_code})"
    if tomorrow_source_label:
        w_tip += f" | Source: {tomorrow_source_label}"
    if tomorrow_source_days:
        w_tip += f" ({tomorrow_source_days}d)"
    pv_tip = (
        f"PV quality indicator (not weather). Label: {pv_label}. "
        f"Score: {score}/100. Clear-sky ratio: {ratio_percent:.0f}%."
    )

    pv_quality_icon = _pv_quality_signal_html(pv_label, pv_color, pv_tip)
    icons_html = (
        "<div style='display:inline-flex;gap:0.5rem;align-items:center;white-space:nowrap;'>"
        f"{pv_quality_icon}"
        f"<span title=\"{_esc_attr(w_tip)}\" style='font-size:1.25rem;line-height:1;'>{weather_icon}</span>"
        "</div>"
    )

    header_html = f"""
    <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;white-space:nowrap;">
      <div style="display:flex;align-items:center;gap:10px;min-width:0;overflow:hidden;">
        <div style="font-weight:700;letter-spacing:0.03em;opacity:0.92;">PV OUTLOOK</div>
        {icons_html}
      </div>
      <div style="font-weight:700;opacity:0.9;white-space:nowrap;">{score}/100</div>
    </div>
    """

    pv_range_html = ""
    if pv_tomorrow_low_kwh is not None and pv_tomorrow_high_kwh is not None:
        pv_range_html = (
            "<div style='margin-top:0.20rem;font-size:0.74rem;opacity:0.82;'>"
            f"Low {pv_tomorrow_low_kwh:.2f} kWh - High {pv_tomorrow_high_kwh:.2f} kWh"
            "</div>"
        )

    container.markdown(
        (
            "<div style='border:1px solid rgba(255,255,255,0.12);border-radius:16px;padding:0.65rem 0.75rem;"
            "background:linear-gradient(140deg, rgba(43,48,58,0.9), rgba(20,24,31,0.85));min-width:245px;'>"
            "<div style='display:flex;flex-direction:column;gap:0.35rem;'>"
            f"{header_html}"
            "</div>"
            "<div style='margin-top:0.35rem;font-size:0.95rem;font-weight:650;'>"
            f"{pv_label} day · {pv_quality_dict['pv_total_kwh']:.1f} kWh"
            "</div>"
            + pv_range_html
            + f"<div style='margin-top:0.45rem;height:{UI_PROGRESS_BAR_HEIGHT_PX}px;border-radius:999px;overflow:hidden;background:rgba(255,255,255,0.12);'>"
            f"<div style='height:100%;width:{ratio_percent:.1f}%;background:linear-gradient(90deg,#d62828 0%,#f4a261 45%,#52b788 70%,#2a9d8f 100%);'></div>"
            "</div>"
            "<div style='margin-top:0.32rem;font-size:0.72rem;opacity:0.8;'>"
            f"{ratio_percent:.0f}% of clear-sky potential"
            "</div>"
            "<div style='margin-top:0.55rem;'>"
            f"<div title='Green = off-peak, Red = peak' style='height:{UI_PROGRESS_BAR_HEIGHT_PX}px;width:100%;position:relative;{timeline_base}"
            "border-radius:999px;overflow:hidden;'>"
            + "".join(overlays)
            + "</div>"
            + "<div style='margin-top:0.28rem;font-size:0.68rem;opacity:0.8;'>"
            "Tariff timeline (00–24)"
            "</div>"
            + summary_html
            + savings_html
            + (
                f"<div style='margin-top:0.30rem;font-size:0.68rem;opacity:0.78;'>"
                f"DEBUG PV score: {score} label: {pv_label} ratio: {ratio_percent:.1f}%"
                "</div>"
                if APP_DEBUG
                else ""
            )
            + "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_key_charging_widget(container, allowed_charge_kw: float, cutoff_soc_pct: float) -> None:
    power_pct = clamp_pct((allowed_charge_kw / 7.0) * 100.0)
    soc_pct = clamp_pct(cutoff_soc_pct)
    tip = (
        "These are the two FusionSolar settings for tonight. "
        "Allowed AC charge power is the grid charging power limit. "
        "AC charge cutoff SOC is the battery level where grid charging stops. "
        "Example: 0.50 kW and 70%."
    )
    container.markdown(
        (
            "<div style='border:1px solid rgba(255,255,255,0.12);border-radius:16px;padding:0.75rem 0.85rem;"
            "background:linear-gradient(140deg, rgba(43,48,58,0.9), rgba(20,24,31,0.85));'>"
            "<div style='display:flex;align-items:center;justify-content:flex-start;gap:0.5rem;'>"
            "<div style='font-size:0.72rem;opacity:0.8;text-transform:uppercase;letter-spacing:0.06em;'>Key charging targets</div>"
            f"<span class='info-tooltip' title='{_esc_attr(tip)}'>ⓘ</span>"
            "</div>"
            "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0.9rem;margin-top:0.55rem;'>"
            "<div style='padding:0.45rem 0.55rem;border:1px solid rgba(255,255,255,0.08);border-radius:12px;background:rgba(255,255,255,0.02);'>"
            "<div style='font-size:0.72rem;opacity:0.84;'>Allowed AC charge power (kW)</div>"
            f"<div style='margin-top:0.22rem;font-size:1.45rem;font-weight:700;'>{allowed_charge_kw:.2f}</div>"
            f"<div style='margin-top:0.32rem;height:{UI_PROGRESS_BAR_HEIGHT_PX}px;border-radius:999px;background:rgba(255,255,255,0.12);overflow:hidden;'>"
            f"<div style='height:100%;width:{power_pct:.1f}%;background:linear-gradient(90deg,#4cc9f0,#4895ef,#4361ee);'></div>"
            "</div>"
            "<div style='margin-top:0.22rem;font-size:0.68rem;opacity:0.78;'>Range: 0 to 7 kW</div>"
            "</div>"
            "<div style='padding:0.45rem 0.55rem;border:1px solid rgba(255,255,255,0.08);border-radius:12px;background:rgba(255,255,255,0.02);'>"
            "<div style='font-size:0.72rem;opacity:0.84;'>AC charge cutoff SOC (%)</div>"
            f"<div style='margin-top:0.22rem;font-size:1.45rem;font-weight:700;'>{cutoff_soc_pct:.1f}%</div>"
            f"<div style='margin-top:0.32rem;height:{UI_PROGRESS_BAR_HEIGHT_PX}px;border-radius:999px;background:rgba(255,255,255,0.12);overflow:hidden;'>"
            f"<div style='height:100%;width:{soc_pct:.1f}%;background:linear-gradient(90deg,#f4a261,#e9c46a,#52b788);'></div>"
            "</div>"
            "<div style='margin-top:0.22rem;font-size:0.68rem;opacity:0.78;'>Range: 0 to 100%</div>"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def windows_to_segments(windows: list[tuple[str, str]]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for start, end in windows:
        start_min = parse_hhmm(start, allow_24_end=False)
        end_min = parse_hhmm(end, allow_24_end=True)
        segments.extend(compute_offpeak_segments(start_min, end_min))

    return segments


def clamp_pct(x: float) -> float | None:
    return max(0.0, min(x, 100.0))



def load_api_token() -> str:
    env_token = os.getenv("PVBP_API_TOKEN", "").strip()
    if env_token:
        return env_token
    if API_TOKEN_FILE.exists():
        return API_TOKEN_FILE.read_text(encoding="utf-8").strip()
    raise RuntimeError("Missing API token. Set PVBP_API_TOKEN or create local_state/api_token.txt.")


def api_headers() -> dict:
    return {"Authorization": f"Bearer {load_api_token()}"}




@st.cache_resource
def http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.4,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET", "PUT"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def api_get(path: str) -> dict:
    response = http_session().get(f"{API_BASE_URL}{path}", headers=api_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def api_put(path: str, payload: dict) -> dict:
    response = http_session().put(f"{API_BASE_URL}{path}", headers=api_headers(), json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict) -> dict:
    url = f"{API_BASE_URL}{path}"
    delays = [0.5, 1.0, 2.0]

    for attempt, delay in enumerate([0.0] + delays):
        if delay:
            time.sleep(delay)

        try:
            response = http_session().post(url, headers=api_headers(), json=payload, timeout=120)
            if response.status_code == 423:
                if attempt < len(delays):
                    continue
                raise RuntimeError("Backend is busy running a forecast. Try again.")

            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < len(delays):
                continue
            raise

    raise RuntimeError("Could not complete backend request.")


def df_from_split(payload: dict) -> pd.DataFrame:
    return pd.read_json(StringIO(json.dumps(payload)), orient="split")


def series_from_split(payload: dict) -> pd.Series:
    frame = df_from_split(payload)
    if "value" in frame.columns:
        return frame["value"]
    return pd.Series(dtype=float)


def _parse_json_dict_maybe(payload: object) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def run_history_from_backend(show_all_runs: bool = False, days: int = 30) -> pd.DataFrame:
    history_columns = [
        "run_id",
        "Date",
        "Run at",
        "Status",
        "Status label",
        "Models",
        "PV p50",
        "PV p10",
        "PV p90",
        "PV range (p10–p90)",
        "Load (estimated)",
        "Charge",
        "Warnings",
        "Duration (ms)",
        "Models OK/Failed",
        "Primary model",
        "warnings_count",
        "run_type",
        "models_raw",
        "warnings_raw",
        "models_summary_raw",
        "cutoff_soc",
        "pv_quality_label",
        "pv_quality_score",
        "PV quality",
    ]
    try:
        show_all_text = "true" if show_all_runs else "false"
        items = api_get(f"/v1/results/history?days={max(1, int(days))}&show_all_runs={show_all_text}").get("items", [])
    except Exception:
        return pd.DataFrame(columns=history_columns)

    rows = []
    for item in items:
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        status_raw = str(item.get("status") or "").strip().lower()
        warnings_count = int(item.get("warnings_count") or 0)
        if status_raw == "error":
            status_label = "❌ Error"
        elif status_raw == "degraded" or warnings_count > 0:
            status_label = "⚠ Degraded"
        else:
            status_label = "✅ OK"

        models_summary = item.get("models_summary") if isinstance(item.get("models_summary"), dict) else {}
        models_list = models_summary.get("selected_models") if isinstance(models_summary, dict) else []
        if not isinstance(models_list, list):
            models_list = []
        models_text = ", ".join(str(m) for m in models_list) if models_list else "—"
        models_ok_count = int(item.get("models_ok_count") or 0)
        models_failed_count = int(item.get("models_failed_count") or 0)
        models_total_count = models_ok_count + models_failed_count
        if models_failed_count > 0:
            models_summary_text = f"{models_ok_count} OK, {models_failed_count} failed"
        elif models_total_count > 0:
            models_summary_text = f"{models_total_count} models OK"
        else:
            models_summary_text = "—"

        pv_p10_raw = item.get("pv_p10_kwh")
        pv_p10 = float(pv_p10_raw) if pv_p10_raw is not None else None
        pv_p50 = _safe_float(item.get("pv_p50_kwh") or metrics.get("pv_forecast_kwh"), 0.0)
        pv_p90_raw = item.get("pv_p90_kwh")
        pv_p90 = float(pv_p90_raw) if pv_p90_raw is not None else None
        warnings_raw = item.get("warnings") if isinstance(item.get("warnings"), list) else []
        warnings_text = " | ".join(str(w) for w in warnings_raw) if warnings_raw else (f"{warnings_count} warning(s)" if warnings_count else "None")

        pv_quality = item.get("pv_quality") if isinstance(item.get("pv_quality"), dict) else {}
        pv_quality_label = str(pv_quality.get("label") or "").strip()
        pv_quality_score_raw = pv_quality.get("score")
        try:
            pv_quality_score = int(float(pv_quality_score_raw)) if pv_quality_score_raw is not None else None
        except (TypeError, ValueError):
            pv_quality_score = None
        if status_raw == "error":
            pv_quality_display = "—"
        elif pv_quality_label and pv_quality_score is not None:
            pv_quality_display = f"{pv_quality_label} ({pv_quality_score}/100)"
        elif pv_quality_label:
            pv_quality_display = pv_quality_label
        else:
            pv_quality_display = "—"

        rows.append({
            "run_id": str(item.get("run_id") or ""),
            "Date": item.get("target_date"),
            "Run at": item.get("run_at"),
            "Status": status_raw or "ok",
            "Status label": status_label,
            "Models": models_summary_text,
            "PV p50": round(pv_p50, 2),
            "PV p10": round(pv_p10, 2) if pv_p10 is not None else None,
            "PV p90": round(pv_p90, 2) if pv_p90 is not None else None,
            "PV range (p10–p90)": f"{pv_p10:.2f}–{pv_p90:.2f} kWh" if (pv_p10 is not None and pv_p90 is not None) else "—",
            "Load (estimated)": round(_safe_float(metrics.get("cons_forecast_kwh"), 0.0), 2),
            "Charge": round(_safe_float(metrics.get("charge_kw"), 0.0), 2),
            "Warnings": warnings_text,
            "Duration (ms)": item.get("run_duration_ms"),
            "Models OK/Failed": f"{models_ok_count}/{models_failed_count}",
            "Primary model": str(item.get("primary_model_id") or "—"),
            "warnings_count": warnings_count,
            "run_type": str(item.get("run_type") or "manual"),
            "models_raw": models_text,
            "warnings_raw": warnings_raw,
            "models_summary_raw": models_summary,
            "cutoff_soc": metrics.get("cutoff_soc"),
            "pv_quality_label": pv_quality_label or "—",
            "pv_quality_score": pv_quality_score,
            "PV quality": pv_quality_display,
        })

    if not rows:
        return pd.DataFrame(columns=history_columns)

    history_df = pd.DataFrame(rows)
    history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
    history_df["Run at"] = pd.to_datetime(history_df["Run at"], errors="coerce")
    history_df = history_df.dropna(subset=["Date"])
    history_df = history_df.sort_values(["Date", "Run at"], ascending=[True, True])
    return history_df


def _prepare_history_df(df: pd.DataFrame, all_runs: bool, show_run_at: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    working = df.copy()

    colmap = {}
    for c in working.columns:
        lc = c.strip().lower()
        if lc in ("run_at", "run at", "runat", "created_at", "created at"):
            colmap[c] = "Run at"
        if lc in ("date", "target_date", "target date"):
            colmap[c] = "Date"
    if colmap:
        working = working.rename(columns=colmap)

    # Normalize core history schema once and centrally so downstream sorting/
    # rendering logic can rely on stable dtypes even if backend payload drifts.
    working["Date"] = pd.to_datetime(working.get("Date"), errors="coerce").dt.normalize()
    if "Run at" not in working.columns:
        working["Run at"] = pd.NaT
    working["Run at"] = (
        pd.to_datetime(working.get("Run at"), errors="coerce", utc=True)
        .dt.tz_convert("Europe/Brussels")
    )

    if not all_runs:
        if "Date" in working.columns and "Run at" in working.columns:
            working = working.sort_values(["Date", "Run at"]).groupby("Date", as_index=False).tail(1)
        elif "Date" in working.columns:
            working = working.drop_duplicates(subset=["Date"], keep="last")

    sort_cols: list[str] = []
    if "Date" in working.columns:
        sort_cols.append("Date")
    if all_runs and "Run at" in working.columns:
        sort_cols.append("Run at")
    if sort_cols:
        working = working.sort_values(sort_cols, ascending=True)

    return working.reset_index(drop=True)


def _to_py_date(val):
    """Return a python datetime.date or None from date/datetime/Timestamp/str."""
    if val is None:
        return None
    if isinstance(val, dt.date) and not isinstance(val, dt.datetime):
        return val
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def _safe_filename_part(value: object, fallback: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        text = fallback
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _target_date_for_filename(value: object) -> str:
    date_value = _to_py_date(value)
    if date_value is None:
        return "unknown_date"
    return date_value.strftime("%Y%m%d")


def _run_label(row: pd.Series, fallback_index: int = 0) -> str:
    date_value = _to_py_date(row.get("Date"))
    date_part = date_value.isoformat() if date_value is not None else "unknown-date"
    status = str(row.get("Status label") or "—")
    run_type = str(row.get("run_type") or "manual")
    run_id = str(row.get("run_id") or f"row-{fallback_index + 1}")
    return f"{date_part} · {status} · {run_type} · {run_id}"


def _get_run_detail(run_id: str) -> dict:
    if not run_id:
        return {}
    cache = st.session_state.setdefault("_run_detail_cache", {})
    if run_id in cache:
        return cache[run_id]
    try:
        payload = api_get(f"/v1/results/run/{run_id}")
    except Exception:
        payload = {}
    cache[run_id] = payload
    return payload


def _extract_hourly_df(detail_payload: dict) -> pd.DataFrame:
    if isinstance(detail_payload.get("hourly"), list):
        hourly_df = pd.DataFrame([r for r in detail_payload.get("hourly", []) if isinstance(r, dict)])
        if not hourly_df.empty:
            if "ts_local" not in hourly_df.columns and "time" in hourly_df.columns:
                hourly_df = hourly_df.rename(columns={"time": "ts_local"})
            hourly_df["ts_local"] = pd.to_datetime(hourly_df.get("ts_local"), errors="coerce")
            for col in ["pv_total_kwh", "pv_kwh", "soc_end_pct", "soc_pct"]:
                if col in hourly_df.columns:
                    hourly_df[col] = pd.to_numeric(hourly_df[col], errors="coerce")
            if "pv_total_kwh" not in hourly_df.columns and "pv_kwh" in hourly_df.columns:
                hourly_df["pv_total_kwh"] = hourly_df["pv_kwh"]
            if "soc_end_pct" not in hourly_df.columns and "soc_pct" in hourly_df.columns:
                hourly_df["soc_end_pct"] = hourly_df["soc_pct"]
            return hourly_df[[c for c in ["ts_local", "pv_total_kwh", "soc_end_pct"] if c in hourly_df.columns]].dropna(subset=["ts_local"])

    if not (
        isinstance(detail_payload.get("pv"), dict)
        and isinstance(detail_payload.get("flows"), dict)
        and isinstance(detail_payload.get("soc"), dict)
    ):
        return pd.DataFrame()

    try:
        pv_df = df_from_split(detail_payload["pv"])
        flows_df = df_from_split(detail_payload["flows"])
        soc_series = series_from_split(detail_payload["soc"])
    except Exception:
        return pd.DataFrame()

    hourly_df = pd.concat([pv_df, flows_df], axis=1)
    if "soc_end_pct" not in hourly_df.columns:
        hourly_df["soc_end_pct"] = pd.to_numeric(soc_series, errors="coerce") * 100.0
    if "pv_total_kwh" not in hourly_df.columns and "pv_kwh" in hourly_df.columns:
        hourly_df["pv_total_kwh"] = pd.to_numeric(hourly_df["pv_kwh"], errors="coerce")
    hourly_df = hourly_df.reset_index().rename(columns={"index": "ts_local"})
    hourly_df["ts_local"] = pd.to_datetime(hourly_df["ts_local"], errors="coerce")
    return hourly_df[[c for c in ["ts_local", "pv_total_kwh", "soc_end_pct"] if c in hourly_df.columns]].dropna(subset=["ts_local"])


NOISY_DIFF_KEYS = {
    "run_id",
    "run_at_utc",
    "warnings_count",
    "status",
    "duration",
    "session_id",
    "session_state",
    "ui_state",
    "updated_at",
    "created_at",
}

NOISY_DIFF_PATH_FRAGMENTS = {
    "metadata",
    "timestamps",
    "session",
    "transient",
}


def _is_noisy_diff_path(path: str) -> bool:
    lowered_tokens = [token.strip().lower() for token in path.split(".") if token.strip()]
    if not lowered_tokens:
        return False
    if lowered_tokens[-1] in NOISY_DIFF_KEYS:
        return True
    return any(fragment in lowered_tokens for fragment in NOISY_DIFF_PATH_FRAGMENTS)


def _flatten_diff_payload(payload: object, prefix: str = "", include_noisy: bool = True) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(payload, dict):
        for key in sorted(payload.keys(), key=lambda x: str(x)):
            key_name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_diff_payload(payload[key], key_name, include_noisy=include_noisy))
        return out
    if isinstance(payload, list):
        if prefix and not include_noisy and _is_noisy_diff_path(prefix):
            return out
        out[prefix] = ", ".join(str(v) for v in payload)
        return out
    if prefix and not include_noisy and _is_noisy_diff_path(prefix):
        return out
    out[prefix] = "—" if payload is None else str(payload)
    return out


def _diff_table(left: dict, right: dict, include_noisy: bool = True) -> pd.DataFrame:
    left_flat = _flatten_diff_payload(left, include_noisy=include_noisy)
    right_flat = _flatten_diff_payload(right, include_noisy=include_noisy)
    keys = sorted(set(left_flat.keys()) | set(right_flat.keys()))
    rows = []
    for key in keys:
        a_val = left_flat.get(key, "∅")
        b_val = right_flat.get(key, "∅")
        if a_val != b_val:
            rows.append({"Field": key, "Run A": a_val, "Run B": b_val})
    return pd.DataFrame(rows)


def _render_run_inspector(filtered_df: pd.DataFrame) -> None:
    if filtered_df.empty:
        return

    filtered_df = filtered_df.sort_values(["Date", "Run at"], ascending=[False, False]).reset_index(drop=True)
    options = {
        f"{_run_label(row, i)}": i
        for i, row in filtered_df.iterrows()
    }
    default_pick_label = None
    selected_run_id = str(st.session_state.get("history_inspector_selected_run_id") or "").strip()
    if selected_run_id:
        for label, idx in options.items():
            option_run_id = str(filtered_df.iloc[idx].get("run_id") or "").strip()
            if option_run_id and option_run_id == selected_run_id:
                default_pick_label = label
                break
    labels = list(options.keys())
    default_idx = labels.index(default_pick_label) if default_pick_label in labels else 0

    picked = st.selectbox("Inspect a run", labels, key="history_inspector_run", index=default_idx)
    row = filtered_df.iloc[options[picked]]

    detail = {}
    run_id = str(row.get("run_id") or "").strip()
    if run_id:
        try:
            detail = api_get(f"/v1/results/run/{run_id}")
        except Exception:
            detail = {}

    warnings_list = []
    if isinstance(detail.get("warnings"), list):
        warnings_list = [str(w) for w in detail.get("warnings", [])]
    elif isinstance(row.get("warnings_raw"), list):
        warnings_list = [str(w) for w in row.get("warnings_raw", [])]

    with st.expander("Run Inspector", expanded=bool(default_pick_label)):
        tab_summary, tab_models, tab_inputs, tab_settings, tab_debug = st.tabs(
            ["Summary", "Weather models", "Inputs used", "Settings used", "Debug bundle"]
        )
        with tab_summary:
            date_label = _to_py_date(row.get("Date")) or row.get("Date") or "unknown-date"
            date_text = date_label.isoformat() if hasattr(date_label, "isoformat") else str(date_label)
            st.markdown(
                f"**Date:** {date_text}\n"
                f"**Run at:** {row['Run at'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['Run at']) else '—'}\n"
                f"**Status:** {row['Status label']}\n"
                f"**Run type:** {row.get('run_type', 'manual')}"
            )
            if warnings_list:
                st.warning("\n".join(f"• {w}" for w in warnings_list))
            elif int(row.get("warnings_count") or 0) > 0:
                st.warning(f"{int(row.get('warnings_count') or 0)} warning(s) were recorded for this run.")
            else:
                st.success("No warnings recorded.")

        with tab_models:
            models_summary = detail.get("models_summary") if isinstance(detail.get("models_summary"), dict) else row.get("models_summary_raw", {})
            selected_models = models_summary.get("selected_models", []) if isinstance(models_summary, dict) else []
            failed_models = models_summary.get("failed_models", []) if isinstance(models_summary, dict) else []
            weights_used = models_summary.get("weights_used", {}) if isinstance(models_summary, dict) else {}
            weather_ensemble = _parse_json_dict_maybe(detail.get("weather_ensemble_json"))
            if not weather_ensemble:
                weather_ensemble = detail.get("weather_ensemble") if isinstance(detail.get("weather_ensemble"), dict) else {}

            failed_models_set = set(str(m) for m in failed_models)
            failed_reasons = weather_ensemble.get("failed_model_reasons") if isinstance(weather_ensemble.get("failed_model_reasons"), dict) else {}
            missing_by_model = weather_ensemble.get("missing_vars_by_model") if isinstance(weather_ensemble.get("missing_vars_by_model"), dict) else {}
            derived_by_model = weather_ensemble.get("derived_irradiance_by_model") if isinstance(weather_ensemble.get("derived_irradiance_by_model"), dict) else {}

            model_ids: list[str] = []
            for model_id in selected_models:
                model_text = str(model_id)
                if model_text and model_text not in model_ids:
                    model_ids.append(model_text)
            if isinstance(weights_used, dict):
                for model_id in weights_used.keys():
                    model_text = str(model_id)
                    if model_text and model_text not in model_ids:
                        model_ids.append(model_text)
            for model_id in failed_models_set:
                if model_id and model_id not in model_ids:
                    model_ids.append(model_id)

            st.caption("Short status in History keeps the table readable. Full per-model diagnostics are shown below.")
            if not model_ids:
                st.info("No model details available for this run.")
            else:
                model_rows = []
                for model_id in model_ids:
                    is_selected = model_id in [str(m) for m in selected_models]
                    is_failed = model_id in failed_models_set
                    failure_payload = failed_reasons.get(model_id, {})
                    failure_reason = ""
                    if isinstance(failure_payload, dict):
                        failure_reason = str(failure_payload.get("error") or failure_payload.get("reason") or "").strip()
                    elif failure_payload:
                        failure_reason = str(failure_payload).strip()
                    missing_vars = missing_by_model.get(model_id, [])
                    missing_count = len(missing_vars) if isinstance(missing_vars, list) else 0
                    model_rows.append({
                        "Model": model_id,
                        "Selected": "Yes" if is_selected else "No",
                        "Weight": _safe_float(weights_used.get(model_id) if isinstance(weights_used, dict) else None, 0.0),
                        "Status": "Failed" if is_failed else "OK",
                        "Failure reason": (failure_reason[:117] + "...") if len(failure_reason) > 120 else (failure_reason or "—"),
                        "Missing vars": int(missing_count),
                        "Derived irradiance": "Yes" if bool(derived_by_model.get(model_id)) else "No",
                    })

                model_details_df = pd.DataFrame(model_rows)
                status_counts = model_details_df["Status"].value_counts()
                ok_count = int(status_counts.get("OK", 0))
                failed_count = int(status_counts.get("Failed", 0))
                st.markdown(f"**Weather models:** {ok_count} OK, {failed_count} failed")
                model_column_config = {
                    "Weight": st.column_config.NumberColumn(format="%.4f"),
                    "Missing vars": st.column_config.NumberColumn(format="%d"),
                }
                render_modern_table(model_details_df, column_config=model_column_config)

        with tab_inputs:
            inputs_used = detail.get("inputs_used") if isinstance(detail.get("inputs_used"), dict) else {}
            if inputs_used:
                st.json(inputs_used, expanded=False)
            else:
                st.info("Inputs snapshot is not available for this run.")

        with tab_settings:
            settings_used = detail.get("settings_used") if isinstance(detail.get("settings_used"), dict) else {}
            if settings_used:
                st.json(settings_used, expanded=False)
            else:
                st.info("Settings snapshot is not available for this run.")

        with tab_debug:
            def _parse_config_json(raw: object) -> dict | str:
                if isinstance(raw, dict):
                    return raw
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            return parsed
                    except (TypeError, ValueError):
                        pass
                    return raw
                return {}

            def _hourly_records_from_detail(detail_payload: dict) -> list[dict]:
                if isinstance(detail_payload.get("hourly"), list):
                    return [r for r in detail_payload.get("hourly", []) if isinstance(r, dict)]

                if not (
                    isinstance(detail_payload.get("pv"), dict)
                    and isinstance(detail_payload.get("flows"), dict)
                    and isinstance(detail_payload.get("soc"), dict)
                ):
                    return []

                try:
                    pv_df = df_from_split(detail_payload["pv"])
                    flows_df = df_from_split(detail_payload["flows"])
                    soc_series = series_from_split(detail_payload["soc"])
                except Exception:
                    return []

                pv_df = pv_df.copy()
                if "index" in pv_df.columns:
                    pv_df = pv_df.drop(columns=["index"])
                flows_df = flows_df.copy()
                if "index" in flows_df.columns:
                    flows_df = flows_df.drop(columns=["index"])

                hourly_df = pd.concat([pv_df, flows_df], axis=1)
                hourly_df["soc_end_pct"] = pd.to_numeric(soc_series, errors="coerce") * 100.0
                hourly_df = hourly_df.reset_index().rename(columns={"index": "ts_local"})
                hourly_df["ts_local"] = hourly_df["ts_local"].astype(str)
                return json.loads(hourly_df.to_json(orient="records", date_format="iso"))

            def _build_debug_bundle(detail_payload: dict, history_row: pd.Series, warnings: list[str]) -> dict:
                prebuilt = detail_payload.get("debug_bundle") if isinstance(detail_payload.get("debug_bundle"), dict) else {}
                weather_ensemble = detail_payload.get("weather_ensemble") if isinstance(detail_payload.get("weather_ensemble"), dict) else {}
                if not weather_ensemble and isinstance(prebuilt.get("weather_ensemble"), dict):
                    weather_ensemble = prebuilt.get("weather_ensemble", {})

                pv_totals = detail_payload.get("pv_totals_kwh") if isinstance(detail_payload.get("pv_totals_kwh"), dict) else {}
                if not pv_totals and isinstance(weather_ensemble.get("pv_totals_kwh"), dict):
                    pv_totals = weather_ensemble.get("pv_totals_kwh", {})
                if not pv_totals:
                    pv_totals = {
                        "p10": float(history_row.get("PV p10")) if pd.notna(history_row.get("PV p10")) else None,
                        "p50": float(history_row.get("PV p50") or 0.0),
                        "p90": float(history_row.get("PV p90")) if pd.notna(history_row.get("PV p90")) else None,
                    }

                outputs = detail_payload.get("outputs") if isinstance(detail_payload.get("outputs"), dict) else {}
                if not outputs:
                    outputs = detail_payload.get("metrics") if isinstance(detail_payload.get("metrics"), dict) else {}
                if not outputs and isinstance(prebuilt.get("outputs"), dict):
                    outputs = prebuilt.get("outputs", {})

                bundle: dict[str, object] = {
                    "run_id": str(detail_payload.get("run_id") or history_row.get("run_id") or ""),
                    "inputs_used": detail_payload.get("inputs_used") if isinstance(detail_payload.get("inputs_used"), dict) else prebuilt.get("inputs_used", {}),
                    "config_hash": str(detail_payload.get("config_hash") or prebuilt.get("config_hash") or ""),
                    "config_json": _parse_config_json(detail_payload.get("config_json") or prebuilt.get("config_json") or {}),
                    "weather_ensemble": weather_ensemble,
                    "warnings": warnings,
                    "outputs": outputs,
                    "pv_totals": pv_totals,
                }

                hourly_records = _hourly_records_from_detail(detail_payload)
                if hourly_records:
                    bundle["hourly"] = hourly_records
                return bundle

            debug_bundle = _build_debug_bundle(detail, row, warnings_list)
            debug_bundle_json = json.dumps(debug_bundle, indent=2, ensure_ascii=False)

            st.code(debug_bundle_json, language="json")
            st.caption("Copy JSON debug bundle: use the copy icon in the top-right of the code block.")
            st.download_button(
                "Download JSON debug bundle",
                data=debug_bundle_json,
                file_name=(
                    f"debug_bundle_{_target_date_for_filename(row.get('Date'))}"
                    f"_{_safe_filename_part(debug_bundle.get('run_id'), 'run')}"
                    f"_{_safe_filename_part(row.get('Status'), 'ok')}"
                    f"_{_safe_filename_part(row.get('run_type'), 'manual')}.json"
                ),
                mime="application/json",
                key=f"history_inspector_debug_bundle_download_{run_id or 'row'}",
            )


def _render_compare_runs_block(filtered_df: pd.DataFrame) -> None:
    if filtered_df.empty:
        return

    working = filtered_df.sort_values(["Date", "Run at"], ascending=[False, False]).reset_index(drop=True)
    options = {_run_label(row, i): i for i, row in working.iterrows()}
    labels = list(options.keys())
    default_b_idx = 1 if len(labels) > 1 else 0

    with st.expander("Compare Runs", expanded=False):
        st.caption("Select two runs to answer what changed and why in one view.")
        c1, c2 = st.columns(2)
        with c1:
            run_a_label = st.selectbox("Run A", labels, index=0, key="compare_run_a")
        with c2:
            run_b_label = st.selectbox("Run B", labels, index=default_b_idx, key="compare_run_b")

        if run_a_label == run_b_label:
            st.info("Pick two different runs to compare.")
            return

        row_a = working.iloc[options[run_a_label]]
        row_b = working.iloc[options[run_b_label]]
        run_id_a = str(row_a.get("run_id") or "")
        run_id_b = str(row_b.get("run_id") or "")
        detail_a = _get_run_detail(run_id_a)
        detail_b = _get_run_detail(run_id_b)

        metrics_a = detail_a.get("metrics") if isinstance(detail_a.get("metrics"), dict) else {}
        metrics_b = detail_b.get("metrics") if isinstance(detail_b.get("metrics"), dict) else {}
        pv_a = float(row_a.get("PV p50") or metrics_a.get("pv_forecast_kwh") or 0.0)
        pv_b = float(row_b.get("PV p50") or metrics_b.get("pv_forecast_kwh") or 0.0)
        load_a = float(row_a.get("Load (estimated)") or metrics_a.get("cons_forecast_kwh") or 0.0)
        load_b = float(row_b.get("Load (estimated)") or metrics_b.get("cons_forecast_kwh") or 0.0)
        charge_a = float(row_a.get("Charge") or metrics_a.get("charge_kw") or 0.0)
        charge_b = float(row_b.get("Charge") or metrics_b.get("charge_kw") or 0.0)
        warn_a = int(row_a.get("warnings_count") or 0)
        warn_b = int(row_b.get("warnings_count") or 0)

        weather_a = detail_a.get("weather_ensemble") if isinstance(detail_a.get("weather_ensemble"), dict) else {}
        weather_b = detail_b.get("weather_ensemble") if isinstance(detail_b.get("weather_ensemble"), dict) else {}

        def _extract_pv_range_width(history_row: pd.Series, run_detail: dict) -> float | None:
            p10_raw = history_row.get("PV p10")
            p90_raw = history_row.get("PV p90")
            p10 = float(p10_raw) if pd.notna(p10_raw) else None
            p90 = float(p90_raw) if pd.notna(p90_raw) else None

            weather = run_detail.get("weather_ensemble") if isinstance(run_detail.get("weather_ensemble"), dict) else {}
            pv_totals = weather.get("pv_totals_kwh") if isinstance(weather.get("pv_totals_kwh"), dict) else {}
            if p10 is None:
                p10_alt = pv_totals.get("p10")
                p10 = float(p10_alt) if p10_alt is not None else None
            if p90 is None:
                p90_alt = pv_totals.get("p90")
                p90 = float(p90_alt) if p90_alt is not None else None

            if p10 is None or p90 is None:
                return None
            return p90 - p10

        def _extract_cutoff_soc_pct(run_detail: dict) -> float | None:
            metrics = run_detail.get("metrics") if isinstance(run_detail.get("metrics"), dict) else {}
            cutoff = metrics.get("cutoff_soc")
            if cutoff is None:
                return None
            cutoff_val = float(cutoff)
            return cutoff_val * 100.0 if cutoff_val <= 1.0 else cutoff_val

        pv_width_a = _extract_pv_range_width(row_a, detail_a)
        pv_width_b = _extract_pv_range_width(row_b, detail_b)
        cutoff_a = _extract_cutoff_soc_pct(detail_a)
        cutoff_b = _extract_cutoff_soc_pct(detail_b)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PV p50 Δ (kWh)", f"{pv_b - pv_a:+.2f}", help=f"A: {pv_a:.2f} → B: {pv_b:.2f}")
        m2.metric("Load Δ (kWh)", f"{load_b - load_a:+.2f}", help=f"A: {load_a:.2f} → B: {load_b:.2f}")
        m3.metric("Charge Δ (kW)", f"{charge_b - charge_a:+.2f}", help=f"A: {charge_a:.2f} → B: {charge_b:.2f}")
        m4.metric("Warnings Δ", f"{warn_b - warn_a:+d}", help=f"A: {warn_a} → B: {warn_b}")

        m5, m6 = st.columns(2)
        if pv_width_a is not None and pv_width_b is not None:
            m5.metric(
                "PV range width Δ (kWh)",
                f"{pv_width_b - pv_width_a:+.2f}",
                help=f"A: {pv_width_a:.2f} → B: {pv_width_b:.2f}",
            )
        else:
            m5.metric("PV range width Δ (kWh)", "—", help="Unavailable when uncertainty bounds (p10/p90) are missing.")

        if cutoff_a is not None and cutoff_b is not None:
            m6.metric("Cutoff SOC Δ (pp)", f"{cutoff_b - cutoff_a:+.1f}", help=f"A: {cutoff_a:.1f}% → B: {cutoff_b:.1f}%")
        else:
            m6.metric("Cutoff SOC Δ (pp)", "—", help="Unavailable when cutoff SOC is missing for one or both runs.")

        hourly_a = _extract_hourly_df(detail_a)
        hourly_b = _extract_hourly_df(detail_b)
        if not hourly_a.empty and not hourly_b.empty:
            overlay_df = hourly_a.rename(columns={"pv_total_kwh": "pv_a", "soc_end_pct": "soc_a"}).merge(
                hourly_b.rename(columns={"pv_total_kwh": "pv_b", "soc_end_pct": "soc_b"}),
                on="ts_local",
                how="outer",
            ).sort_values("ts_local")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=overlay_df["ts_local"], y=overlay_df.get("pv_a"), mode="lines", name="PV A", line=dict(color="#2ca02c")))
            fig.add_trace(go.Scatter(x=overlay_df["ts_local"], y=overlay_df.get("pv_b"), mode="lines", name="PV B", line=dict(color="#98df8a", dash="dash")))
            fig.add_trace(go.Scatter(x=overlay_df["ts_local"], y=overlay_df.get("soc_a"), mode="lines", name="SOC A", yaxis="y2", line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=overlay_df["ts_local"], y=overlay_df.get("soc_b"), mode="lines", name="SOC B", yaxis="y2", line=dict(color="#aec7e8", dash="dash")))
            fig.update_layout(
                template=PLOTLY_DARK,
                margin=dict(l=10, r=10, t=25, b=10),
                legend=dict(orientation="h", y=1.02, x=0),
                yaxis=dict(title="PV (kWh)"),
                yaxis2=dict(title="SOC (%)", overlaying="y", side="right"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Hourly overlays unavailable for one or both runs.")

        weather_focus_a = {
            "failed_models": weather_a.get("failed_models") or [],
            "weights_used": weather_a.get("weights_used") or {},
            "derived_irradiance_by_model": weather_a.get("derived_irradiance_by_model") or {},
            "missing_vars_by_model": weather_a.get("missing_vars_by_model") or {},
        }
        weather_focus_b = {
            "failed_models": weather_b.get("failed_models") or [],
            "weights_used": weather_b.get("weights_used") or {},
            "derived_irradiance_by_model": weather_b.get("derived_irradiance_by_model") or {},
            "missing_vars_by_model": weather_b.get("missing_vars_by_model") or {},
        }

        diffs_tabs = st.tabs(["Weather diff", "Inputs diff", "Settings diff", "Export"])
        with diffs_tabs[0]:
            weather_diff_df = _diff_table(weather_focus_a, weather_focus_b)
            if weather_diff_df.empty:
                st.success("No weather ensemble differences found.")
            else:
                st.dataframe(weather_diff_df, use_container_width=True, hide_index=True)
        with diffs_tabs[1]:
            inputs_a = detail_a.get("inputs_used") if isinstance(detail_a.get("inputs_used"), dict) else {}
            inputs_b = detail_b.get("inputs_used") if isinstance(detail_b.get("inputs_used"), dict) else {}
            inputs_diff_df = _diff_table(inputs_a, inputs_b)
            if inputs_diff_df.empty:
                st.success("No input differences found.")
            else:
                st.dataframe(inputs_diff_df, use_container_width=True, hide_index=True)
        include_noisy_settings = False
        with diffs_tabs[2]:
            include_noisy_settings = st.toggle(
                "Include noisy keys",
                value=False,
                key="compare_settings_diff_include_noisy",
                help="Show technical/transient fields (run_id, timestamps, metadata, session/UI state) for debugging.",
            )
            settings_a = detail_a.get("settings_used") if isinstance(detail_a.get("settings_used"), dict) else {}
            settings_b = detail_b.get("settings_used") if isinstance(detail_b.get("settings_used"), dict) else {}
            settings_diff_df = _diff_table(settings_a, settings_b, include_noisy=include_noisy_settings)
            if settings_diff_df.empty:
                st.success("No settings differences found.")
            else:
                st.dataframe(settings_diff_df, use_container_width=True, hide_index=True)
        with diffs_tabs[3]:
            compare_bundle = {
                "run_a": {"label": run_a_label, "run_id": run_id_a, "summary": {"pv_p50": pv_a, "pv_range_width_kwh": pv_width_a, "load": load_a, "charge_kw": charge_a, "cutoff_soc_pct": cutoff_a, "warnings_count": warn_a}},
                "run_b": {"label": run_b_label, "run_id": run_id_b, "summary": {"pv_p50": pv_b, "pv_range_width_kwh": pv_width_b, "load": load_b, "charge_kw": charge_b, "cutoff_soc_pct": cutoff_b, "warnings_count": warn_b}},
                "deltas": {
                    "pv_p50_kwh": pv_b - pv_a,
                    "pv_range_width_kwh": (pv_width_b - pv_width_a) if (pv_width_a is not None and pv_width_b is not None) else None,
                    "load_kwh": load_b - load_a,
                    "charge_kw": charge_b - charge_a,
                    "cutoff_soc_pp": (cutoff_b - cutoff_a) if (cutoff_a is not None and cutoff_b is not None) else None,
                    "warnings_count": warn_b - warn_a,
                },
                "diffs": {
                    "weather": _diff_table(weather_focus_a, weather_focus_b).to_dict(orient="records"),
                    "inputs": _diff_table(
                        detail_a.get("inputs_used") if isinstance(detail_a.get("inputs_used"), dict) else {},
                        detail_b.get("inputs_used") if isinstance(detail_b.get("inputs_used"), dict) else {},
                    ).to_dict(orient="records"),
                    "settings": _diff_table(
                        detail_a.get("settings_used") if isinstance(detail_a.get("settings_used"), dict) else {},
                        detail_b.get("settings_used") if isinstance(detail_b.get("settings_used"), dict) else {},
                        include_noisy=include_noisy_settings,
                    ).to_dict(orient="records"),
                },
            }
            compare_json = json.dumps(compare_bundle, indent=2, ensure_ascii=False)
            st.code(compare_json, language="json")
            st.download_button(
                "Download compare bundle JSON",
                data=compare_json,
                file_name=(
                    f"compare_{_target_date_for_filename(row_a.get('Date'))}_{_safe_filename_part(run_id_a, 'run_a')}"
                    f"_vs_{_target_date_for_filename(row_b.get('Date'))}_{_safe_filename_part(run_id_b, 'run_b')}.json"
                ),
                mime="application/json",
                key=f"compare_bundle_download_{run_id_a}_{run_id_b}",
            )


def _render_history_log_block() -> None:
    tooltip_heading("History log", TABLE_TOOLTIPS["History log"])

    with st.expander("History log", expanded=True):
        mode_col, c1, c2 = st.columns([1.3, 1.2, 1.2])

        with mode_col:
            st.radio(
                "Mode",
                options=["Simple", "Debug"],
                key="history_mode",
                horizontal=True,
                help="Simple keeps History easy to scan. Debug reveals deeper run diagnostics.",
            )

        history_debug_mode = st.session_state.get("history_mode", "Simple") == "Debug"
        if history_debug_mode:
            with c1:
                st.toggle(
                    "All runs",
                    key="history_all_runs",
                    help="Off = only the latest run per forecast day. On = show every run you made.",
                )

            with c2:
                if st.session_state.get("history_all_runs", False):
                    st.checkbox(
                        "Show run time",
                        key="history_show_run_at",
                        help="Shows when you pressed Run forecast / when the nightly job ran. Useful if you have multiple runs for the same day.",
                    )
                else:
                    st.session_state["history_show_run_at"] = False
                    st.caption("")
        else:
            st.session_state["history_all_runs"] = False
            st.session_state["history_show_run_at"] = False

        st.caption("Open Run Inspector to see full model reasons and settings snapshot.")

        raw = run_history_from_backend(show_all_runs=st.session_state["history_all_runs"], days=365)
        prepared = _prepare_history_df(
            raw,
            all_runs=st.session_state["history_all_runs"],
            show_run_at=st.session_state["history_show_run_at"],
        )

        if not prepared.empty:
            date_min = _to_py_date(prepared["Date"].min())
            date_max = _to_py_date(prepared["Date"].max())
            if date_min is None or date_max is None:
                date_min = dt.date.today()
                date_max = dt.date.today()
            f1, f2 = st.columns([1.2, 1.2])
            with f1:
                selected_date_range = st.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
            with f2:
                status_options = ["✅ OK", "⚠ Degraded", "❌ Error"]
                status_filter = st.multiselect("Status", options=status_options, default=status_options)

            run_type_filter: list[str] | None = None
            has_warnings = "All"
            model_filter = "All models"
            if history_debug_mode:
                f3, f4, f5 = st.columns([1.2, 1.2, 2.4])
                with f3:
                    run_types = sorted({str(v) for v in prepared["run_type"].dropna().tolist()})
                    run_type_filter = st.multiselect("Run type", options=run_types, default=run_types)
                with f4:
                    has_warnings = st.selectbox("Has warnings", ["All", "Yes", "No"], index=0)
                with f5:
                    all_models: set[str] = set()
                    for cell in prepared.get("models_raw", pd.Series(dtype=object)).dropna().tolist():
                        all_models.update([x.strip() for x in str(cell).split(",") if x.strip() and x.strip() != "—"])
                    model_options = sorted(all_models)
                    model_filter = st.selectbox("Model filter (optional)", ["All models", *model_options], index=0)

            filtered = prepared.copy()
            if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
                start_date, end_date = selected_date_range
                filtered = filtered[(filtered["Date"].dt.date >= start_date) & (filtered["Date"].dt.date <= end_date)]
            if status_filter:
                filtered = filtered[filtered["Status label"].isin(status_filter)]
            if history_debug_mode:
                if run_type_filter:
                    filtered = filtered[filtered["run_type"].isin(run_type_filter)]
                if has_warnings == "Yes":
                    filtered = filtered[filtered["warnings_count"] > 0]
                elif has_warnings == "No":
                    filtered = filtered[filtered["warnings_count"] == 0]
                if model_filter != "All models":
                    filtered = filtered[filtered["models_raw"].str.contains(model_filter, na=False)]
            filtered = filtered.reset_index(drop=True)
        else:
            filtered = prepared

        if filtered.empty:
            st.info("No history records yet. Run a forecast to create the first record.")
        else:
            latest_row = filtered.iloc[-1]
            latest_date = latest_row.get("Date")
            if pd.isna(latest_date):
                latest_date_text = "—"
            elif hasattr(latest_date, "strftime"):
                latest_date_text = latest_date.strftime("%Y-%m-%d")
            else:
                latest_date_text = str(latest_date)
            latest_cutoff_raw = pd.to_numeric(pd.Series([latest_row.get("cutoff_soc")]), errors="coerce").iloc[0]
            latest_cutoff_pct = (latest_cutoff_raw * 100.0) if pd.notna(latest_cutoff_raw) else None
            latest_warnings = int(pd.to_numeric(pd.Series([latest_row.get("warnings_count")]), errors="coerce").fillna(0).iloc[0])

            st.markdown("**Last run summary**")
            s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
            s1.metric("Target date", latest_date_text)
            s2.metric("Status", str(latest_row.get("Status label") or "—"))
            s3.metric("PV p50", f"{float(pd.to_numeric(pd.Series([latest_row.get('PV p50')]), errors='coerce').fillna(0).iloc[0]):.2f} kWh")
            s4.metric("Load (estimated)", f"{float(pd.to_numeric(pd.Series([latest_row.get('Load (estimated)')]), errors='coerce').fillna(0).iloc[0]):.2f} kWh")
            s5.metric("Allowed AC charge power (kW)", f"{float(pd.to_numeric(pd.Series([latest_row.get('Charge')]), errors='coerce').fillna(0).iloc[0]):.2f} kW")
            s6.metric("Cutoff SOC (%)", f"{latest_cutoff_pct:.1f}%" if latest_cutoff_pct is not None else "—")
            s7.metric("Warnings count", str(latest_warnings))

            display_df = filtered.copy()
            display_df["Date"] = display_df["Date"].astype(str)
            if "Run at" in display_df.columns:
                display_df["Run at"] = display_df["Run at"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
                if not st.session_state.get("history_show_run_at", False):
                    display_df = display_df.drop(columns=["Run at"])

            display_df["Allowed AC charge power"] = display_df["Charge"]
            display_df["Warnings badge"] = display_df["warnings_count"].apply(lambda n: f"⚠ {int(n)}" if int(n or 0) > 0 else "✅ 0")
            pv_p10_vals = pd.to_numeric(display_df["PV p10"], errors="coerce")
            pv_p90_vals = pd.to_numeric(display_df["PV p90"], errors="coerce")
            pv_range_width = (pv_p90_vals - pv_p10_vals).round(2)
            display_df["PV range width"] = pv_range_width.where(pv_p10_vals.notna() & pv_p90_vals.notna(), None)
            cutoff_soc_pct = pd.to_numeric(display_df.get("cutoff_soc", pd.Series([None] * len(display_df))), errors="coerce") * 100.0
            display_df["Cutoff SOC"] = cutoff_soc_pct.round(1)
            display_df["Run duration"] = pd.to_numeric(display_df["Duration (ms)"], errors="coerce").round(0)
            display_df["Run type"] = display_df["run_type"].fillna("manual")
            display_df["Run id"] = display_df["run_id"].fillna("")

            display_df = display_df.rename(columns={"Status label": "Status", "Models": "Models summary"})

            simple_columns = ["Date", "Status", "PV quality", "PV p50", "Load (estimated)", "Allowed AC charge power", "Warnings badge"]
            debug_columns = [
                "Date",
                "Status",
                "PV quality",
                "pv_quality_label",
                "pv_quality_score",
                "PV p50",
                "PV p10",
                "PV p90",
                "PV range width",
                "Load (estimated)",
                "Allowed AC charge power",
                "Warnings badge",
                "Models summary",
                "Cutoff SOC",
                "Run duration",
                "Run type",
                "Run id",
            ]
            active_columns = debug_columns if history_debug_mode else simple_columns
            keep_columns = [c for c in active_columns if c in display_df.columns]

            drop_cols = [c for c in display_df.columns if c not in keep_columns]
            display_df = display_df.drop(columns=[c for c in drop_cols if c in display_df.columns])

            start_date_for_filename: object = filtered["Date"].min()
            end_date_for_filename: object = filtered["Date"].max()
            if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
                start_date_for_filename, end_date_for_filename = selected_date_range
            date_range_part = (
                f"{_target_date_for_filename(start_date_for_filename)}"
                f"-{_target_date_for_filename(end_date_for_filename)}"
            )
            mode_part = _safe_filename_part(st.session_state.get("history_mode", "Simple"), "simple")
            csv_bytes = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export CSV",
                data=csv_bytes,
                file_name=f"history_{date_range_part}_{mode_part}.csv",
                mime="text/csv",
                key="history_log_export_csv",
            )

            history_column_config = build_column_config(
                display_df,
                {
                    "pv_quality_label": st.column_config.TextColumn("PV quality label"),
                    "pv_quality_score": st.column_config.NumberColumn("PV quality score", format="%.0f/100"),
                    "PV p50": st.column_config.NumberColumn(format="%.2f kWh"),
                    "PV p10": st.column_config.NumberColumn(format="%.2f kWh"),
                    "PV p90": st.column_config.NumberColumn(format="%.2f kWh"),
                    "PV range width": st.column_config.NumberColumn(format="%.2f kWh"),
                    "Load (estimated)": st.column_config.NumberColumn(format="%.2f kWh"),
                    "Allowed AC charge power": st.column_config.NumberColumn(format="%.2f kW"),
                    "Run duration": st.column_config.NumberColumn(format="%.0f ms"),
                    "Cutoff SOC": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
            selected_history_row = render_selectable_table(
                display_df,
                key="history_log_table",
                column_config=history_column_config,
            )
            if selected_history_row is not None and 0 <= selected_history_row < len(filtered):
                selected_row = filtered.iloc[selected_history_row]
                selected_run_id = str(selected_row.get("run_id") or "").strip()
                if selected_run_id:
                    st.session_state["history_inspector_selected_run_id"] = selected_run_id
            if history_debug_mode and not filtered.empty:
                run_id_options = [str(v) for v in filtered["run_id"].dropna().tolist() if str(v).strip()]
                if run_id_options:
                    selected_copy_run_id = st.selectbox("Run id", options=run_id_options, key="history_copy_run_id")
                    st.code(selected_copy_run_id, language="text")
                    st.caption("Use the copy button in the code block to copy the selected run id.")
            _render_run_inspector(filtered)
            _render_compare_runs_block(filtered)


if hasattr(st, "fragment"):
    @st.fragment
    def render_history_fragment() -> None:
        _render_history_log_block()
else:
    def render_history_fragment() -> None:
        _render_history_log_block()


def normalize_detail_df_for_ui(df: pd.DataFrame, effective_cfg: dict) -> pd.DataFrame:
    out = df.copy()
    pv_cfg = effective_cfg.get("pv", {}) if isinstance(effective_cfg, dict) else {}
    east_panels = int(pv_cfg.get("array_east_panels", 0) or 0)
    south_panels = int(pv_cfg.get("array_south_panels", 1) or 1)
    panel_total = max(east_panels + south_panels, 1)
    east_ratio = east_panels / panel_total
    south_ratio = south_panels / panel_total

    if "pv_total_kwh" not in out.columns:
        if "pv_ac_limited_kwh" in out.columns:
            out["pv_total_kwh"] = out["pv_ac_limited_kwh"]
        else:
            out["pv_total_kwh"] = 0.0

    if "pv_total_unclipped_kwh" not in out.columns:
        if "pv_dc_available_kwh" in out.columns:
            out["pv_total_unclipped_kwh"] = out["pv_dc_available_kwh"]
        else:
            out["pv_total_unclipped_kwh"] = out["pv_total_kwh"]

    if "pv_east_kwh" not in out.columns:
        out["pv_east_kwh"] = pd.to_numeric(out["pv_total_kwh"], errors="coerce").fillna(0.0) * east_ratio
    if "pv_south_kwh" not in out.columns:
        out["pv_south_kwh"] = pd.to_numeric(out["pv_total_kwh"], errors="coerce").fillna(0.0) * south_ratio

    out["pv_total_kwh"] = pd.to_numeric(out["pv_total_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["pv_total_unclipped_kwh"] = pd.to_numeric(out["pv_total_unclipped_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["pv_total_unclipped_kwh"] = out[["pv_total_unclipped_kwh", "pv_total_kwh"]].max(axis=1)
    out["pv_east_kwh"] = pd.to_numeric(out["pv_east_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["pv_south_kwh"] = pd.to_numeric(out["pv_south_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["pv_clipped_kwh"] = (out["pv_total_unclipped_kwh"] - out["pv_total_kwh"]).clip(lower=0.0)

    if "pv_surplus_kwh" not in out.columns and "surplus_kwh" in out.columns:
        out["pv_surplus_kwh"] = pd.to_numeric(out["surplus_kwh"], errors="coerce").fillna(0.0)
    if "pv_deficit_kwh" not in out.columns and "deficit_kwh" in out.columns:
        out["pv_deficit_kwh"] = pd.to_numeric(out["deficit_kwh"], errors="coerce").fillna(0.0)
    if "pv_surplus_kwh" not in out.columns:
        load_series = out.get("load_kwh")
        if load_series is None:
            load_series = pd.Series(0.0, index=out.index)
        load = pd.to_numeric(load_series, errors="coerce").fillna(0.0)
        out["pv_surplus_kwh"] = (out["pv_total_kwh"] - load).clip(lower=0.0)
    if "pv_deficit_kwh" not in out.columns:
        load_series = out.get("load_kwh")
        if load_series is None:
            load_series = pd.Series(0.0, index=out.index)
        load = pd.to_numeric(load_series, errors="coerce").fillna(0.0)
        out["pv_deficit_kwh"] = (load - out["pv_total_kwh"]).clip(lower=0.0)

    return out


def make_chart_pv_load(df: pd.DataFrame, soc: pd.Series, cutoff_soc: float, effective_cfg: dict) -> go.Figure:
    working = normalize_detail_df_for_ui(df.copy(), effective_cfg)
    has_range = ("pv_total_low_kwh" in working.columns) and ("pv_total_high_kwh" in working.columns)
    if "load_kwh" not in working.columns:
        working["load_kwh"] = 0.0
    if "pv_surplus_kwh" not in working.columns:
        working["pv_surplus_kwh"] = (working["pv_total_kwh"] - working["load_kwh"]).clip(lower=0.0)
    if "pv_deficit_kwh" not in working.columns:
        working["pv_deficit_kwh"] = (working["load_kwh"] - working["pv_total_kwh"]).clip(lower=0.0)
    cutoff_pct = float(cutoff_soc) * 100.0

    custom_data = working[[
        "pv_east_kwh",
        "pv_south_kwh",
        "pv_total_kwh",
        "load_kwh",
        "pv_surplus_kwh",
        "pv_deficit_kwh",
        "pv_total_unclipped_kwh",
        "pv_clipped_kwh",
    ]].to_numpy()
    hover = (
        "Hour: %{x|%H:%M}<br>"
        "PV East kWh (hour): %{customdata[0]:.2f}<br>"
        "PV South kWh (hour): %{customdata[1]:.2f}<br>"
        "PV total kWh (hour): %{customdata[2]:.2f}<br>"
        "Load (estimated) kWh (hour): %{customdata[3]:.2f}<br>"
        "Surplus/Deficit (hour): +%{customdata[4]:.2f} / -%{customdata[5]:.2f}<br>"
        "PV potential (unclipped): %{customdata[6]:.2f} kWh<br>"
        "PV delivered (after AC limit): %{customdata[2]:.2f} kWh<br>"
        "Inverter clipped (lost): %{customdata[7]:.2f} kWh"
        "<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=working.index,
            y=working["pv_east_kwh"],
            name="PV east",
            marker_color="#4cc9f0",
            opacity=0.8,
            customdata=custom_data,
            hovertemplate=hover,
        )
    )
    fig.add_trace(
        go.Bar(
            x=working.index,
            y=working["pv_south_kwh"],
            name="PV south",
            marker_color="#f8961e",
            opacity=0.8,
            customdata=custom_data,
            hovertemplate=hover,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=working.index,
            y=working["pv_total_kwh"],
            mode="lines",
            name="PV total (Typical)",
            line=dict(width=3, color="#90be6d"),
            customdata=custom_data,
            hovertemplate=hover,
        )
    )
    if has_range:
        fig.add_trace(
            go.Scatter(
                x=working.index,
                y=working["pv_total_low_kwh"],
                mode="lines",
                name="PV total (Low)",
                visible="legendonly",
                line=dict(width=2, dash="dot", color="#90be6d"),
                hovertemplate="Hour: %{x|%H:%M}<br>PV total (Low): %{y:.2f} kWh<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=working.index,
                y=working["pv_total_high_kwh"],
                mode="lines",
                name="PV total (High)",
                visible="legendonly",
                line=dict(width=2, dash="dot", color="#90be6d"),
                hovertemplate="Hour: %{x|%H:%M}<br>PV total (High): %{y:.2f} kWh<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=working.index,
            y=working["load_kwh"],
            mode="lines",
            name="Load (estimated)",
            line=dict(width=3, color="#e63946"),
            customdata=custom_data,
            hovertemplate=hover,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=working.index,
            y=working["pv_total_unclipped_kwh"],
            mode="lines",
            name="PV unclipped",
            line=dict(dash="dot", width=2, color="#adb5bd"),
            customdata=custom_data,
            hovertemplate=hover,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=soc.index,
            y=soc.values,
            mode="lines",
            name="Battery SOC",
            line=dict(width=2, dash="dash", color="#c77dff"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        template=PLOTLY_DARK,
        hovermode="x unified",
        barmode="relative",
        bargap=0.16,
        xaxis=dict(tickformat="%H:%M", showgrid=False),
        xaxis_title="Hour",
        yaxis_title="Energy (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=70),
        yaxis2=dict(title="SOC (%)", overlaying="y", side="right", range=[0, 100], showgrid=False),
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=cutoff_pct,
        y1=cutoff_pct,
        yref="y2",
        line=dict(dash="dash", width=1),
    )
    fig.add_annotation(
        x=1,
        xref="paper",
        y=cutoff_pct,
        yref="y2",
        text=f"Cutoff SOC {cutoff_pct:.0f}%",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
    )
    return fig


def make_chart_surplus(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["pv_surplus_kwh"], name="Surplus"))
    fig.add_trace(go.Bar(x=df.index, y=-df["pv_deficit_kwh"], name="Deficit"))
    fig.update_layout(
        template=PLOTLY_DARK,
        barmode="relative",
        xaxis=dict(tickformat="%H:%M", title="Hour", showgrid=False),
        yaxis=dict(title="Energy (kWh)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=70),
    )
    return fig


def make_chart_grid(flows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=flows.index, y=flows["grid_import_kwh"], name="Grid import"))
    fig.add_trace(go.Bar(x=flows.index, y=-flows["grid_export_kwh"], name="Grid export (negative)"))
    fig.add_trace(go.Bar(x=flows.index, y=-flows["curtailed_kwh"], name="Curtailed PV (negative)"))
    fig.update_layout(
        template=PLOTLY_DARK,
        barmode="relative",
        xaxis=dict(tickformat="%H:%M", title="Hour", showgrid=False),
        yaxis=dict(title="Energy (kWh)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=70),
    )
    return fig


def add_tariff_and_sun_markers(fig: go.Figure, tomorrow_date: dt.date, sunrise: pd.Timestamp, sunset: pd.Timestamp) -> None:
    _ = tomorrow_date

    # Plotly's add_vline(annotation_text=...) can fail on datetime axes with:
    # "unsupported operand type(s) for +: 'int' and 'datetime.datetime'".
    # Add the vertical line and annotation separately to avoid that path.
    fig.add_vline(x=sunrise, line_dash="dash", line_width=1, line_color="#ffd166")
    fig.add_annotation(x=sunrise, y=1.0, yref="paper", text="Sunrise", showarrow=False, yshift=8, font=dict(color="#ffd166"))
    fig.add_vline(x=sunset, line_dash="dash", line_width=1, line_color="#f4a261")
    fig.add_annotation(x=sunset, y=1.0, yref="paper", text="Sunset", showarrow=False, yshift=8, font=dict(color="#f4a261"))


st.set_page_config(page_title="PV Battery Planner", layout="wide")
inject_tooltip_css()
st.title("PV Battery Planner")

try:
    health_payload = api_get("/v1/health")
    backend_settings = api_get("/v1/settings")
    weather_models_catalog = api_get("/v1/weather/models").get("items", [])
    valid_model_ids = {m.get("id") for m in weather_models_catalog if isinstance(m.get("id"), str)}
except Exception as exc:
    st.error(
        f"Backend unavailable at {API_BASE_URL}. Start backend with: "
        "uvicorn backend_api:app --host 127.0.0.1 --port 8787. "
        f"Details: {exc}"
    )
    st.stop()

if "last_soc" not in st.session_state:
    st.session_state.last_soc = 45.0
if "last_kwh" not in st.session_state:
    st.session_state.last_kwh = 18.0

left, right = st.columns([1, 2])
with left:
    effective_cfg = backend_settings.get("config", core.DEFAULT_CONFIG)
    loc_cfg = effective_cfg["location"]
    apply_pending_location_state()
    apply_location_lookup_result(effective_cfg)

    loc_structured = loc_cfg.get("address_structured", {}) if isinstance(loc_cfg.get("address_structured"), dict) else {}
    if "loc_address_query_display" not in st.session_state:
        st.session_state["loc_address_query_display"] = str(loc_cfg.get("address_query", ""))
    if "loc_latitude" not in st.session_state:
        st.session_state["loc_latitude"] = _safe_float(loc_cfg.get("latitude"), core.LATITUDE)
    if "loc_longitude" not in st.session_state:
        st.session_state["loc_longitude"] = _safe_float(loc_cfg.get("longitude"), core.LONGITUDE)
    if "loc_timezone" not in st.session_state:
        st.session_state["loc_timezone"] = str(loc_cfg.get("timezone", core.TIMEZONE))
    if "loc_street" not in st.session_state:
        st.session_state["loc_street"] = str(loc_structured.get("street", ""))
    if "loc_house_number" not in st.session_state:
        st.session_state["loc_house_number"] = str(loc_structured.get("house_number", ""))
    if "loc_postal_code" not in st.session_state:
        st.session_state["loc_postal_code"] = str(loc_structured.get("postal_code", ""))
    if "loc_city" not in st.session_state:
        st.session_state["loc_city"] = str(loc_structured.get("city", ""))
    if "loc_country" not in st.session_state:
        st.session_state["loc_country"] = str(loc_structured.get("country", ""))
    with st.expander("Inputs", expanded=True):
        soc_percent = st.number_input(
            "Battery SOC at 22:00 (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.last_soc),
            step=1.0,
            format="%.0f",
            help=get_help("soc_percent"),
        )
        yesterday_kwh = st.number_input(
            "Yesterday total consumption (kWh)",
            min_value=0.1,
            value=float(st.session_state.last_kwh),
            step=0.1,
            format="%.1f",
            help=get_help("yesterday_kwh"),
        )
        if yesterday_kwh < 2.0 or yesterday_kwh > 60.0:
            st.error("Run forecast is blocked: Yesterday total consumption must be between 2.0 and 60.0 kWh. If yesterday was unusual, enter a typical day such as 12.0 kWh.")

    weather_models_box = st.container()

    with st.expander("Settings", expanded=False):
        st.markdown("#### Location")
        addr_col, status_col, btn_col = st.columns([6, 1, 2], vertical_alignment="center")
        with addr_col:
            st.text_input("Address query", key="loc_address_query_display", disabled=True, help=get_help("address_query"))
        with status_col:
            has_lookup_details = isinstance(st.session_state.get("loc_latitude"), (float, int)) and isinstance(
                st.session_state.get("loc_longitude"), (float, int)
            ) and bool(str(st.session_state.get("loc_timezone", "")).strip())
            if has_lookup_details:
                st.markdown(
                    f"<div title=\"Latitude: {float(st.session_state['loc_latitude']):.5f} | "
                    f"Longitude: {float(st.session_state['loc_longitude']):.5f} | "
                    f"Timezone: {str(st.session_state['loc_timezone'])}\" "
                    "style='font-size:1.4rem;line-height:2.4rem;text-align:center'>✅</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)
        with btn_col:
            if st.button("Lookup", type="primary", key="btn_open_lookup"):
                open_lookup(loc_cfg)
                st.rerun()

        if st.session_state.get("loc_lookup_open"):
            lookup_location_dialog()

        loc_col1, loc_col2 = st.columns(2)
        with loc_col1:
            cfg_latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                step=0.00001,
                format="%.5f",
                key="loc_latitude",
                help=get_help("latitude"),
            )
        with loc_col2:
            cfg_longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                step=0.00001,
                format="%.5f",
                key="loc_longitude",
                help=get_help("longitude"),
            )
        cfg_timezone = st.text_input("Timezone", key="loc_timezone", help=get_help("timezone"))

        st.markdown("#### Tariff settings")
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        tariff_cfg = effective_cfg.get("tariff", core.DEFAULT_CONFIG["tariff"])
        cfg_peak_price = float(tariff_cfg.get("peak_grid_price_eur_per_kwh", core.DEFAULT_CONFIG["tariff"]["peak_grid_price_eur_per_kwh"]))
        cfg_offpeak_price = float(tariff_cfg.get("offpeak_grid_price_eur_per_kwh", core.DEFAULT_CONFIG["tariff"]["offpeak_grid_price_eur_per_kwh"]))
        cfg_injection_price = float(tariff_cfg.get("injection_grid_price_eur_per_kwh", core.DEFAULT_CONFIG["tariff"]["injection_grid_price_eur_per_kwh"]))
        def render_tariff_section_title(title: str) -> None:
            st.markdown(f"##### {title}")

        render_tariff_section_title("Energy prices")
        c1, c2, c3 = st.columns(3)
        with c1:
            cfg_peak_price_input = st.number_input(
                "Peak grid price",
                min_value=0.0,
                step=0.001,
                format="%.4f",
                value=cfg_peak_price,
                key="tariff_peak_price",
                help=get_help("peak_price"),
            )
        with c2:
            cfg_offpeak_price_input = st.number_input(
                "Off-peak grid price",
                min_value=0.0,
                step=0.001,
                format="%.4f",
                value=cfg_offpeak_price,
                key="tariff_offpeak_price",
                help=get_help("peak_price"),
            )
        with c3:
            cfg_injection_price_input = st.number_input(
                "Export (injection) price",
                min_value=-1.0,
                step=0.001,
                format="%.4f",
                value=cfg_injection_price,
                key="tariff_injection_price",
                help="All-in export price (€/kWh). Enter your full export (injection) price. Use a negative value if export costs money.",
            )

        st.markdown("")
        render_tariff_section_title("Off-peak hours")

        tariff_source = tariff_cfg.get("offpeak_windows_by_dow", core.DEFAULT_CONFIG["tariff"]["offpeak_windows_by_dow"])
        tariff_by_day = core.parse_offpeak_windows_by_dow(tariff_source)
        default_tariff_by_day = core.parse_offpeak_windows_by_dow(core.DEFAULT_CONFIG["tariff"]["offpeak_windows_by_dow"])
        tariff_inputs: list[tuple[str, str]] = []
        for day_idx, day_name in enumerate(day_names):
            day_windows = tariff_by_day.get(day_idx) or default_tariff_by_day.get(day_idx, [("00:00", "24:00")])
            day_from, day_to = day_windows[0]
            cols = st.columns([1.0, 1.2, 1.2])
            cols[0].markdown(f"**{day_name[:3]}**")
            from_value = cols[1].text_input(
                f"From {day_name}",
                value=day_from,
                key=f"tariff_from_{day_idx}",
                label_visibility="collapsed",
            ).strip()
            to_value = cols[2].text_input(
                f"To {day_name}",
                value=day_to,
                key=f"tariff_to_{day_idx}",
                label_visibility="collapsed",
            ).strip()
            tariff_inputs.append((from_value, to_value))

        st.markdown("#### PV")
        cfg_pv = effective_cfg["pv"]

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            cfg_panel_wp = st.number_input("Panel power (Wp)", min_value=1, value=int(cfg_pv["panel_wp"]), step=1, help=get_help("panel_wp"))
        with row1_col2:
            cfg_performance_ratio = st.number_input(
                "Performance ratio",
                min_value=0.50,
                max_value=1.00,
                step=0.01,
                value=float(cfg_pv["performance_ratio"]),
                help=INPUT_TOOLTIPS["performance_ratio"],
                key="pv_pr",
            )

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            cfg_array_east_panels = st.number_input("East array panels", min_value=0, value=int(cfg_pv["array_east_panels"]), step=1, help=get_help("array_panels"))
        with row2_col2:
            cfg_tilt_east_deg = st.number_input(
                "Tilt East (deg)",
                min_value=0.0,
                max_value=90.0,
                value=float(cfg_pv["tilt_east_deg"]),
                step=1.0,
                format="%.0f",
            )
        with row2_col3:
            cfg_azimuth_east_deg = st.number_input(
                "Azimuth East (deg)",
                min_value=0.0,
                max_value=360.0,
                value=float(cfg_pv["azimuth_east_deg"]),
                step=1.0,
                format="%.0f",
            )

        row3_col1, row3_col2, row3_col3 = st.columns(3)
        with row3_col1:
            cfg_array_south_panels = st.number_input("South array panels", min_value=0, value=int(cfg_pv["array_south_panels"]), step=1, help=get_help("array_panels"))
        with row3_col2:
            cfg_tilt_south_deg = st.number_input(
                "Tilt South (deg)",
                min_value=0.0,
                max_value=90.0,
                value=float(cfg_pv["tilt_south_deg"]),
                step=1.0,
                format="%.0f",
            )
        with row3_col3:
            cfg_azimuth_south_deg = st.number_input(
                "Azimuth South (deg)",
                min_value=0.0,
                max_value=360.0,
                value=float(cfg_pv["azimuth_south_deg"]),
                step=1.0,
                format="%.0f",
            )

        cfg_inverter_ac_kw_limit = st.number_input("Inverter AC limit (kW)", min_value=0.1, value=float(cfg_pv["inverter_ac_kw_limit"]), step=0.1, help=get_help("inverter_ac_kw_limit"))

        apply_pv_reco = st.button(
            "Use recommended PV defaults",
            type="secondary",
            width="content",
            key="btn_pv_reco",
        )

        with st.expander("Advanced", expanded=False):
            with st.expander("Advanced PV modelling", expanded=False):
                row4_col1, row4_col2 = st.columns(2)
                with row4_col1:
                    cfg_pv_loss_model = st.selectbox(
                        "PV loss model",
                        options=["split", "combined"],
                        index=["split", "combined"].index(str(cfg_pv.get("pv_loss_model", "split")).strip().lower() if str(cfg_pv.get("pv_loss_model", "split")).strip().lower() in {"split", "combined"} else "split"),
                        help=INPUT_TOOLTIPS["pv_loss_model"],
                    )
                with row4_col2:
                    cfg_inverter_eff = st.number_input(
                        "Inverter efficiency",
                        min_value=0.50,
                        max_value=1.00,
                        value=float(cfg_pv["inverter_eff"]),
                        step=0.01,
                        disabled=(cfg_pv_loss_model == "combined"),
                        help=INPUT_TOOLTIPS["inverter_eff"],
                        key="pv_inverter_eff",
                    )
                if cfg_pv_loss_model == "combined":
                    st.caption("Inverter efficiency is not used in combined loss mode.")

                row5b_col1, row5b_col2 = st.columns(2)
                with row5b_col1:
                    inverter_ac_model_value = str(cfg_pv.get("inverter_ac_model", "linear")).strip().lower()
                    cfg_inverter_ac_model = st.selectbox(
                        "Inverter AC model",
                        options=["linear", "pvwatts"],
                        index=["linear", "pvwatts"].index(inverter_ac_model_value if inverter_ac_model_value in {"linear", "pvwatts"} else "linear"),
                        help=INPUT_TOOLTIPS["inverter_ac_model"],
                        key="pv_inverter_ac_model",
                    )
                with row5b_col2:
                    iam_model_value = str(cfg_pv.get("iam_model", "none")).strip().lower()
                    cfg_iam_model = st.selectbox(
                        "IAM model",
                        options=["none", "ashrae"],
                        index=["none", "ashrae"].index(iam_model_value if iam_model_value in {"none", "ashrae"} else "none"),
                        help=INPUT_TOOLTIPS["iam_model"],
                        key="pv_iam_model",
                    )

                row5c_col1, row5c_col2 = st.columns(2)
                with row5c_col1:
                    cfg_iam_ashrae_b = st.number_input(
                        "IAM ASHRAE b",
                        min_value=0.00,
                        max_value=0.50,
                        value=float(cfg_pv.get("iam_ashrae_b", 0.05)),
                        step=0.01,
                        disabled=(cfg_iam_model != "ashrae"),
                        help=INPUT_TOOLTIPS["iam_ashrae_b"],
                        key="pv_iam_b",
                    )
                with row5c_col2:
                    albedo_default = cfg_pv.get("albedo", None)
                    cfg_albedo_enabled = st.checkbox(
                        "Set custom albedo",
                        value=albedo_default is not None,
                        help=INPUT_TOOLTIPS["albedo_enabled"],
                        key="pv_albedo_enabled",
                    )
                    cfg_albedo = st.number_input(
                        "Albedo",
                        min_value=0.00,
                        max_value=1.00,
                        value=float(albedo_default if albedo_default is not None else 0.20),
                        step=0.01,
                        disabled=(not cfg_albedo_enabled),
                        help=INPUT_TOOLTIPS["albedo"],
                        key="pv_albedo",
                    )

                row6_col1, row6_col2, row6_col3 = st.columns(3)
                with row6_col1:
                    cfg_pv_calibration_factor = st.number_input(
                        "PV calibration factor (global)",
                        min_value=0.70,
                        max_value=1.30,
                        value=float(cfg_pv.get("pv_calibration_factor", 1.0)),
                        step=0.01,
                        format="%.2f",
                        help=INPUT_TOOLTIPS["pv_calibration_factor"],
                        key="pv_cal_global",
                    )
                with row6_col2:
                    cfg_pv_calibration_factor_east = st.number_input(
                        "PV calibration factor east (relative)",
                        min_value=0.70,
                        max_value=1.30,
                        value=float(cfg_pv.get("pv_calibration_factor_east", 1.0)),
                        step=0.01,
                        format="%.2f",
                        help=INPUT_TOOLTIPS["pv_calibration_factor_east"],
                        key="pv_cal_east",
                    )
                with row6_col3:
                    cfg_pv_calibration_factor_south = st.number_input(
                        "PV calibration factor south (relative)",
                        min_value=0.70,
                        max_value=1.30,
                        value=float(cfg_pv.get("pv_calibration_factor_south", 1.0)),
                        step=0.01,
                        format="%.2f",
                        help=INPUT_TOOLTIPS["pv_calibration_factor_south"],
                        key="pv_cal_south",
                    )

            buffer_percent = st.slider("Forecast safety buffer SOC (%)", 0.0, 10.0, 0.0, 0.5, help=get_help("buffer_percent"))
            user_max_ac_kw = st.number_input(
                "Max allowed AC charge power (kW)",
                min_value=0.0,
                max_value=10.0,
                value=float(backend_settings.get("max_ac_charge_power_kw_default", 5.0)),
                step=0.1,
                help=get_help("max_ac_user_cap"),
            )
            nightly_time_str = str(backend_settings.get("nightly_run_time", "22:00"))
            try:
                nightly_minutes = parse_hhmm(nightly_time_str)
            except ValueError:
                nightly_minutes = parse_hhmm("22:00")
                st.warning("Stored nightly run time was invalid, so 22:00 is shown instead.")

            nightly_hour, nightly_minute = divmod(nightly_minutes, 60)
            nightly_time_value = dt.time(hour=nightly_hour, minute=nightly_minute)

            nightly_run_time = st.time_input(
                "Nightly run time (HH:MM)",
                value=nightly_time_value,
                step=dt.timedelta(minutes=5),
                help="Pick scheduler time. It controls automatic run timing. Example: 22:00.",
            ).strftime("%H:%M")
            if st.button("Save nightly schedule settings"):
                try:
                    api_put(
                        "/v1/settings",
                        {
                            "config": effective_cfg,
                            "nightly_run_time": nightly_run_time,
                            "timezone": str(effective_cfg.get("location", {}).get("timezone", backend_settings.get("timezone", "Europe/Brussels"))),
                            "max_ac_charge_power_kw_default": float(user_max_ac_kw),
                        },
                    )
                    st.success("Saved nightly schedule settings.")
                except Exception as exc:
                    st.error(f"Could not save nightly settings: {exc}")

        st.markdown("#### Battery")
        bat_col1, bat_col2 = st.columns(2)
        with bat_col1:
            cfg_battery_kwh = st.number_input("Battery capacity (kWh)", min_value=0.0, value=float(effective_cfg["battery"]["battery_kwh"]), step=0.1, help=get_help("battery_kwh"))
            cfg_min_soc_percent = st.number_input("Min SOC (%)", min_value=0.0, max_value=100.0, value=float(effective_cfg["battery"]["min_soc_percent"]), step=0.5, help=get_help("min_soc"))
            cfg_max_cutoff_soc_percent = st.number_input("Max cutoff SOC (%)", min_value=0.0, max_value=100.0, value=float(effective_cfg["battery"]["max_cutoff_soc_percent"]), step=0.5, help=get_help("cutoff_soc"))
        with bat_col2:
            cfg_battery_max_charge_kw = st.number_input("Battery max charge (kW)", min_value=0.0, value=float(effective_cfg["battery"]["battery_max_charge_kw"]), step=0.1, help=get_help("battery_max_charge_kw"))
            cfg_battery_max_discharge_kw = st.number_input("Battery max discharge (kW)", min_value=0.0, value=float(effective_cfg["battery"]["battery_max_discharge_kw"]), step=0.1, help=get_help("battery_max_discharge_kw"))
            cfg_max_ac_charge_kw_hard_limit = st.number_input("Max AC charge kW hard limit", min_value=0.0, value=float(effective_cfg["battery"]["max_ac_charge_kw_hard_limit"]), step=0.1, help=get_help("max_ac_charge_kw_hard_limit"))

        cfg_load_profile = [float(v) for v in effective_cfg["load_profile"]["load_profile_24h"]]

        if st.session_state.get("_geo_success"):
            st.success(st.session_state["_geo_success"])
        if st.session_state.get("_geo_error"):
            st.error(st.session_state["_geo_error"])
        if apply_pv_reco:
            st.session_state["pv_pr"] = PV_RECO_PR
            st.session_state["pv_inverter_eff"] = PV_RECO_INVERTER_EFF
            st.session_state["pv_inverter_ac_model"] = PV_RECO_INVERTER_AC_MODEL
            st.session_state["pv_iam_model"] = PV_RECO_IAM_MODEL
            st.session_state["pv_iam_b"] = PV_RECO_IAM_B
            st.session_state["pv_albedo_enabled"] = False
            st.session_state["pv_albedo"] = PV_RECO_ALBEDO
            st.session_state["pv_cal_global"] = PV_RECO_CAL_GLOBAL
            st.session_state["pv_cal_east"] = PV_RECO_CAL_EAST
            st.session_state["pv_cal_south"] = PV_RECO_CAL_SOUTH
            st.session_state["_settings_flash"] = "Applied recommended PV defaults (not saved yet). Click Save settings to persist."
            st.rerun()

        st.session_state["_cfg_ui_snapshot"] = {
            "day_names": day_names,
            "tariff_inputs": tariff_inputs,
            "tariff_by_day": tariff_by_day,
            "cfg_latitude": cfg_latitude,
            "cfg_longitude": cfg_longitude,
            "cfg_peak_price_input": cfg_peak_price_input,
            "cfg_offpeak_price_input": cfg_offpeak_price_input,
            "cfg_injection_price_input": cfg_injection_price_input,
            "cfg_panel_wp": cfg_panel_wp,
            "cfg_array_south_panels": cfg_array_south_panels,
            "cfg_array_east_panels": cfg_array_east_panels,
            "cfg_tilt_east_deg": cfg_tilt_east_deg,
            "cfg_tilt_south_deg": cfg_tilt_south_deg,
            "cfg_azimuth_east_deg": cfg_azimuth_east_deg,
            "cfg_azimuth_south_deg": cfg_azimuth_south_deg,
            "cfg_performance_ratio": cfg_performance_ratio,
            "cfg_inverter_eff": cfg_inverter_eff,
            "cfg_pv_loss_model": cfg_pv_loss_model,
            "cfg_iam_model": cfg_iam_model,
            "cfg_iam_ashrae_b": cfg_iam_ashrae_b,
            "cfg_albedo_enabled": cfg_albedo_enabled,
            "cfg_albedo": cfg_albedo,
            "cfg_inverter_ac_model": cfg_inverter_ac_model,
            "cfg_pv_calibration_factor": cfg_pv_calibration_factor,
            "cfg_pv_calibration_factor_east": cfg_pv_calibration_factor_east,
            "cfg_pv_calibration_factor_south": cfg_pv_calibration_factor_south,
            "cfg_inverter_ac_kw_limit": cfg_inverter_ac_kw_limit,
            "cfg_battery_kwh": cfg_battery_kwh,
            "cfg_min_soc_percent": cfg_min_soc_percent,
            "cfg_max_cutoff_soc_percent": cfg_max_cutoff_soc_percent,
            "cfg_battery_max_charge_kw": cfg_battery_max_charge_kw,
            "cfg_battery_max_discharge_kw": cfg_battery_max_discharge_kw,
            "cfg_max_ac_charge_kw_hard_limit": cfg_max_ac_charge_kw_hard_limit,
            "cfg_load_profile": cfg_load_profile,
            "saved_sat": bool((effective_cfg.get("weather", {}) if isinstance(effective_cfg, dict) else {}).get("use_satellite_nowcast_0_6h", False)),
        }
        current_settings_payload, settings_error = build_settings_payload(effective_cfg, valid_model_ids)
        saved_settings_payload = normalize_effective_cfg_to_payload(effective_cfg, valid_model_ids)
        settings_valid = settings_error is None and current_settings_payload is not None
        if settings_valid:
            current_payload_hash = json.dumps(current_settings_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            saved_payload_hash = json.dumps(saved_settings_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            settings_dirty = current_payload_hash != saved_payload_hash
        else:
            settings_dirty = True

    wm_latitude = float(st.session_state.get("loc_latitude", core.LATITUDE))
    wm_longitude = float(st.session_state.get("loc_longitude", core.LONGITUDE))
    wm_timezone = str(st.session_state.get("loc_timezone", core.TIMEZONE))

    model_options = {m.get("id"): m for m in weather_models_catalog if isinstance(m.get("id"), str)}
    available_ids = set(model_options.keys())

    saved = effective_cfg.get("weather_models_selected")
    if isinstance(saved, list):
        saved_set = {str(x) for x in saved if isinstance(x, str)}
    else:
        saved_set = set()

    initial_selected = (saved_set & available_ids) if saved_set else set()
    if not initial_selected:
        initial_selected = (WEATHER_MODEL_DEFAULT & available_ids) or available_ids.copy()

    current_mode = str(effective_cfg.get("forecast_mode", "auto")).strip().lower()
    mode_label_default = "Expert" if current_mode == "expert" else "Auto (System picks the best models)"

    with weather_models_box:
        with st.expander("Weather models", expanded=True):
            mode_col_left, mode_col_right = st.columns([2.8, 2.2], vertical_alignment="center")
            with mode_col_left:
                st.markdown(f"Forecast mode <span class='info-tooltip' title='{_esc(get_help('forecast_mode'))}'>ⓘ</span>", unsafe_allow_html=True)
            with mode_col_right:
                forecast_mode_label = st.selectbox(
                    "Forecast mode",
                    options=["Auto (System picks the best models)", "Expert"],
                    index=0 if mode_label_default != "Expert" else 1,
                    key="forecast_mode_select",
                    label_visibility="collapsed",
                )
            forecast_mode = FORECAST_MODE_OPTIONS.get(forecast_mode_label, "auto")
            weather_cfg = effective_cfg.get("weather", {}) if isinstance(effective_cfg, dict) else {}
            saved_sat = bool(weather_cfg.get("use_satellite_nowcast_0_6h", False))

            auto_selected = set(auto_select_models_for_location(wm_latitude, wm_longitude, requested_days=1)) & available_ids
            if not auto_selected:
                auto_selected = (WEATHER_MODEL_DEFAULT & available_ids) or available_ids.copy()
            used_auto = set(st.session_state.get("last_weather_ensemble_models_used", []) or [])

            if forecast_mode == "auto":
                st.markdown("**System will try:** " + " ".join([f"`{(model_options.get(mid, {}).get('label', mid)).split()[0]}`" for mid in WEATHER_MODEL_ORDER if mid in auto_selected]))
                used_line = " ".join([f"`{mid}`" for mid in used_auto]) if used_auto else "(no previous run yet)"
                st.markdown(f"**Used in last run:** {used_line}")
                _ = render_weather_models(
                    weather_models_catalog,
                    auto_selected,
                    widget_key_prefix="wm_auto",
                    disabled=True,
                    used_models=used_auto,
                    auto_locked_models=auto_selected,
                    show_capability_badges=True,
                )
                selected_models = []
                sat_nowcast_for_run = should_use_satellite_nowcast_auto(
                    latitude=wm_latitude,
                    longitude=wm_longitude,
                    timezone_name=wm_timezone,
                    requested_days=1,
                )
                st.checkbox(
                    "Use satellite nowcast radiation (0-6)",
                    value=bool(sat_nowcast_for_run),
                    disabled=True,
                    key="use_sat_nowcast_auto",
                    help=get_help("sat_nowcast"),
                )
            else:
                selected_models = render_weather_models(
                    weather_models_catalog,
                    initial_selected,
                    widget_key_prefix="wm",
                    show_capability_badges=True,
                )
                sat_nowcast_ui = st.checkbox(
                    "Use satellite nowcast radiation (0-6)",
                    value=saved_sat,
                    key="use_sat_nowcast_expert",
                    help=get_help("sat_nowcast"),
                )
                sat_nowcast_for_run = bool(sat_nowcast_ui)

    readiness_issues = validate_sidebar_readiness(
        st.session_state.get("_cfg_ui_snapshot", {}),
        yesterday_kwh=float(yesterday_kwh),
        forecast_mode=forecast_mode,
        selected_models=selected_models,
    )
    settings_valid = not any(readiness_issues[k] for k in ["Location", "Tariffs", "PV", "Battery"])
    inputs_valid = not readiness_issues["Inputs"]
    weather_valid = not readiness_issues["Weather"]
    run_allowed = settings_valid and inputs_valid and weather_valid

    summary_parts = []
    for key in ["Inputs", "Location", "Tariffs", "PV", "Battery", "Weather"]:
        if readiness_issues[key]:
            summary_parts.append(f"⚠ {key}: {readiness_issues[key][0]}")
        else:
            summary_parts.append(f"✅ {key}")
    st.caption(" | ".join(summary_parts))

    ensemble_method = "weighted"
    spacer, col_reset, col_save, col_run = st.columns([4.2, 2.6, 2.2, 2.2])
    with col_reset:
        reset_clicked = st.button(
            "Reset all settings",
            type="secondary",
            key="btn_reset_defaults",
            width="stretch",
        )
    with col_save:
        save_clicked = st.button(
            "Save settings",
            type="secondary",
            disabled=(not settings_valid) or (not settings_dirty),
            key="btn_save_settings_top",
            width="stretch",
        )
    with col_run:
        run_clicked = st.button(
            "Run forecast",
            type="primary",
            disabled=(not run_allowed),
            key="btn_run_forecast",
            width="stretch",
        )

    if reset_clicked:
        st.session_state["confirm_reset_repo_defaults_open"] = True

    if st.session_state.get("confirm_reset_repo_defaults_open"):
        with st.container(border=True):
            st.warning("Resetting will restore all settings to their default values.")
            st.caption("This action cannot be undone.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                confirm_reset = st.button(
                    "Yes, reset now",
                    type="primary",
                    key="btn_confirm_reset_repo_defaults",
                    width="stretch",
                )
            with cancel_col:
                cancel_reset = st.button(
                    "Cancel",
                    key="btn_cancel_reset_repo_defaults",
                    width="stretch",
                )

            if confirm_reset:
                try:
                    updated = api_post("/v1/settings/reset_to_repo_defaults", {})
                    st.cache_data.clear()
                    st.session_state["_pending_location_state"] = updated["config"]["location"]
                    st.session_state["_settings_flash"] = "Reset to defaults."
                    st.session_state["confirm_reset_repo_defaults_open"] = False
                    for mid in WEATHER_MODEL_ORDER:
                        st.session_state.pop(f"wm_{mid}", None)
                    for k in [
                        "pv_pr",
                        "pv_inverter_eff",
                        "pv_inverter_ac_model",
                        "pv_iam_model",
                        "pv_iam_b",
                        "pv_albedo_enabled",
                        "pv_albedo",
                        "pv_cal_global",
                        "pv_cal_east",
                        "pv_cal_south",
                    ]:
                        st.session_state.pop(k, None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not reset settings: {exc}")
            if cancel_reset:
                st.session_state["confirm_reset_repo_defaults_open"] = False
                st.rerun()

    flash = st.session_state.pop("_settings_flash", None)
    if flash:
        st.success(flash)

    if save_clicked:
        if not settings_valid:
            st.error(settings_error or "Could not save settings.")
        else:
            save_settings_payload(current_settings_payload)

if run_clicked:
    if settings_dirty:
        if not settings_valid:
            st.error(settings_error or "Could not save settings.")
            st.stop()
        ok = save_settings_payload(current_settings_payload, rerun=False)
        if not ok:
            st.stop()
    st.session_state.last_soc = soc_percent
    st.session_state.last_kwh = yesterday_kwh
    try:
        with st.spinner("Calling backend and loading results..."):
            api_put(
                "/v1/inputs/last",
                {
                    "soc_at_22_percent": float(soc_percent),
                    "yesterday_consumption_kwh": float(yesterday_kwh),
                },
            )
            run_response = api_post(
                "/v1/run/now",
                {
                    "buffer_percent": float(buffer_percent),
                    "user_max_ac_kw": float(user_max_ac_kw),
                    "weather_models": selected_models if forecast_mode == "expert" else None,
                    "forecast_mode": forecast_mode,
                    "use_satellite_nowcast_0_6h": bool(sat_nowcast_for_run),
                    "ensemble_method": ensemble_method,
                    "pv_uncertainty": True,
                },
            )
            result = run_response["result"]
            dbg = result.get("weather_ensemble")
            st.session_state["last_weather_ensemble_debug"] = dbg if isinstance(dbg, dict) else {}
            st.session_state["last_weather_ensemble_debug_at"] = dt.datetime.now(dt.UTC).isoformat()
            models_used = (
                result.get("tomorrow_models_used")
                or result.get("weather_ensemble", {}).get("selected_models")
                or result.get("weather_ensemble", {}).get("models_used")
                or []
            )
            st.session_state["last_forecast_mode"] = forecast_mode
            if forecast_mode == "auto":
                st.session_state["last_weather_ensemble_models_used"] = list(models_used)
            tomorrow = dt.date.fromisoformat(result["target_date"])
            weather_df = df_from_split(result["weather"])
            pv = df_from_split(result["pv"])
            detail_df = df_from_split(result["detail"])
            pv = normalize_detail_df_for_ui(pv, effective_cfg)
            detail_df = normalize_detail_df_for_ui(detail_df, effective_cfg)
            flows_df = df_from_split(result["flows"])
            soc_series = series_from_split(result["soc"])
            sunrise = pd.Timestamp(result["sunrise"])
            sunset = pd.Timestamp(result["sunset"])
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            pv_quality = result.get("pv_quality", {})
            cutoff_soc = _safe_float(metrics.get("cutoff_soc"), 0.0)
            charge_kw = _safe_float(metrics.get("charge_kw"), 0.0)
            charge_note = str(metrics.get("charge_note", ""))
            grid_import = _safe_float(metrics.get("grid_import"), 0.0)
            grid_export = _safe_float(metrics.get("grid_export"), 0.0)
            weather_ensemble = result.get("weather_ensemble", {}) if isinstance(result.get("weather_ensemble"), dict) else {}
            pv_week_ahead = result.get("pv_week_ahead") if isinstance(result.get("pv_week_ahead"), list) else []
            tomorrow_weather_code = None
            tomorrow_source_label = None
            tomorrow_source_days = None

            tomorrow_weather_code = result.get("tomorrow_weather_code")
            tomorrow_source_label = result.get("tomorrow_weather_code_source_model_label")
            tomorrow_source_days = result.get("tomorrow_weather_code_source_max_days")
            pv_low, pv_high = resolve_tomorrow_pv_low_high_kwh(
                result,
                weather_ensemble,
                tomorrow_p50_kwh=(pv_quality.get("pv_total_kwh") if isinstance(pv_quality, dict) else None),
            )

            weather_primary_model_id = result.get("weather_primary_model_id")
            weather_ensemble_table_payload = result.get("weather_ensemble_table")
            weather_by_model_payload = result.get("weather_by_model")
            ensemble_weather_df = (
                df_from_split(weather_ensemble_table_payload)
                if isinstance(weather_ensemble_table_payload, dict)
                else None
            )
            per_model_weather_dfs = (
                {
                    model_id: df_from_split(payload)
                    for model_id, payload in weather_by_model_payload.items()
                    if isinstance(payload, dict)
                }
                if isinstance(weather_by_model_payload, dict)
                else {}
            )

        with right:
            top_left, top_right = st.columns([4, 3], gap="large")
            with top_left:
                render_key_charging_widget(
                    st.container(),
                    allowed_charge_kw=float(charge_kw),
                    cutoff_soc_pct=float(cutoff_soc * 100.0),
                )
            with top_right:
                render_pv_quality_widget(
                    top_right,
                    pv,
                    pv_quality,
                    tomorrow,
                    pv_tomorrow_low_kwh=pv_low,
                    pv_tomorrow_high_kwh=pv_high,
                    tomorrow_weather_code=tomorrow_weather_code,
                    tomorrow_source_label=tomorrow_source_label,
                    tomorrow_source_days=tomorrow_source_days,
                )

            pv_week_ahead_display = (pv_week_ahead or [])[:6]
            render_pv_week_ahead_widget(pv_week_ahead_display)

            if charge_note.startswith("Warning"):
                st.warning(charge_note)

            st.markdown("### Forecast summary")
            c1, c2, c3, c4 = st.columns(4)

            pv_p50 = resolve_forecast_summary_pv_kwh(
                pv_quality,
                pv_week_ahead,
                pv,
                result,
                metrics,
                weather_ensemble,
            )
            metric_with_help(c1, "Forecast total PV (kWh)", f"{pv_p50:.2f}" if pv_p50 is not None else "—")
            if APP_DEBUG:
                if pv_low is not None and pv_high is not None:
                    st.caption(f"DEBUG Tomorrow PV low/high from usable models: {pv_low:.2f}/{pv_high:.2f} kWh")
                else:
                    st.caption("DEBUG Tomorrow PV low/high unavailable (<2 usable models)")
            load_total_value = None
            load_total_source = "missing"
            if "load_kwh" in pv.columns:
                load_total_value = float(pd.to_numeric(pv["load_kwh"], errors="coerce").sum(min_count=1))
                load_total_source = "pv.load_kwh"
            elif metrics.get("cons_forecast_kwh") is not None:
                load_total_value = float(_safe_float(metrics.get("cons_forecast_kwh"), 0.0))
                load_total_source = "metrics.cons_forecast_kwh"

            metric_with_help(
                c2,
                "Forecast total load (kWh)",
                f"{load_total_value:.2f}" if load_total_value is not None and not pd.isna(load_total_value) else "—",
            )
            if APP_DEBUG and load_total_source != "pv.load_kwh":
                c2.caption(
                    "DEBUG Forecast total load fallback: "
                    f"source={load_total_source}; pv_has_load_kwh={'load_kwh' in pv.columns}"
                )
            metric_with_help(c3, "Estimated grid import (expensive h)", f"{grid_import:.2f}")
            metric_with_help(c4, "Estimated export/curtailment (kWh)", f"{(grid_export + detail_df['curtailed_kwh'].sum() if not detail_df.empty else 0.0):.2f}")
            tooltip_heading("PV production vs Load (estimated) (hourly)", CHART_TOOLTIPS["PV production vs Load (estimated) (hourly)"])
            pv_load_fig = make_chart_pv_load(pv, soc_series, cutoff_soc, effective_cfg)
            add_tariff_and_sun_markers(pv_load_fig, tomorrow, sunrise, sunset)
            st.plotly_chart(pv_load_fig, use_container_width=True)

            chart_left, chart_right = st.columns(2, gap="large")
            with chart_left:
                tooltip_heading("Surplus vs Deficit (hourly)", CHART_TOOLTIPS["Surplus vs Deficit (hourly)"])
                surplus_fig = make_chart_surplus(pv)
                add_tariff_and_sun_markers(surplus_fig, tomorrow, sunrise, sunset)
                st.plotly_chart(surplus_fig, use_container_width=True)

            with chart_right:
                tooltip_heading("Grid import/export + curtailment", CHART_TOOLTIPS["Grid import/export + curtailment"])
                grid_fig = make_chart_grid(flows_df)
                add_tariff_and_sun_markers(grid_fig, tomorrow, sunrise, sunset)
                st.plotly_chart(grid_fig, use_container_width=True)

            run_inspector_debug_mode = st.session_state.get("history_mode", "Simple") == "Debug"
            if run_inspector_debug_mode:
                tooltip_heading("Weather inputs used", TABLE_TOOLTIPS["Weather inputs used"])
                with st.expander("Weather inputs used", expanded=False):
                    labels_by_id = {m.get("id"): m.get("label", m.get("id")) for m in weather_models_catalog}
                    selected_ids = weather_ensemble.get("selected_models", []) if isinstance(weather_ensemble.get("selected_models"), list) else []
                    selected_labels = [str(labels_by_id.get(mid, mid)) for mid in selected_ids]
                    st.write(
                        f"PV forecast built from {len(selected_labels)} models: {', '.join(selected_labels)} "
                        f"(method: {weather_ensemble.get('ensemble_method', 'weighted')})"
                    )

                    model_diag = summarize_model_diagnostics(weather_ensemble)
                    st.caption(
                        f"Models selected: {model_diag['selected']} · Models OK: {model_diag['ok']} · "
                        f"Models failed: {model_diag['failed']}"
                    )
                    if model_diag["failed_models"]:
                        st.caption(f"Failed models: {', '.join(model_diag['failed_models'])}")
                    st.caption(
                        "Derived irradiance used: "
                        + (f"Yes ({', '.join(model_diag['derived_models'])})" if model_diag["derived_models"] else "No")
                    )
                    st.caption(
                        "Missing important vars: "
                        + (f"Yes ({'; '.join(model_diag['missing_important'])})" if model_diag["missing_important"] else "No")
                    )
                    quality_factors = weather_ensemble.get("quality_weight_factors_by_model", {}) if isinstance(weather_ensemble.get("quality_weight_factors_by_model"), dict) else {}
                    if quality_factors:
                        st.caption("Quality weight factors by model (weighted ensemble)")
                        st.json(quality_factors, expanded=False)
                    derived_wc = weather_ensemble.get("derived_weather_code_by_model", {}) if isinstance(weather_ensemble.get("derived_weather_code_by_model"), dict) else {}
                    if derived_wc:
                        st.caption("Derived weather_code fallback by model")
                        st.json(derived_wc, expanded=False)
                    sat_used = weather_ensemble.get("satellite_nowcast_used")
                    if sat_used is not None:
                        st.caption(
                            "Satellite nowcast 0–6h: "
                            f"used={bool(sat_used)} · hours={int(weather_ensemble.get('satellite_nowcast_hours') or 0)} · "
                            f"reason={weather_ensemble.get('satellite_nowcast_reason')}"
                        )

                    weather_units_help_map = {
                        "temperature_2m": {"label": "Temp (°C)", "help": "Air temperature at 2m above ground.", "format": "%.1f"},
                        "wind_speed_10m": {"label": "Wind (m/s)", "help": "Wind speed at 10m.", "format": "%.1f"},
                        "cloud_cover": {"label": "Cloud cover (%)", "help": "Fraction of sky covered by clouds.", "format": "%d"},
                        "shortwave_radiation": {"label": "GHI (W/m²)", "help": "Global horizontal irradiance used by PV model.", "format": "%.0f"},
                        "ghi": {"label": "GHI (W/m²)", "help": "Global horizontal irradiance used by PV model.", "format": "%.0f"},
                        "dni": {"label": "DNI (W/m²)", "help": "Direct normal irradiance on surface normal to sun.", "format": "%.0f"},
                        "dhi": {"label": "DHI (W/m²)", "help": "Diffuse horizontal irradiance.", "format": "%.0f"},
                        "direct_normal_irradiance": {"label": "DNI (W/m²)", "help": "Direct normal irradiance.", "format": "%.0f"},
                        "diffuse_radiation": {"label": "DHI (W/m²)", "help": "Diffuse irradiance component.", "format": "%.0f"},
                        "shortwave_radiation_min": {"label": "GHI min (W/m²)", "help": "Minimum model irradiance among ensemble.", "format": "%.0f"},
                        "shortwave_radiation_max": {"label": "GHI max (W/m²)", "help": "Maximum model irradiance among ensemble.", "format": "%.0f"},
                    }

                    weather_source = ensemble_weather_df if isinstance(ensemble_weather_df, pd.DataFrame) and not ensemble_weather_df.empty else weather_df
                    if isinstance(weather_source, pd.DataFrame) and not weather_source.empty:
                        weather_display = weather_source.copy()
                        weather_display.insert(0, "hour", format_hour_from_index(weather_display.index, "%H:00").values)
                        weather_display = weather_display.head(24).reset_index(drop=True)
                        weather_preset = st.selectbox("Preset", options=["Core", "PV-relevant", "Full"], index=0, key="run_weather_preset")
                        weather_cols = get_preset_columns(tuple(str(c) for c in weather_display.columns), weather_preset, "weather")
                        weather_visible = weather_display[weather_cols]
                        weather_cfg = make_column_config(weather_visible, weather_units_help_map)
                        render_modern_table(weather_visible, weather_cfg)
                        weather_csv = weather_visible.to_csv(index=False)
                        weather_json = json.dumps({
                            "metadata": {
                                "preset": weather_preset,
                                "run_id": str(result.get("run_id", "")),
                                "target_date": str(result.get("target_date", "")),
                            },
                            "table": weather_display.to_dict(orient="records"),
                        }, ensure_ascii=False, indent=2)
                        weather_name = f"weather_inputs_{_target_date_for_filename(result.get('target_date'))}_{_safe_filename_part(result.get('run_id'),'run')}_{weather_preset.lower().replace(' ','_')}"
                        d1, d2 = st.columns(2)
                        d1.download_button("Download CSV", data=weather_csv, file_name=f"{weather_name}.csv", mime="text/csv", key="weather_csv_download")
                        d2.download_button("Download JSON", data=weather_json, file_name=f"{weather_name}.json", mime="application/json", key="weather_json_download")
                    else:
                        st.info("No data available.")

                combined = pv.join(flows_df[["soc_end_pct", "grid_import_kwh", "grid_export_kwh", "curtailed_kwh"]], how="left")
                if "pv_curtailed_kwh" not in combined.columns and "curtailed_kwh" in combined.columns:
                    combined["pv_curtailed_kwh"] = pd.to_numeric(combined["curtailed_kwh"], errors="coerce").fillna(0.0)
                combined = compute_residual_kwh(combined)
                combined_display = combined.copy()
                combined_display.insert(0, "hour", format_hour_from_index(combined_display.index, "%H:%M").values)
                combined_display = combined_display.reset_index(drop=True)
                tooltip_heading("Hourly planning output", TABLE_TOOLTIPS["Hourly planning output"])
                with st.expander("Hourly planning output", expanded=False):
                    planning_preset = st.selectbox("Preset", options=["Core", "Energy balance", "Full"], index=0, key="run_planning_preset")
                    planning_cols = get_preset_columns(tuple(str(c) for c in combined_display.columns), planning_preset, "planning")
                    planning_visible = combined_display[planning_cols]
                    planning_units_help_map = {
                        "pv_total_kwh": {"label": "PV total (kWh)", "help": "Total PV output available to planner.", "format": "%.2f"},
                        "pv_clipped_kwh": {"label": "PV clipped (kWh)", "help": "PV lost to clipping limits.", "format": "%.2f"},
                        "pv_ac_limited_kwh": {"label": "PV AC-limited (kWh)", "help": "PV output after AC-side limiting.", "format": "%.2f"},
                        "grid_import_kwh": {"label": "Grid import (kWh)", "help": "Energy imported from grid.", "format": "%.2f"},
                        "grid_export_kwh": {"label": "Grid export (kWh)", "help": "Energy exported to grid.", "format": "%.2f"},
                        "batt_charge_kwh": {"label": "Battery charge (kWh)", "help": "Energy sent into battery.", "format": "%.2f"},
                        "batt_discharge_kwh": {"label": "Battery discharge (kWh)", "help": "Energy discharged from battery.", "format": "%.2f"},
                        "soc_end_pct": {"label": "SOC end (%)", "help": "Battery SOC at end of hour.", "format": "%.1f"},
                        "residual_kwh": {"label": "Energy balance residual (kWh)", "help": "(PV + import + batt discharge) - (load + batt charge + export + curtailed).", "format": "%.3f"},
                        "charge_kw": {"label": "Charge (kW)", "help": "Charge power setting used for planning.", "format": "%.2f"},
                        "cutoff_soc_pct": {"label": "Cutoff SOC (%)", "help": "Configured AC charge cutoff SOC.", "format": "%.1f"},
                        "load_kwh": {"label": "Load (estimated) (kWh)", "help": "Estimated household demand based on yesterday total + profile; real usage can differ (EV, heat pump, weekend effects).", "format": "%.2f"},
                    }
                    residual_series = pd.to_numeric(combined_display.get("residual_kwh"), errors="coerce") if "residual_kwh" in combined_display.columns else pd.Series(dtype=float)
                    residual_abs = residual_series.abs()
                    max_res = float(residual_abs.max()) if not residual_abs.empty else 0.0
                    mean_res = float(residual_abs.mean()) if not residual_abs.empty else 0.0
                    st.metric("Max |residual| (kWh)", f"{max_res:.3f}")
                    st.caption(f"Mean |residual| (kWh): {mean_res:.3f}")
                    if max_res > 0.05:
                        st.warning("Energy balance residual is higher than expected; check inputs or rounding.")
                    if not residual_series.empty:
                        worst = residual_series.reindex(combined.index).dropna().abs().sort_values(ascending=False).head(3)
                        if not worst.empty:
                            worst_rows = []
                            for ts in worst.index:
                                worst_rows.append({"timestamp": str(ts), "residual_kwh": float(residual_series.loc[ts])})
                            st.caption("Top 3 residual hours")
                            st.dataframe(pd.DataFrame(worst_rows), use_container_width=True, hide_index=True)
                    render_modern_table(planning_visible, make_column_config(planning_visible, planning_units_help_map))
                    planning_csv = planning_visible.to_csv(index=False)
                    planning_json = json.dumps({
                        "metadata": {
                            "preset": planning_preset,
                            "run_id": str(result.get("run_id", "")),
                            "target_date": str(result.get("target_date", "")),
                        },
                        "table": combined_display.to_dict(orient="records"),
                    }, ensure_ascii=False, indent=2)
                    planning_name = f"hourly_planning_{_target_date_for_filename(result.get('target_date'))}_{_safe_filename_part(result.get('run_id'),'run')}_{planning_preset.lower().replace(' ','_')}"
                    p1, p2 = st.columns(2)
                    p1.download_button("Download CSV", data=planning_csv, file_name=f"{planning_name}.csv", mime="text/csv", key="planning_csv_download")
                    p2.download_button("Download JSON", data=planning_json, file_name=f"{planning_name}.json", mime="application/json", key="planning_json_download")
            render_history_fragment()

            for warning in result.get("warnings", []):
                st.warning(f"Nightly context warning: {warning}")
    except ImportError as exc:
        st.error(f"Missing dependency: {exc}. Install with: python -m pip install -r requirements.txt")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.error("Backend unreachable. Is backend running?")
    except RuntimeError as exc:
        st.error(str(exc))
    except requests.RequestException as exc:
        st.error(f"Backend API call failed: {exc}")
    except core.ExternalServiceError as exc:
        st.error(f"Weather fetch failed: {exc.category}")
        st.info(exc.hint)
    except Exception as exc:
        st.error(f"Could not fetch weather or compute forecast. Please retry in a minute. Details: {exc}")
        if APP_DEBUG:
            with st.expander("Debug traceback", expanded=False):
                st.code(traceback.format_exc())
