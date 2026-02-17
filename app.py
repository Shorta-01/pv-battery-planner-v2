from __future__ import annotations

import datetime as dt
import html
import inspect
import json
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import planner_core as core
from tariff_time import compute_offpeak_segments, make_summary_lines, parse_hhmm

PLOTLY_DARK = "plotly_dark"

INPUT_TOOLTIPS = {
    "soc_percent": "This is your battery level at 22:00. It matters because charging need is based on how full the battery already is. Example: 35 means the battery starts at 35%.",
    "yesterday_kwh": "This is your total home usage yesterday. It matters because the app uses it to estimate tomorrow's hourly load. Example: if yesterday was 18 kWh, tomorrow's hourly load profile scales to 18 kWh.",
    "buffer_percent": "This adds a safety margin to the target SOC. It matters when forecasts are uncertain. Example: 3% means the target cutoff SOC is increased by 3 percentage points.",
    "performance_ratio": "This is overall PV system efficiency after real-world losses. It matters because lower efficiency means lower expected production. Example: 0.85 means around 85% of ideal output.",
    "inverter_eff": "This is inverter conversion efficiency from DC to AC. It matters when PV loss model is split; in combined mode it is ignored in PV calculations.",
    "pv_loss_model": "Choose how PV losses are applied before inverter modeling: split = performance ratio then inverter efficiency/model, combined = performance ratio only.",
    "iam_model": "Incidence-angle modifier model for reflection losses at high sun angles. none keeps legacy behavior; ashrae applies AOI-based optical losses.",
    "iam_ashrae_b": "ASHRAE IAM coefficient b (only used when IAM model = ashrae). Typical range 0.02-0.12; higher means stronger angular losses.",
    "albedo": "Ground reflectance used in transposition (None keeps pvlib default). Typical values: 0.2 grass, 0.6+ bright snow.",
    "inverter_ac_model": "AC conversion model: linear reproduces legacy constant multiplier, pvwatts enables part-load inverter efficiency behavior.",
    "pv_calibration_factor": "Global PV tuning factor applied to both arrays. Effective east = global × east, effective south = global × south. 1.00 = unchanged.",
    "pv_calibration_factor_east": "East-array relative tuning factor multiplied by the global PV calibration factor. 1.00 keeps east at the global factor.",
    "pv_calibration_factor_south": "South-array relative tuning factor multiplied by the global PV calibration factor. 1.00 keeps south at the global factor.",
    "max_ac_user_cap": "The app computes a recommended AC charge power from required energy and off-peak window hours. This field is your safety cap: final used value is min(recommended, your cap, inverter/battery limits).",
}

METRIC_TOOLTIPS = {
    "Allowed AC charge power (kW)": "Final AC charge power used by the planner after all limits. It matters because this is the value to configure in FusionSolar. It never exceeds your safety cap, inverter limits, or battery limits.",
    "AC charge cutoff SOC (%)": "Battery SOC where FusionSolar should stop charging from the grid. Set this value as the 'AC charge cutoff SOC'.",
    "Forecast total PV (kWh)": "Estimated total PV energy produced tomorrow (after inverter AC limit).",
    "Forecast total load (kWh)": "Estimated total consumption for tomorrow. This is based on yesterday's total and a default hourly profile.",
    "Estimated grid import (expensive h)": "Estimated energy you may still buy from the grid during expensive tariff hours after using PV and the battery.",
    "Estimated export/curtailment (kWh)": "Estimated PV energy that cannot be used or stored and may be exported to the grid (or clipped/curtailed).",
}

CHART_TOOLTIPS = {
    "PV production vs Load (hourly)": "This chart shows hourly energy from your PV and your home load, plus battery SOC on a secondary axis. It helps you see when solar covers usage and how battery charge evolves through the day.",
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

WEATHER_MODEL_HOVERTEXT = {
    "knmi_harmonie_arome": "High-resolution KNMI regional model for Benelux. Strong for short-term local cloud and wind changes.",
    "dwd_icon_d2": "Very high-resolution DWD model (Germany region). Often captures fast cloud transitions that impact PV.",
    "ecmwf_ifs": "ECMWF global model. Very reliable for fronts and the overall weather pattern, good stable baseline.",
    "dwd_icon_eu": "European ICON model. Useful secondary view when the high-res model is noisy or inconsistent.",
    "meteofrance_seamless": "Météo-France seamless blend. Helpful extra perspective for Western Europe cloud patterns.",
}

BADGE_HOVERTEXT = {
    "⭐": "Core model. Recommended default for Belgium.",
    "🟩": "Best PV inputs. This model provides the main solar irradiance fields directly, which usually improves PV accuracy.",
    "🧩": "Derived PV inputs. Some solar irradiance fields are missing, so we estimate them. PV still works, but accuracy can drop on difficult cloud days.",
    "🔎": "High-resolution (local). Best for short-term local cloud timing, which often improves PV ramps and hour-to-hour changes.",
    "🗺️": "Regional (Europe-scale). A good second opinion that is usually smoother and more stable than high-resolution models.",
    "🌍": "Global. Stable big-picture baseline for fronts and the overall weather pattern.",
    "⏱️": "Uses 15-minute solar radiation (then aggregated to hourly). Can improve the PV curve shape when clouds change quickly.",
}


def _esc(s: str) -> str:
    return html.escape(str(s or ""), quote=True)


def get_selected_weather_models(valid_model_ids: set[str]) -> list[str]:
    selected: list[str] = []
    for mid in WEATHER_MODEL_ORDER:
        if mid in valid_model_ids and bool(st.session_state.get(f"wm_{mid}", False)):
            selected.append(mid)
    return selected

LOCAL_STATE_DIR = Path("local_state")
API_BASE_URL = os.getenv("PVBP_BACKEND_URL", "http://127.0.0.1:8787")
API_TOKEN_FILE = LOCAL_STATE_DIR / "api_token.txt"

st.session_state.setdefault("history_all_runs", False)
st.session_state.setdefault("history_show_run_at", False)

def apply_pending_location_state() -> None:
    pending = st.session_state.pop("_pending_location_state", None)
    if not isinstance(pending, dict):
        return
    structured = pending.get("address_structured", {}) if isinstance(pending.get("address_structured"), dict) else {}
    st.session_state["loc_address_query_display"] = str(pending.get("address_query", ""))
    st.session_state["loc_latitude"] = float(pending.get("latitude", core.LATITUDE))
    st.session_state["loc_longitude"] = float(pending.get("longitude", core.LONGITUDE))
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
    cfg["location"]["latitude"] = float(res.get("latitude", core.LATITUDE))
    cfg["location"]["longitude"] = float(res.get("longitude", core.LONGITUDE))
    cfg["location"]["timezone"] = str(res.get("timezone", core.TIMEZONE))
    cfg["location"]["address_structured"] = res.get("address_structured", {})

    st.session_state["loc_address_query_display"] = str(res.get("address_query", ""))
    st.session_state["loc_latitude"] = float(res.get("latitude", core.LATITUDE))
    st.session_state["loc_longitude"] = float(res.get("longitude", core.LONGITUDE))
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
        "latitude": float(st.session_state.get("loc_latitude", loc_cfg.get("latitude", core.LATITUDE))),
        "longitude": float(st.session_state.get("loc_longitude", loc_cfg.get("longitude", core.LONGITUDE))),
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def weather_model_option_help(model: dict) -> str:
    badge_meanings = {
        "⭐": "recommended for Belgium",
        "🟩": "full irradiance fields",
        "🧩": "derived/approximated components",
    }
    badge_meanings.update(
        {
            "🔎": "high-resolution local",
            "🗺️": "regional Europe-scale",
            "🌍": "global baseline",
            "⏱️": "uses 15-minute solar data",
        }
    )
    badges = [badge for badge in model.get("badges", []) if badge in badge_meanings]
    unique_badges = list(dict.fromkeys(badges))
    badge_summary = ", ".join(f"{badge} {badge_meanings[badge]}" for badge in unique_badges)
    notes = str(model.get("notes") or model.get("capability", {}).get("notes") or "")
    if notes:
        return f"{notes}\n\nLegend: {badge_summary}."
    return f"Legend: {badge_summary}."


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
        return ("⚠️", " ".join(parts))

    return (None, "")


def render_weather_models(
    weather_models_catalog: list[dict],
    default_selected: set[str],
    *,
    widget_key_prefix: str = "wm",
) -> list[str]:
    with st.expander("Weather models", expanded=True):
        st.caption("Select which weather models to use. We combine them automatically using Belgium-tuned weighting.")
        st.caption("After you run a forecast, we show warnings (⚠️) or failures (❌) next to models if needed.")

        model_options = {m.get("id"): m for m in weather_models_catalog if isinstance(m.get("id"), str)}
        selected_models: list[str] = []

        st.markdown(
            "<style>.wm-name{cursor:help}.wm-icon{cursor:help;margin-left:6px}</style>",
            unsafe_allow_html=True,
        )

        for model_id in WEATHER_MODEL_ORDER:
            model = model_options.get(model_id)
            if not model:
                continue

            cols = st.columns([0.35, 3.2, 1.3], vertical_alignment="center")

            with cols[0]:
                checked = st.checkbox(
                    "enabled",
                    value=(model_id in default_selected),
                    key=f"{widget_key_prefix}_{model_id}",
                    label_visibility="collapsed",
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

                badge_items: list[tuple[str, str]] = []
                if status_icon:
                    badge_items.append((status_icon, status_tip))

                for b in static_badges:
                    badge_items.append((b, BADGE_HOVERTEXT.get(b, "")))

                icon_html = " ".join(
                    f"<span class='wm-icon' title='{_esc(tip)}'>{_esc(icon)}</span>"
                    for icon, tip in badge_items
                )
                st.markdown(icon_html, unsafe_allow_html=True)

            if checked:
                selected_models.append(model_id)

        if not selected_models:
            st.error("Select at least one weather model.")
        else:
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


def render_modern_table(df: pd.DataFrame, column_config: dict | None = None) -> None:
    if df is None or df.empty:
        st.info("No data available.")
        return
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )


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


def render_pv_quality_widget(container, pv_df: pd.DataFrame, pv_quality_dict: dict, tomorrow_date: dt.date) -> None:
    _ = pv_df
    fallback_note = ""
    if pv_quality_dict.get("is_fallback"):
        fallback_note = "<div style='font-size:0.72rem;opacity:0.75;margin-top:0.25rem;'>(fallback scoring)</div>"

    quality_emojis = {
        "Excellent": "☀️",
        "Good": "🌤️",
        "Mixed": "⛅",
        "Poor": "🌥️",
        "Very low": "🌧️",
    }
    score = int(pv_quality_dict["score"])
    ratio_percent = max(0.0, min(float(pv_quality_dict.get("ratio", 0.0)) * 100.0, 100.0))
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
    summary_html = (
        f"<div style='margin-top:0.30rem;font-size:0.70rem;opacity:0.92;'>{offpeak_summary}</div>"
        + (f"<div style='font-size:0.70rem;opacity:0.86;'>{peak_summary}</div>" if peak_summary else "")
    )

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
            "Savings (no battery vs battery plan)"
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
            "Hourly savings for tomorrow (00–24). Total includes tonight 22–24 charging."
            "</div>"
            "</div>"
        )
    else:
        savings_html = "<div style='margin-top:0.50rem;font-size:0.70rem;opacity:0.78;'>Run forecast to see € savings.</div>"

    container.markdown(
        (
            "<div style='border:1px solid rgba(255,255,255,0.12);border-radius:16px;padding:0.65rem 0.75rem;"
            "background:linear-gradient(140deg, rgba(43,48,58,0.9), rgba(20,24,31,0.85));min-width:245px;'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;gap:0.5rem;'>"
            f"<div style='font-size:1.25rem;line-height:1;'>{quality_emojis.get(pv_quality_dict['label'], '☁️')}</div>"
            "<div style='font-size:0.72rem;opacity:0.8;text-transform:uppercase;letter-spacing:0.06em;'>PV Outlook</div>"
            f"<div style='font-size:0.9rem;font-weight:700;color:{pv_quality_dict['color']};'>{score}/100</div>"
            "</div>"
            "<div style='margin-top:0.35rem;font-size:0.95rem;font-weight:650;'>"
            f"{pv_quality_dict['label']} day · {pv_quality_dict['pv_total_kwh']:.1f} kWh"
            "</div>"
            "<div style='margin-top:0.45rem;height:8px;border-radius:999px;overflow:hidden;background:rgba(255,255,255,0.12);'>"
            f"<div style='height:100%;width:{ratio_percent:.1f}%;background:linear-gradient(90deg,#d62828 0%,#f4a261 45%,#52b788 70%,#2a9d8f 100%);'></div>"
            "</div>"
            "<div style='margin-top:0.32rem;font-size:0.72rem;opacity:0.8;'>"
            f"{ratio_percent:.0f}% of clear-sky potential"
            "</div>"
            "<div style='margin-top:0.55rem;'>"
            f"<div title='Green = off-peak, Red = peak' style='height:8px;width:100%;position:relative;{timeline_base}"
            "border-radius:999px;overflow:hidden;'>"
            + "".join(overlays)
            + "</div>"
            + "<div style='margin-top:0.28rem;font-size:0.68rem;opacity:0.8;'>"
            "Tariff timeline (00–24)"
            "</div>"
            + summary_html
            + savings_html
            + "</div>"
            "</div>"
            f"{fallback_note}"
        ),
        unsafe_allow_html=True,
    )


def render_key_charging_widget(container, allowed_charge_kw: float, cutoff_soc_pct: float, cutoff_note: str) -> None:
    power_pct = clamp_pct((allowed_charge_kw / 7.0) * 100.0)
    soc_pct = clamp_pct(cutoff_soc_pct)
    container.markdown(
        (
            "<div style='border:1px solid rgba(255,255,255,0.12);border-radius:16px;padding:0.75rem 0.85rem;"
            "background:linear-gradient(140deg, rgba(43,48,58,0.9), rgba(20,24,31,0.85));'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;gap:0.5rem;'>"
            "<div style='font-size:0.72rem;opacity:0.8;text-transform:uppercase;letter-spacing:0.06em;'>Key charging targets</div>"
            "<div style='font-size:0.95rem;'>🔋⚡</div>"
            "</div>"
            "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0.9rem;margin-top:0.55rem;'>"
            "<div style='padding:0.45rem 0.55rem;border:1px solid rgba(255,255,255,0.08);border-radius:12px;background:rgba(255,255,255,0.02);'>"
            "<div style='font-size:0.72rem;opacity:0.84;'>Allowed AC charge power (kW)</div>"
            f"<div style='margin-top:0.22rem;font-size:1.45rem;font-weight:700;'>{allowed_charge_kw:.2f}</div>"
            "<div style='margin-top:0.32rem;height:10px;border-radius:999px;background:rgba(255,255,255,0.12);overflow:hidden;'>"
            f"<div style='height:100%;width:{power_pct:.1f}%;background:linear-gradient(90deg,#4cc9f0,#4895ef,#4361ee);'></div>"
            "</div>"
            "<div style='margin-top:0.22rem;font-size:0.68rem;opacity:0.78;'>Range: 0 to 7 kW</div>"
            "</div>"
            "<div style='padding:0.45rem 0.55rem;border:1px solid rgba(255,255,255,0.08);border-radius:12px;background:rgba(255,255,255,0.02);'>"
            "<div style='font-size:0.72rem;opacity:0.84;'>AC charge cutoff SOC (%)</div>"
            f"<div style='margin-top:0.22rem;font-size:1.45rem;font-weight:700;'>{cutoff_soc_pct:.1f}%</div>"
            "<div style='margin-top:0.32rem;height:10px;border-radius:999px;background:rgba(255,255,255,0.12);overflow:hidden;'>"
            f"<div style='height:100%;width:{soc_pct:.1f}%;background:linear-gradient(90deg,#f4a261,#e9c46a,#52b788);'></div>"
            "</div>"
            "<div style='margin-top:0.22rem;font-size:0.68rem;opacity:0.78;'>Range: 0 to 100%</div>"
            "</div>"
            "</div>"
            f"<div style='margin-top:0.5rem;font-size:0.72rem;opacity:0.84;'>{cutoff_note}</div>"
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


def clamp_pct(x: float) -> float:
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
        "Load",
        "Charge",
        "Warnings",
        "warnings_count",
        "run_type",
        "models_raw",
        "warnings_raw",
        "models_summary_raw",
    ]
    try:
        show_all_text = "true" if show_all_runs else "false"
        items = api_get(f"/v1/results/history?days={max(1, int(days))}&show_all_runs={show_all_text}").get("items", [])
    except Exception:
        return pd.DataFrame(columns=history_columns)

    rows = []
    for item in items:
        metrics = item.get("metrics", {})
        status_raw = str(item.get("status") or "").strip().lower()
        warnings_count = int(item.get("warnings_count") or 0)
        if status_raw == "error":
            status_label = "❌ Error"
        elif status_raw == "degraded" or warnings_count > 0:
            status_label = "⚠️ Degraded"
        else:
            status_label = "✅ OK"

        models_summary = item.get("models_summary") if isinstance(item.get("models_summary"), dict) else {}
        models_list = models_summary.get("selected_models") if isinstance(models_summary, dict) else []
        if not isinstance(models_list, list):
            models_list = []
        models_text = ", ".join(str(m) for m in models_list) if models_list else "—"

        pv_p10 = float(item.get("pv_p10_kwh") or 0.0)
        pv_p50 = float(item.get("pv_p50_kwh") or metrics.get("pv_forecast_kwh") or 0.0)
        pv_p90 = float(item.get("pv_p90_kwh") or 0.0)
        warnings_raw = item.get("warnings") if isinstance(item.get("warnings"), list) else []
        warnings_text = " | ".join(str(w) for w in warnings_raw) if warnings_raw else (f"{warnings_count} warning(s)" if warnings_count else "None")

        rows.append({
            "run_id": str(item.get("run_id") or ""),
            "Date": item.get("target_date"),
            "Run at": item.get("run_at"),
            "Status": status_raw or "ok",
            "Status label": status_label,
            "Models": models_text,
            "PV p50": round(pv_p50, 2),
            "PV p10": round(pv_p10, 2),
            "PV p90": round(pv_p90, 2),
            "PV range (p10–p90)": f"{pv_p10:.2f}–{pv_p90:.2f} kWh" if (pv_p10 or pv_p90) else "—",
            "Load": round(float(metrics.get("cons_forecast_kwh", 0.0)), 2),
            "Charge": round(float(metrics.get("charge_kw", 0.0)), 2),
            "Warnings": warnings_text,
            "warnings_count": warnings_count,
            "run_type": str(item.get("run_type") or "manual"),
            "models_raw": models_text,
            "warnings_raw": warnings_raw,
            "models_summary_raw": models_summary,
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

    if "Date" in working.columns:
        working["Date"] = pd.to_datetime(working["Date"], errors="coerce").dt.date

    if "Run at" in working.columns:
        working["Run at"] = pd.to_datetime(working["Run at"], errors="coerce")

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


def _render_run_inspector(filtered_df: pd.DataFrame) -> None:
    if filtered_df.empty:
        return

    filtered_df = filtered_df.sort_values(["Date", "Run at"], ascending=[False, False]).reset_index(drop=True)
    options = {
        f"{row['Date'].date().isoformat()} · {row['Status label']} · {row['run_type']} · {row.get('run_id') or ('row-' + str(i + 1))}": i
        for i, row in filtered_df.iterrows()
    }
    picked = st.selectbox("Inspect a run", list(options.keys()), key="history_inspector_run")
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

    with st.expander("Run Inspector", expanded=False):
        tab_summary, tab_models, tab_inputs, tab_settings, tab_debug = st.tabs(
            ["Summary", "Weather models", "Inputs used", "Settings used", "Debug bundle"]
        )

        with tab_summary:
            st.markdown(
                f"**Date:** {row['Date'].date().isoformat()}  \\\n"
                f"**Run at:** {row['Run at'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['Run at']) else '—'}  \\\n"
                f"**Status:** {row['Status label']}  \\\n"
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
            st.write("Selected models:", ", ".join(selected_models) if selected_models else "—")
            st.write("Failed models:", ", ".join(failed_models) if failed_models else "None")
            if isinstance(weights_used, dict) and weights_used:
                weights_df = pd.DataFrame([{"Model": k, "Weight": float(v)} for k, v in weights_used.items()])
                st.dataframe(weights_df, use_container_width=True, hide_index=True)

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
                        "p10": float(history_row.get("PV p10") or 0.0),
                        "p50": float(history_row.get("PV p50") or 0.0),
                        "p90": float(history_row.get("PV p90") or 0.0),
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
                file_name=f"debug_bundle_{debug_bundle.get('run_id') or 'run'}.json",
                mime="application/json",
                key=f"history_inspector_debug_bundle_download_{run_id or 'row'}",
            )


def _render_history_log_block() -> None:
    tooltip_heading("History log", TABLE_TOOLTIPS["History log"])

    with st.expander("History log", expanded=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

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

        raw = run_history_from_backend(show_all_runs=st.session_state["history_all_runs"], days=365)
        prepared = _prepare_history_df(
            raw,
            all_runs=st.session_state["history_all_runs"],
            show_run_at=st.session_state["history_show_run_at"],
        )

        if not prepared.empty:
            date_min = prepared["Date"].min().date()
            date_max = prepared["Date"].max().date()
            f1, f2, f3 = st.columns([1.2, 1.2, 1.2])
            with f1:
                selected_date_range = st.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
            with f2:
                status_options = ["✅ OK", "⚠️ Degraded", "❌ Error"]
                status_filter = st.multiselect("Status", options=status_options, default=status_options)
            with f3:
                run_types = sorted({str(v) for v in prepared["run_type"].dropna().tolist()})
                run_type_filter = st.multiselect("Run type", options=run_types, default=run_types)

            f4, f5 = st.columns([1.2, 2.4])
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
            if run_type_filter:
                filtered = filtered[filtered["run_type"].isin(run_type_filter)]
            if has_warnings == "Yes":
                filtered = filtered[filtered["warnings_count"] > 0]
            elif has_warnings == "No":
                filtered = filtered[filtered["warnings_count"] == 0]
            if model_filter != "All models":
                filtered = filtered[filtered["models_raw"].str.contains(model_filter, na=False)]
        else:
            filtered = prepared

        with c3:
            st.caption("")

        if filtered.empty:
            st.info("No history records yet. Run a forecast to create the first record.")
        else:
            display_df = filtered.copy()
            display_df["Date"] = display_df["Date"].astype(str)
            if "Run at" in display_df.columns:
                display_df["Run at"] = display_df["Run at"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
                if not st.session_state.get("history_show_run_at", False):
                    display_df = display_df.drop(columns=["Run at"])

            drop_cols = ["run_id", "Status", "PV p10", "PV p90", "warnings_count", "run_type", "models_raw", "warnings_raw", "models_summary_raw"]
            display_df = display_df.drop(columns=[c for c in drop_cols if c in display_df.columns])
            history_column_config = build_column_config(
                display_df,
                {
                    "PV p50": st.column_config.NumberColumn(format="%.2f kWh"),
                    "Load": st.column_config.NumberColumn(format="%.2f kWh"),
                    "Charge": st.column_config.NumberColumn(format="%.2f kW"),
                },
            )
            render_modern_table(display_df, column_config=history_column_config)
            _render_run_inspector(filtered)


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
        load = pd.to_numeric(out.get("load_kwh", 0.0), errors="coerce").fillna(0.0)
        out["pv_surplus_kwh"] = (out["pv_total_kwh"] - load).clip(lower=0.0)
    if "pv_deficit_kwh" not in out.columns:
        load = pd.to_numeric(out.get("load_kwh", 0.0), errors="coerce").fillna(0.0)
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
        "Load kWh (hour): %{customdata[3]:.2f}<br>"
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
            name="Estimated load",
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
        st.session_state["loc_latitude"] = float(loc_cfg.get("latitude", core.LATITUDE))
    if "loc_longitude" not in st.session_state:
        st.session_state["loc_longitude"] = float(loc_cfg.get("longitude", core.LONGITUDE))
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
            step=0.5,
            help=INPUT_TOOLTIPS["soc_percent"],
        )
        yesterday_kwh = st.number_input(
            "Yesterday total consumption (kWh)",
            min_value=0.1,
            value=float(st.session_state.last_kwh),
            step=0.1,
            help=INPUT_TOOLTIPS["yesterday_kwh"],
        )

    with st.expander("Saved settings"):
        st.markdown("#### Location")
        addr_col, status_col, btn_col = st.columns([6, 1, 2], vertical_alignment="center")
        with addr_col:
            st.text_input("Address query", key="loc_address_query_display", disabled=True)
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

        with st.form("settings_form"):
            loc_col1, loc_col2 = st.columns(2)
            with loc_col1:
                cfg_latitude = st.number_input(
                    "Latitude",
                    min_value=-90.0,
                    max_value=90.0,
                    step=0.00001,
                    format="%.5f",
                    key="loc_latitude",
                )
            with loc_col2:
                cfg_longitude = st.number_input(
                    "Longitude",
                    min_value=-180.0,
                    max_value=180.0,
                    step=0.00001,
                    format="%.5f",
                    key="loc_longitude",
                )
            cfg_timezone = st.text_input("Timezone", key="loc_timezone", disabled=True)

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
                    help="All-in import price (€/kWh). Enter your full import price.",
                )
            with c2:
                cfg_offpeak_price_input = st.number_input(
                    "Off-peak grid price",
                    min_value=0.0,
                    step=0.001,
                    format="%.4f",
                    value=cfg_offpeak_price,
                    key="tariff_offpeak_price",
                    help="All-in import price (€/kWh). Enter your full import price.",
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
            tariff_inputs: list[tuple[str, str]] = []
            for day_idx, day_name in enumerate(day_names):
                day_windows = tariff_by_day.get(day_idx, [("22:00", "07:00")])
                day_from, day_to = day_windows[0] if day_windows else ("22:00", "07:00")
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
                cfg_panel_wp = st.number_input("Panel power (Wp)", min_value=1, value=int(cfg_pv["panel_wp"]), step=1)
            with row1_col2:
                cfg_performance_ratio = st.number_input(
                    "Performance ratio",
                    min_value=0.50,
                    max_value=1.00,
                    step=0.01,
                    value=float(cfg_pv["performance_ratio"]),
                )

            row2_col1, row2_col2, row2_col3 = st.columns(3)
            with row2_col1:
                cfg_array_east_panels = st.number_input("East array panels", min_value=1, value=int(cfg_pv["array_east_panels"]), step=1)
            with row2_col2:
                cfg_tilt_east_deg = st.number_input("Tilt East (deg)", value=float(cfg_pv["tilt_east_deg"]), step=0.1)
            with row2_col3:
                cfg_azimuth_east_deg = st.number_input("Azimuth East (deg)", value=float(cfg_pv["azimuth_east_deg"]), step=0.1)

            row3_col1, row3_col2, row3_col3 = st.columns(3)
            with row3_col1:
                cfg_array_south_panels = st.number_input("South array panels", min_value=1, value=int(cfg_pv["array_south_panels"]), step=1)
            with row3_col2:
                cfg_tilt_south_deg = st.number_input("Tilt South (deg)", value=float(cfg_pv["tilt_south_deg"]), step=0.1)
            with row3_col3:
                cfg_azimuth_south_deg = st.number_input("Azimuth South (deg)", value=float(cfg_pv["azimuth_south_deg"]), step=0.1)

            row4_col1, row4_col2 = st.columns(2)
            with row4_col1:
                cfg_pv_loss_model = st.selectbox(
                    "PV loss model",
                    options=["split", "combined"],
                    index=["split", "combined"].index(str(cfg_pv.get("pv_loss_model", "split")).strip().lower() if str(cfg_pv.get("pv_loss_model", "split")).strip().lower() in {"split", "combined"} else "split"),
                    help=INPUT_TOOLTIPS["pv_loss_model"],
                )
            with row4_col2:
                cfg_inverter_ac_kw_limit = st.number_input("Inverter AC limit (kW)", min_value=0.1, value=float(cfg_pv["inverter_ac_kw_limit"]), step=0.1)

            row5_col1, row5_col2 = st.columns(2)
            with row5_col1:
                cfg_inverter_eff = st.number_input(
                    "Inverter efficiency",
                    min_value=0.50,
                    max_value=1.00,
                    value=float(cfg_pv["inverter_eff"]),
                    step=0.01,
                    disabled=(cfg_pv_loss_model == "combined"),
                    help=INPUT_TOOLTIPS["inverter_eff"],
                )
            with row5_col2:
                if cfg_pv_loss_model == "combined":
                    st.caption("Inverter efficiency is ignored for linear mode in combined losses and treated as nominal eta for pvwatts.")
                else:
                    st.caption("In split mode, inverter efficiency feeds the selected AC model.")

            row5b_col1, row5b_col2 = st.columns(2)
            with row5b_col1:
                inverter_ac_model_value = str(cfg_pv.get("inverter_ac_model", "linear")).strip().lower()
                cfg_inverter_ac_model = st.selectbox(
                    "Inverter AC model",
                    options=["linear", "pvwatts"],
                    index=["linear", "pvwatts"].index(inverter_ac_model_value if inverter_ac_model_value in {"linear", "pvwatts"} else "linear"),
                    help=INPUT_TOOLTIPS["inverter_ac_model"],
                )
            with row5b_col2:
                iam_model_value = str(cfg_pv.get("iam_model", "none")).strip().lower()
                cfg_iam_model = st.selectbox(
                    "IAM model",
                    options=["none", "ashrae"],
                    index=["none", "ashrae"].index(iam_model_value if iam_model_value in {"none", "ashrae"} else "none"),
                    help=INPUT_TOOLTIPS["iam_model"],
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
                )
            with row5c_col2:
                albedo_default = cfg_pv.get("albedo", None)
                cfg_albedo_enabled = st.checkbox("Set custom albedo", value=albedo_default is not None)
                cfg_albedo = st.number_input(
                    "Albedo",
                    min_value=0.00,
                    max_value=1.00,
                    value=float(albedo_default if albedo_default is not None else 0.20),
                    step=0.01,
                    disabled=(not cfg_albedo_enabled),
                    help=INPUT_TOOLTIPS["albedo"],
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
                )

            st.markdown("#### Battery")
            bat_col1, bat_col2 = st.columns(2)
            with bat_col1:
                cfg_battery_kwh = st.number_input("Battery capacity (kWh)", min_value=0.1, value=float(effective_cfg["battery"]["battery_kwh"]), step=0.1)
                cfg_min_soc_percent = st.number_input("Min SOC (%)", min_value=0.0, max_value=100.0, value=float(effective_cfg["battery"]["min_soc_percent"]), step=0.5)
                cfg_max_cutoff_soc_percent = st.number_input("Max cutoff SOC (%)", min_value=0.0, max_value=100.0, value=float(effective_cfg["battery"]["max_cutoff_soc_percent"]), step=0.5)
            with bat_col2:
                cfg_battery_max_charge_kw = st.number_input("Battery max charge (kW)", min_value=0.1, value=float(effective_cfg["battery"]["battery_max_charge_kw"]), step=0.1)
                cfg_battery_max_discharge_kw = st.number_input("Battery max discharge (kW)", min_value=0.1, value=float(effective_cfg["battery"]["battery_max_discharge_kw"]), step=0.1)
                cfg_max_ac_charge_kw_hard_limit = st.number_input("Max AC charge kW hard limit", min_value=0.1, value=float(effective_cfg["battery"]["max_ac_charge_kw_hard_limit"]), step=0.1)

            st.markdown("#### Load profile")
            edit_load_profile = st.checkbox("Edit load profile", value=False)
            cfg_load_profile = [float(v) for v in effective_cfg["load_profile"]["load_profile_24h"]]
            if edit_load_profile:
                lp_cols = st.columns(4)
                for hour in range(24):
                    with lp_cols[hour % 4]:
                        cfg_load_profile[hour] = st.number_input(
                            f"Hour {hour:02d}",
                            value=float(cfg_load_profile[hour]),
                            step=0.001,
                            format="%.3f",
                            key=f"load_profile_{hour}",
                        )

            btn_left, btn_right = st.columns([3, 2])
            with btn_left:
                save_settings = st.form_submit_button(
                    "Save settings",
                    type="primary",
                    width="content",
                    key="btn_save_settings",
                )
            with btn_right:
                reset_defaults = st.form_submit_button(
                    "Reset to defaults",
                    type="secondary",
                    width="stretch",
                    key="btn_reset_defaults",
                )

        if st.session_state.get("_geo_success"):
            st.success(st.session_state["_geo_success"])
        if st.session_state.get("_geo_error"):
            st.error(st.session_state["_geo_error"])
        flash = st.session_state.pop("_settings_flash", None)
        if flash:
            st.success(flash)

        if save_settings:
            tariff_error = None
            for day_idx, (from_value, to_value) in enumerate(tariff_inputs):
                try:
                    start_min = parse_hhmm(from_value, allow_24_end=False)
                    end_min = parse_hhmm(to_value, allow_24_end=True)
                    compute_offpeak_segments(start_min, end_min)
                except ValueError as exc:
                    tariff_error = f"Tariff settings error for {day_names[day_idx]}: {exc}"
                    break

            if tariff_error:
                st.error(tariff_error)
            else:
                selected_to_save = get_selected_weather_models(valid_model_ids)
                if not selected_to_save:
                    selected_to_save = sorted(list(WEATHER_MODEL_DEFAULT & valid_model_ids)) or sorted(list(valid_model_ids))

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
                        "latitude": float(cfg_latitude),
                        "longitude": float(cfg_longitude),
                        "timezone": str(st.session_state["loc_timezone"]),
                    },
                    "tariff": {
                        "peak_grid_price_eur_per_kwh": float(cfg_peak_price_input),
                        "offpeak_grid_price_eur_per_kwh": float(cfg_offpeak_price_input),
                        "injection_grid_price_eur_per_kwh": float(cfg_injection_price_input),
                        "offpeak_windows_by_dow": [
                            [[from_value, to_value], *[[w_start, w_end] for (w_start, w_end) in tariff_by_day.get(day_idx, [])[1:]]] if tariff_by_day.get(day_idx) else [[from_value, to_value]]
                            for day_idx, (from_value, to_value) in enumerate(tariff_inputs)
                        ],
                    },
                    "pv": {
                        "panel_wp": int(cfg_panel_wp),
                        "array_south_panels": int(cfg_array_south_panels),
                        "array_east_panels": int(cfg_array_east_panels),
                        "tilt_east_deg": float(cfg_tilt_east_deg),
                        "tilt_south_deg": float(cfg_tilt_south_deg),
                        "azimuth_east_deg": float(cfg_azimuth_east_deg),
                        "azimuth_south_deg": float(cfg_azimuth_south_deg),
                        "performance_ratio": float(cfg_performance_ratio),
                        "inverter_eff": float(cfg_inverter_eff),
                        "pv_loss_model": str(cfg_pv_loss_model),
                        "iam_model": str(cfg_iam_model),
                        "iam_ashrae_b": float(cfg_iam_ashrae_b),
                        "albedo": (float(cfg_albedo) if cfg_albedo_enabled else None),
                        "inverter_ac_model": str(cfg_inverter_ac_model),
                        "pv_calibration_factor": float(cfg_pv_calibration_factor),
                        "pv_calibration_factor_east": float(cfg_pv_calibration_factor_east),
                        "pv_calibration_factor_south": float(cfg_pv_calibration_factor_south),
                        "inverter_ac_kw_limit": float(cfg_inverter_ac_kw_limit),
                    },
                    "battery": {
                        "battery_kwh": float(cfg_battery_kwh),
                        "min_soc_percent": float(cfg_min_soc_percent),
                        "max_cutoff_soc_percent": float(cfg_max_cutoff_soc_percent),
                        "battery_max_charge_kw": float(cfg_battery_max_charge_kw),
                        "battery_max_discharge_kw": float(cfg_battery_max_discharge_kw),
                        "max_ac_charge_kw_hard_limit": float(cfg_max_ac_charge_kw_hard_limit),
                    },
                    "load_profile": {
                        "load_profile_24h": [float(v) for v in cfg_load_profile],
                    },
                    "weather_models_selected": selected_to_save,
                }
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
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save settings: {exc}")

        if reset_defaults:
            try:
                updated = api_put(
                    "/v1/settings",
                    {
                        "config": core.DEFAULT_CONFIG,
                        "nightly_run_time": backend_settings.get("nightly_run_time", "22:00"),
                        "timezone": str(core.DEFAULT_CONFIG["location"].get("timezone", backend_settings.get("timezone", "Europe/Brussels"))),
                        "max_ac_charge_power_kw_default": backend_settings.get("max_ac_charge_power_kw_default", 5.0),
                    },
                )
                st.cache_data.clear()
                st.session_state["_pending_location_state"] = updated["config"]["location"]
                st.session_state["_settings_flash"] = "Reset settings to defaults."
                for mid in WEATHER_MODEL_ORDER:
                    st.session_state.pop(f"wm_{mid}", None)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not reset settings: {exc}")

    with st.expander("Advanced"):
        buffer_percent = st.slider("Forecast safety buffer SOC (%)", 0.0, 10.0, 0.0, 0.5, help=INPUT_TOOLTIPS["buffer_percent"])
        user_max_ac_kw = st.number_input(
            "Max allowed AC charge power (kW)",
            min_value=0.0,
            max_value=10.0,
            value=float(backend_settings.get("max_ac_charge_power_kw_default", 5.0)),
            step=0.1,
            help=INPUT_TOOLTIPS["max_ac_user_cap"],
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
            help="Pick the nightly scheduler trigger time.",
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

    weather_models_box = st.empty()
    with weather_models_box.container():
        selected_models = render_weather_models(weather_models_catalog, initial_selected, widget_key_prefix="wm")

    ensemble_method = "weighted"
    run = st.button(
        "Run forecast",
        type="primary",
        disabled=not bool(selected_models),
    )

if run:
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
                    "weather_models": selected_models,
                    "ensemble_method": ensemble_method,
                    "pv_uncertainty": True,
                },
            )
            result = run_response["result"]
            dbg = result.get("weather_ensemble")
            st.session_state["last_weather_ensemble_debug"] = dbg if isinstance(dbg, dict) else {}
            st.session_state["last_weather_ensemble_debug_at"] = dt.datetime.utcnow().isoformat()
            st.session_state["last_weather_ensemble_models_used"] = list(selected_models)
            weather_models_box.empty()
            with weather_models_box.container():
                _ = render_weather_models(weather_models_catalog, initial_selected, widget_key_prefix="wm_last")
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
            metrics = result.get("metrics", {})
            pv_quality = result.get("pv_quality", {})
            cutoff_soc = float(metrics.get("cutoff_soc", 0.0))
            charge_kw = float(metrics.get("charge_kw", 0.0))
            charge_note = str(metrics.get("charge_note", ""))
            cutoff_reason_ui = str(metrics.get("cutoff_reason", ""))
            grid_import = float(metrics.get("grid_import", 0.0))
            grid_export = float(metrics.get("grid_export", 0.0))
            weather_ensemble = result.get("weather_ensemble", {}) if isinstance(result.get("weather_ensemble"), dict) else {}
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
                    cutoff_note=cutoff_reason_ui,
                )
            with top_right:
                render_pv_quality_widget(top_right, pv, pv_quality, tomorrow)

            if charge_note.startswith("Warning"):
                st.warning(charge_note)

            st.markdown("### Forecast summary")
            c1, c2, c3, c4 = st.columns(4)
            metric_with_help(c1, "Forecast total PV (kWh)", f"{pv['pv_total_kwh'].sum():.2f}")
            metric_with_help(c2, "Forecast total load (kWh)", f"{pv['load_kwh'].sum():.2f}")
            metric_with_help(c3, "Estimated grid import (expensive h)", f"{grid_import:.2f}")
            metric_with_help(c4, "Estimated export/curtailment (kWh)", f"{(grid_export + detail_df['curtailed_kwh'].sum() if not detail_df.empty else 0.0):.2f}")
            pv_totals = weather_ensemble.get("pv_totals_kwh") if isinstance(weather_ensemble.get("pv_totals_kwh"), dict) else {}
            show_uncertainty = all(k in pv_totals for k in ("p10", "p90"))
            if pv_totals:
                st.caption(
                    f"PV forecast P50: {float(pv_totals.get('p50', 0.0)):.2f} kWh"
                    + (
                        f" · Range P10–P90: {float(pv_totals.get('p10', 0.0)):.2f}–{float(pv_totals.get('p90', 0.0)):.2f} kWh"
                        if show_uncertainty
                        else ""
                    )
                )

            tooltip_heading("PV production vs Load (hourly)", CHART_TOOLTIPS["PV production vs Load (hourly)"])
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

            tooltip_heading("Weather inputs used", TABLE_TOOLTIPS["Weather inputs used"])
            with st.expander("Weather inputs used"):
                labels_by_id = {m.get("id"): m.get("label", m.get("id")) for m in weather_models_catalog}
                selected_ids = weather_ensemble.get("selected_models", []) if isinstance(weather_ensemble.get("selected_models"), list) else []
                selected_labels = [str(labels_by_id.get(mid, mid)) for mid in selected_ids]
                st.write(
                    f"PV forecast built from {len(selected_labels)} models: {', '.join(selected_labels)} "
                    f"(method: {weather_ensemble.get('ensemble_method', 'weighted')})"
                )
                st.write(
                    "PV forecast is built from the selected models (ensemble). Weather tables below show: "
                    "(a) Ensemble weather + min/max spread, (b) each model separately."
                )
                st.write(f"Address query: {st.session_state.get('loc_address_query_display', '')}")
                st.write(f"Latitude/Longitude: {float(st.session_state.get('loc_latitude', core.LATITUDE)):.5f}, {float(st.session_state.get('loc_longitude', core.LONGITUDE)):.5f}")
                st.write(f"Timezone: {st.session_state.get('loc_timezone', core.TIMEZONE)}")
                st.write("Hourly columns: temperature_2m, cloud_cover, shortwave_radiation, direct_normal_irradiance, diffuse_radiation, wind_speed_10m")

                weather_column_candidates = {
                    "temperature_2m": st.column_config.NumberColumn(format="%.1f"),
                    "wind_speed_10m": st.column_config.NumberColumn(format="%.1f"),
                    "cloud_cover": st.column_config.NumberColumn(format="%.0f"),
                    "ghi": st.column_config.NumberColumn(format="%.0f"),
                    "dni": st.column_config.NumberColumn(format="%.0f"),
                    "dhi": st.column_config.NumberColumn(format="%.0f"),
                    "shortwave_radiation": st.column_config.NumberColumn(format="%.0f"),
                    "direct_normal_irradiance": st.column_config.NumberColumn(format="%.0f"),
                    "diffuse_radiation": st.column_config.NumberColumn(format="%.0f"),
                    "temperature_2m_min": st.column_config.NumberColumn(format="%.1f"),
                    "temperature_2m_max": st.column_config.NumberColumn(format="%.1f"),
                    "wind_speed_10m_min": st.column_config.NumberColumn(format="%.1f"),
                    "wind_speed_10m_max": st.column_config.NumberColumn(format="%.1f"),
                    "cloud_cover_min": st.column_config.NumberColumn(format="%.0f"),
                    "cloud_cover_max": st.column_config.NumberColumn(format="%.0f"),
                    "shortwave_radiation_min": st.column_config.NumberColumn(format="%.0f"),
                    "shortwave_radiation_max": st.column_config.NumberColumn(format="%.0f"),
                    "direct_normal_irradiance_min": st.column_config.NumberColumn(format="%.0f"),
                    "direct_normal_irradiance_max": st.column_config.NumberColumn(format="%.0f"),
                    "diffuse_radiation_min": st.column_config.NumberColumn(format="%.0f"),
                    "diffuse_radiation_max": st.column_config.NumberColumn(format="%.0f"),
                }

                if isinstance(ensemble_weather_df, pd.DataFrame) and not ensemble_weather_df.empty:
                    st.markdown("**Ensemble weather (P50 + min/max)**")
                    ensemble_weather_display = ensemble_weather_df.copy()
                    ensemble_weather_display.insert(0, "Hour", format_hour_from_index(ensemble_weather_display.index, "%H:00").values)
                    ensemble_weather_display = ensemble_weather_display.head(24).reset_index(drop=True)
                    ensemble_weather_config = build_column_config(ensemble_weather_display, weather_column_candidates)
                    render_modern_table(ensemble_weather_display, ensemble_weather_config)
                elif isinstance(weather_df, pd.DataFrame) and not weather_df.empty:
                    weather_display = weather_df.copy()
                    weather_display.insert(0, "Hour", format_hour_from_index(weather_display.index, "%H:00").values)
                    weather_display = weather_display.head(24).reset_index(drop=True)
                    weather_column_config = build_column_config(weather_display, weather_column_candidates)
                    render_modern_table(weather_display, weather_column_config)
                else:
                    st.info("No data available.")

                if per_model_weather_dfs:
                    with st.expander("Per-model weather (selected models)"):
                        ordered_model_ids = [mid for mid in selected_ids if mid in per_model_weather_dfs]
                        ordered_model_ids.extend(sorted(mid for mid in per_model_weather_dfs if mid not in ordered_model_ids))
                        derived_by_model = weather_ensemble.get("derived_irradiance_by_model", {}) if isinstance(weather_ensemble.get("derived_irradiance_by_model"), dict) else {}
                        tab_labels = []
                        for model_id in ordered_model_ids:
                            model_label = str(labels_by_id.get(model_id, model_id))
                            extras = []
                            if model_id == weather_primary_model_id:
                                extras.append("primary")
                            if bool(derived_by_model.get(model_id)):
                                extras.append("derived irradiance")
                            label = f"{model_label} ({', '.join(extras)})" if extras else model_label
                            tab_labels.append(label)
                        tabs = st.tabs(tab_labels)
                        for tab, model_id in zip(tabs, ordered_model_ids):
                            with tab:
                                model_df = per_model_weather_dfs.get(model_id)
                                if model_df is None or model_df.empty:
                                    st.info("No data available.")
                                    continue
                                model_display = model_df.copy()
                                model_display.insert(0, "Hour", format_hour_from_index(model_display.index, "%H:00").values)
                                model_display = model_display.head(24).reset_index(drop=True)
                                model_column_config = build_column_config(model_display, weather_column_candidates)
                                render_modern_table(model_display, model_column_config)

            combined = pv.join(flows_df[["soc_end_pct", "grid_import_kwh", "grid_export_kwh", "curtailed_kwh"]], how="left")
            combined_display = combined.copy()
            combined_display.insert(0, "Hour", format_hour_from_index(combined_display.index, "%H:%M").values)
            combined_display = combined_display.reset_index(drop=True)
            tooltip_heading("Hourly planning output", TABLE_TOOLTIPS["Hourly planning output"])
            with st.expander("Hourly planning output"):
                hourly_column_config = build_column_config(
                    combined_display,
                    {
                        "pv_total_kwh": st.column_config.NumberColumn(format="%.2f"),
                        "load_kwh": st.column_config.NumberColumn(format="%.2f"),
                        "grid_import_kwh": st.column_config.NumberColumn(format="%.2f"),
                        "grid_export_kwh": st.column_config.NumberColumn(format="%.2f"),
                        "batt_charge_kwh": st.column_config.NumberColumn(format="%.2f"),
                        "batt_discharge_kwh": st.column_config.NumberColumn(format="%.2f"),
                        "soc_pct": st.column_config.NumberColumn(format="%.1f"),
                        "soc_end_pct": st.column_config.NumberColumn(format="%.1f"),
                    },
                )
                render_modern_table(combined_display, hourly_column_config)

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
