from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import planner_core as core
from tariff_time import compute_offpeak_segments, make_summary_lines, parse_hhmm

PLOTLY_DARK = "plotly_dark"

INPUT_TOOLTIPS = {
    "soc_percent": "This is your battery level at 22:00. It matters because charging need is based on how full the battery already is. Example: 35 means the battery starts at 35%.",
    "yesterday_kwh": "This is your total home usage yesterday. It matters because the app uses it to estimate tomorrow's hourly load. Example: if yesterday was 18 kWh, tomorrow's hourly load profile scales to 18 kWh.",
    "buffer_percent": "This adds a safety margin to the target SOC. It matters when forecasts are uncertain. Example: 3% means the target cutoff SOC is increased by 3 percentage points.",
    "performance_ratio": "This is overall PV system efficiency after real-world losses. It matters because lower efficiency means lower expected production. Example: 0.85 means around 85% of ideal output.",
    "inverter_eff": "This is inverter conversion efficiency from DC to AC. It matters because some energy is lost in conversion. Example: 0.97 means around 3% conversion loss.",
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
    "History log": "This table stores one forecast record per run date, with the latest run overwriting older runs for the same date.",
}

RUN_HISTORY_PATH = Path("run_history_log.json")
LOCAL_STATE_DIR = Path("local_state")
API_BASE_URL = os.getenv("PVBP_BACKEND_URL", "http://127.0.0.1:8787")
API_TOKEN_FILE = LOCAL_STATE_DIR / "api_token.txt"

PV_QUALITY_THRESHOLDS = {
    "Excellent": 75,
    "Good": 55,
    "Mixed": 35,
    "Poor": 15,
    "Very low": 0,
}

PV_QUALITY_COLORS = {
    "Excellent": "#2a9d8f",
    "Good": "#52b788",
    "Mixed": "#f4a261",
    "Poor": "#e76f51",
    "Very low": "#d62828",
}


def load_run_history() -> pd.DataFrame:
    if not RUN_HISTORY_PATH.exists():
        return pd.DataFrame(columns=["Date", "AC charge cutoff SOC (%)", "Allowed AC charge power (kW)"])
    try:
        history = json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame(columns=["Date", "AC charge cutoff SOC (%)", "Allowed AC charge power (kW)"])

    return pd.DataFrame(history, columns=["Date", "AC charge cutoff SOC (%)", "Allowed AC charge power (kW)"])


def save_run_history_entry(run_date: dt.date, cutoff_soc_pct: float, allowed_charge_kw: float) -> pd.DataFrame:
    history_df = load_run_history()
    date_str = run_date.isoformat()
    new_entry = {
        "Date": date_str,
        "AC charge cutoff SOC (%)": round(cutoff_soc_pct, 1),
        "Allowed AC charge power (kW)": round(allowed_charge_kw, 2),
    }

    if not history_df.empty:
        history_df = history_df[history_df["Date"] != date_str]
    history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
    history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
    history_df = history_df.dropna(subset=["Date"]).sort_values("Date")
    history_df["Date"] = history_df["Date"].dt.date.astype(str)

    RUN_HISTORY_PATH.write_text(
        history_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    return history_df


@st.cache_data(ttl=3600)
def cached_fetch_weather(lat: float, lon: float, tz: str, tomorrow_iso: str) -> core.ForecastResult:
    loc = core.Location(name="Configured", latitude=lat, longitude=lon)
    return core.fetch_tomorrow_weather(loc, tz=tz)


@st.cache_data(ttl=3600)
def cached_pv_forecast(weather_json: str, lat: float, lon: float, tz: str, config_json: str) -> pd.DataFrame:
    loc = core.Location(name="Configured", latitude=lat, longitude=lon)
    weather_df = pd.read_json(weather_json, orient="split")
    pv = core.build_pv_forecast(weather_df, loc, tz=tz)
    return pv


def weather_hash(df: pd.DataFrame) -> str:
    payload = df.to_json(date_format="iso", orient="split")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lookup_geo_callback() -> None:
    # Widget-bound keys (like loc_latitude) must be changed in a callback,
    # not after the widgets have been created in the same run.
    query = (st.session_state.get("loc_address_query") or "").strip()
    if not query:
        st.session_state["_geo_error"] = "Enter an address first."
        st.session_state.pop("_geo_success", None)
        return

    try:
        loc, tz = core.geocode_address_full(query)
        st.session_state["loc_latitude"] = float(loc.latitude)
        st.session_state["loc_longitude"] = float(loc.longitude)
        if tz:
            st.session_state["loc_timezone"] = str(tz)

        st.session_state["_geo_success"] = f"Found {loc.name}: {loc.latitude:.5f}, {loc.longitude:.5f}"
        st.session_state.pop("_geo_error", None)
    except Exception as exc:
        st.session_state["_geo_error"] = f"Could not find coordinates for '{query}'. {exc}"
        st.session_state.pop("_geo_success", None)


def apply_pending_location_state() -> None:
    pending = st.session_state.pop("_pending_location_state", None)
    if not isinstance(pending, dict):
        return
    st.session_state["loc_address_query"] = str(pending.get("address_query", ""))
    st.session_state["loc_latitude"] = float(pending.get("latitude", core.LATITUDE))
    st.session_state["loc_longitude"] = float(pending.get("longitude", core.LONGITUDE))
    st.session_state["loc_timezone"] = str(pending.get("timezone", core.TIMEZONE))


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


def tooltip_heading(label: str, help_text: str) -> None:
    safe_help = help_text.replace('"', "&quot;")
    st.markdown(
        f"<div class='tooltip-heading'>{label}<span class='info-tooltip' title=\"{safe_help}\">ⓘ</span></div>",
        unsafe_allow_html=True,
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


def compute_clear_sky_reference_kwh(weather_df: pd.DataFrame, loc: core.Location, tz: str | None = None) -> float:
    clear_df = pd.DataFrame(index=weather_df.index)
    clear_df["temp_air_c"] = pd.to_numeric(weather_df.get("temp_air_c"), errors="coerce").fillna(10.0)
    clear_df["wind_speed_ms"] = pd.to_numeric(weather_df.get("wind_speed_ms"), errors="coerce").fillna(1.0).clip(lower=0.0)
    clear_df["cloud_cover_pct"] = 0.0

    _, _, _, _, _, pv_ac_limited_kwh = core.estimate_pv_with_pvlib(clear_df, loc, tz=tz)
    return float(pv_ac_limited_kwh.sum())


def compute_pv_quality(pv_df: pd.DataFrame, clear_kwh: float) -> dict:
    pv_total_kwh = float(pd.to_numeric(pv_df.get("pv_total_kwh", 0.0), errors="coerce").fillna(0.0).sum())
    ratio = float(pv_total_kwh / max(float(clear_kwh), 0.1))
    score = int(min(max(round(100 * ratio), 0), 100))

    label = "Very low"
    for candidate, threshold in PV_QUALITY_THRESHOLDS.items():
        if score >= threshold:
            label = candidate
            break

    return {
        "score": score,
        "label": label,
        "ratio": ratio,
        "pv_total_kwh": pv_total_kwh,
        "color": PV_QUALITY_COLORS.get(label, "#d62828"),
    }


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


def api_get(path: str) -> dict:
    response = requests.get(f"{API_BASE_URL}{path}", headers=api_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def api_put(path: str, payload: dict) -> dict:
    response = requests.put(f"{API_BASE_URL}{path}", headers=api_headers(), json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}{path}", headers=api_headers(), json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def df_from_split(payload: dict) -> pd.DataFrame:
    return pd.read_json(json.dumps(payload), orient="split")


def series_from_split(payload: dict) -> pd.Series:
    frame = df_from_split(payload)
    if "value" in frame.columns:
        return frame["value"]
    return pd.Series(dtype=float)


def run_history_from_backend() -> pd.DataFrame:
    try:
        items = api_get("/v1/results/history?days=30").get("items", [])
    except Exception:
        return pd.DataFrame(columns=["Date", "AC charge cutoff SOC (%)", "Allowed AC charge power (kW)"])
    rows = []
    for item in items:
        metrics = item.get("metrics", {})
        rows.append({
            "Date": item.get("target_date"),
            "AC charge cutoff SOC (%)": round(float(metrics.get("cutoff_soc", 0.0)) * 100.0, 1),
            "Allowed AC charge power (kW)": round(float(metrics.get("charge_kw", 0.0)), 2),
        })
    if not rows:
        return pd.DataFrame(columns=["Date", "AC charge cutoff SOC (%)", "Allowed AC charge power (kW)"])
    history_df = pd.DataFrame(rows)
    history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
    history_df = history_df.dropna(subset=["Date"]).sort_values("Date")
    history_df["Date"] = history_df["Date"].dt.date.astype(str)
    return history_df
def make_chart_pv_load(df: pd.DataFrame, soc: pd.Series, cutoff_soc: float) -> go.Figure:
    working = df.copy()
    if "pv_clipped_kwh" not in working.columns:
        working["pv_clipped_kwh"] = (working["pv_total_unclipped_kwh"] - working["pv_total_kwh"]).clip(lower=0.0)
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
            name="PV total",
            line=dict(width=3, color="#90be6d"),
            customdata=custom_data,
            hovertemplate=hover,
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
    if "loc_address_query" not in st.session_state:
        st.session_state["loc_address_query"] = str(loc_cfg["address_query"])
    if "loc_latitude" not in st.session_state:
        st.session_state["loc_latitude"] = float(loc_cfg["latitude"])
    if "loc_longitude" not in st.session_state:
        st.session_state["loc_longitude"] = float(loc_cfg["longitude"])
    if "loc_timezone" not in st.session_state:
        st.session_state["loc_timezone"] = str(loc_cfg["timezone"])
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
        with st.form("settings_form"):
            st.markdown("#### Location")
            addr_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")
            with addr_col:
                cfg_address_query = st.text_input("Address query", key="loc_address_query")
            with btn_col:
                st.form_submit_button("Lookup", type="primary", on_click=lookup_geo_callback)
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
                cfg_inverter_eff = st.number_input("Inverter efficiency", min_value=0.50, max_value=1.00, value=float(cfg_pv["inverter_eff"]), step=0.01)
            with row4_col2:
                cfg_inverter_ac_kw_limit = st.number_input("Inverter AC limit (kW)", min_value=0.1, value=float(cfg_pv["inverter_ac_kw_limit"]), step=0.1)

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
                new_cfg = {
                    "location": {
                        "use_geocoding": False,
                        "address_query": str(cfg_address_query),
                        "latitude": float(cfg_latitude),
                        "longitude": float(cfg_longitude),
                        "timezone": str(st.session_state["loc_timezone"]),
                    },
                    "tariff": {
                        "peak_grid_price_eur_per_kwh": float(cfg_peak_price_input),
                        "offpeak_grid_price_eur_per_kwh": float(cfg_offpeak_price_input),
                        "injection_grid_price_eur_per_kwh": float(cfg_injection_price_input),
                        "offpeak_windows_by_dow": [[[from_value, to_value]] for from_value, to_value in tariff_inputs],
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
                }
                try:
                    updated = api_put(
                        "/v1/settings",
                        {
                            "config": new_cfg,
                            "nightly_run_time": backend_settings.get("nightly_run_time", "22:00"),
                            "timezone": backend_settings.get("timezone", "Europe/Brussels"),
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
                        "timezone": backend_settings.get("timezone", "Europe/Brussels"),
                        "max_ac_charge_power_kw_default": backend_settings.get("max_ac_charge_power_kw_default", 5.0),
                    },
                )
                st.cache_data.clear()
                st.session_state["_pending_location_state"] = updated["config"]["location"]
                st.session_state["_settings_flash"] = "Reset settings to defaults."
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
                        "timezone": "Europe/Brussels",
                        "max_ac_charge_power_kw_default": float(user_max_ac_kw),
                    },
                )
                st.success("Saved nightly schedule settings.")
            except Exception as exc:
                st.error(f"Could not save nightly settings: {exc}")

    run = st.button("Run forecast", type="primary", help="Click to fetch tomorrow weather and recompute all charts and recommendations.")

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
                },
            )
            result = run_response["result"]
            tomorrow = dt.date.fromisoformat(result["target_date"])
            weather_df = df_from_split(result["weather"])
            pv = df_from_split(result["pv"])
            detail_df = df_from_split(result["detail"])
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

            history_df = run_history_from_backend()

            st.markdown("### Forecast summary")
            c1, c2, c3, c4 = st.columns(4)
            metric_with_help(c1, "Forecast total PV (kWh)", f"{pv['pv_total_kwh'].sum():.2f}")
            metric_with_help(c2, "Forecast total load (kWh)", f"{pv['load_kwh'].sum():.2f}")
            metric_with_help(c3, "Estimated grid import (expensive h)", f"{grid_import:.2f}")
            metric_with_help(c4, "Estimated export/curtailment (kWh)", f"{(grid_export + detail_df['curtailed_kwh'].sum() if not detail_df.empty else 0.0):.2f}")

            tooltip_heading("PV production vs Load (hourly)", CHART_TOOLTIPS["PV production vs Load (hourly)"])
            pv_load_fig = make_chart_pv_load(pv, soc_series, cutoff_soc)
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
                st.write("Source: Open-Meteo ECMWF")
                st.write(f"Address query: {st.session_state.get('loc_address_query', '')}")
                st.write(f"Latitude/Longitude: {float(st.session_state.get('loc_latitude', core.LATITUDE)):.5f}, {float(st.session_state.get('loc_longitude', core.LONGITUDE)):.5f}")
                st.write(f"Timezone: {st.session_state.get('loc_timezone', core.TIMEZONE)}")
                st.write("Hourly columns: temperature_2m, cloud_cover, shortwave_radiation, direct_normal_irradiance, diffuse_radiation, wind_speed_10m")
                weather_display = weather_df.copy()
                weather_display.insert(0, "Hour", format_hour_from_index(weather_display.index, "%H:00").values)
                weather_display = weather_display.reset_index(drop=True)
                st.dataframe(weather_display.head(24), use_container_width=True)

            combined = pv.join(flows_df[["soc_end_pct", "grid_import_kwh", "grid_export_kwh", "curtailed_kwh"]], how="left")
            combined_display = combined.copy()
            combined_display.insert(0, "Hour", format_hour_from_index(combined_display.index, "%H:%M").values)
            combined_display = combined_display.reset_index(drop=True)
            tooltip_heading("Hourly planning output", TABLE_TOOLTIPS["Hourly planning output"])
            with st.expander("Hourly planning output"):
                st.dataframe(combined_display, use_container_width=True)
                st.download_button("Download CSV", combined.to_csv().encode("utf-8"), file_name="pv_battery_plan.csv", mime="text/csv", help="Download the full hourly planning table as CSV.")
                st.download_button("Download JSON", combined.reset_index().to_json(orient="records", date_format="iso", indent=2), file_name="pv_battery_plan.json", mime="application/json", help="Download the full hourly planning table as JSON.")

            tooltip_heading("History log", TABLE_TOOLTIPS["History log"])
            with st.expander("History log"):
                st.dataframe(history_df.reset_index(drop=True), use_container_width=True)

            for warning in result.get("warnings", []):
                st.warning(f"Nightly context warning: {warning}")
    except ImportError as exc:
        st.error(f"Missing dependency: {exc}. Install with: python -m pip install -r requirements.txt")
    except requests.RequestException as exc:
        st.error(f"Backend API call failed: {exc}")
    except core.ExternalServiceError as exc:
        st.error(f"Weather fetch failed: {exc.category}")
        st.info(exc.hint)
    except Exception as exc:
        st.error(f"Could not fetch weather or compute forecast. Please retry in a minute. Details: {exc}")
