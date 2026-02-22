from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import planner_core as core


RUN_HISTORY_PATH = Path("run_history_log.json")


DATA_SOURCE_TOOLTIPS = {
    "soc": "SOC source used in planning. 'manual' means user-entered value.",
    "load": "Load source used in planning. 'manual' means based on user-entered yesterday consumption.",
    "pv": "PV source used in planning. 'forecast' means weather/PV model forecast output.",
}

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
def cached_fetch_weather(lat: float, lon: float, tomorrow_iso: str, elevation_m: float | None = None) -> core.ForecastResult:
    loc = core.Location(name="Configured", latitude=lat, longitude=lon, elevation_m=elevation_m)
    return core.fetch_tomorrow_weather(loc)


@st.cache_data(ttl=3600)
def cached_pv_forecast(weather_json: str, lat: float, lon: float, config_json: str, elevation_m: float | None = None) -> pd.DataFrame:
    loc = core.Location(name="Configured", latitude=lat, longitude=lon, elevation_m=elevation_m)
    weather_df = pd.read_json(weather_json, orient="split")
    pv = core.build_pv_forecast(weather_df, loc)
    return pv


def weather_hash(df: pd.DataFrame) -> str:
    payload = df.to_json(date_format="iso", orient="split")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_hour_from_index(index: pd.Index, fmt: str) -> pd.Series:
    dt_index = pd.to_datetime(index, errors="coerce")
    if isinstance(dt_index, pd.DatetimeIndex):
        return pd.Series(dt_index.strftime(fmt), index=index)
    return pd.Series(index.astype(str), index=index)


def compute_clear_sky_reference_kwh(weather_df: pd.DataFrame, loc: core.Location) -> float:
    clear_df = pd.DataFrame(index=weather_df.index)
    clear_df["temp_air_c"] = pd.to_numeric(weather_df.get("temp_air_c"), errors="coerce").fillna(10.0)
    clear_df["wind_speed_ms"] = pd.to_numeric(weather_df.get("wind_speed_ms"), errors="coerce").fillna(1.0).clip(lower=0.0)
    clear_df["cloud_cover_pct"] = 0.0

    _, _, _, _, _, pv_ac_limited_kwh = core.estimate_pv_with_pvlib(clear_df, loc)
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

