#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# How to run:
# 1) Optional virtual environment:
#    python3 -m venv .venv
#    source .venv/bin/activate
# 2) Install dependencies:
#    pip install -r requirements.txt
#    (or: pip install pandas requests pvlib)
# 3) Run:
#    python3 pv_battery_planner.py
# 4) Update from Git, then run again:
#    git pull
#    python3 pv_battery_planner.py

from __future__ import annotations

import sys
import math
import json
import copy
import os
import tempfile
import warnings
import datetime as dt
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Install pandas: pip install pandas")
    sys.exit(1)

PVLIB_AVAILABLE = False
try:
    import pvlib  # type: ignore
    PVLIB_AVAILABLE = True
except Exception:
    PVLIB_AVAILABLE = False

PRINT_MODE = "compact"

# ============================================================
# CONSTANTS
# ============================================================

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
LEGACY_TILT_KEY = "tilt_" "common_deg"

# --- Locatie
ADDRESS_QUERY = "Voetvolkstraat 14, 1502 Lembeek, Belgium"
USE_GEOCODING = False
LATITUDE = 50.71864
LONGITUDE = 4.21247
TIMEZONE = "Europe/Brussels"

# --- PV-installatie (zadeldak: 2 vlakken)
PANEL_WP = 440
ARRAY_SOUTH_PANELS = 11
ARRAY_EAST_PANELS = 7  # OOST

# Dakhoeken/richting (pvlib conventie: N=0, E=90, S=180, W=270)
TILT_EAST_DEG = 35.0
TILT_SOUTH_DEG = 35.0
AZIMUTH_EAST_DEG = 90.0
AZIMUTH_SOUTH_DEG = 180.0

# Systeemfactoren (tunen met eigen metingen)
PERFORMANCE_RATIO = 0.82
INVERTER_EFF = 0.97
PV_LOSS_MODEL = "split"
PV_IAM_MODEL = "ashrae"
PV_IAM_ASHRAE_B = 0.05
PV_ALBEDO: float | None = 0.20
INVERTER_AC_MODEL = "pvwatts"
PV_CALIBRATION_FACTOR_EAST = 1.00
PV_CALIBRATION_FACTOR_SOUTH = 1.00
PV_GAMMA_PDC = -0.003

# Irradiance consistency controls
IRR_REL_ERR_MEDIAN_THRESHOLD = 0.25
IRR_REL_ERR_POINT_THRESHOLD = 0.35
IRR_BAD_POINT_FRACTION = 0.40
IRR_MIN_GHI_WM2 = 5.0
IRR_REPAIR_METHOD = "disc"  # or "erbs"
IRRADIANCE_HOURLY_MAX_WM2 = 1400.0
IRRADIANCE_HOURLY_EXTREME_WM2 = 2500.0
IRRADIANCE_DAILY_CLEARSKY_FACTOR = 1.35
CLOUD_ATTENUATION_EXPONENT = 3.4
CLOUD_ATTENUATION_WEIGHT = 0.75
CLOUD_TRANSMITTANCE_MIN = 0.08

# Batterij
BATTERY_KWH = 14.0
MIN_SOC_PERCENT = 5.0  # minimum SOC (End-of-discharge SOC)
MIN_SOC = MIN_SOC_PERCENT / 100.0
MAX_CUTOFF_SOC_PERCENT = 95.0
MAX_CUTOFF_SOC = MAX_CUTOFF_SOC_PERCENT / 100.0

# Efficiënties (simulatie)
BATTERY_AC_CHARGE_EFF = 0.93  # net -> batterij
BATTERY_PV_CHARGE_EFF = 0.95  # PV -> batterij
BATTERY_DISCHARGE_EFF = 0.95  # batterij -> load

# Hard safety limit for automatically computed AC charge power
MAX_AC_CHARGE_KW_HARD_LIMIT = 5.0
INVERTER_AC_KW_LIMIT = 4.6
BATTERY_MAX_CHARGE_KW = 5.0
BATTERY_MAX_DISCHARGE_KW = 5.0

# Tariff prices (all-in €/kWh)
PEAK_GRID_PRICE_EUR_PER_KWH = 0.30
OFFPEAK_GRID_PRICE_EUR_PER_KWH = 0.20
INJECTION_GRID_PRICE_EUR_PER_KWH = 0.05

# Load profile (24 uur): verdeling van "gisteren verbruik" over de uren
# Wordt genormaliseerd (som hoeft niet exact 1 te zijn)
LOAD_PROFILE = [
    0.035, 0.030, 0.028, 0.026, 0.026, 0.030,  # 00-05
    0.040, 0.050, 0.045, 0.040, 0.038, 0.038,  # 06-11
    0.040, 0.040, 0.040, 0.042, 0.045, 0.055,  # 12-17
    0.065, 0.070, 0.065, 0.055, 0.045, 0.040   # 18-23
]

ENABLE_INVARIANT_CHECKS = False

DEFAULT_CONFIG = {
    "location": {
        "use_geocoding": USE_GEOCODING,
        "address_query": ADDRESS_QUERY,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "elevation_m": None,
        "timezone": TIMEZONE,
        "address_structured": {
            "street": "",
            "house_number": "",
            "postal_code": "",
            "city": "",
            "country": "",
        },
    },
    "pv": {
        "panel_wp": PANEL_WP,
        "array_south_panels": ARRAY_SOUTH_PANELS,
        "array_east_panels": ARRAY_EAST_PANELS,
        "tilt_east_deg": TILT_EAST_DEG,
        "tilt_south_deg": TILT_SOUTH_DEG,
        "azimuth_east_deg": AZIMUTH_EAST_DEG,
        "azimuth_south_deg": AZIMUTH_SOUTH_DEG,
        "loss_model": PV_LOSS_MODEL,
        "performance_ratio": PERFORMANCE_RATIO,
        "inverter_eff": INVERTER_EFF,
        "pv_loss_model": PV_LOSS_MODEL,
        "iam_model": PV_IAM_MODEL,
        "iam_ashrae_b": PV_IAM_ASHRAE_B,
        "albedo": PV_ALBEDO,
        "inverter_ac_model": INVERTER_AC_MODEL,
        "pv_calibration_factor_east": PV_CALIBRATION_FACTOR_EAST,
        "pv_calibration_factor_south": PV_CALIBRATION_FACTOR_SOUTH,
        "inverter_ac_kw_limit": INVERTER_AC_KW_LIMIT,
    },
    "battery": {
        "battery_kwh": BATTERY_KWH,
        "min_soc_percent": MIN_SOC_PERCENT,
        "max_cutoff_soc_percent": MAX_CUTOFF_SOC_PERCENT,
        "battery_max_charge_kw": BATTERY_MAX_CHARGE_KW,
        "battery_max_discharge_kw": BATTERY_MAX_DISCHARGE_KW,
        "max_ac_charge_kw_hard_limit": MAX_AC_CHARGE_KW_HARD_LIMIT,
    },
    "car_charger": {
        "enabled": False,
        "basic_user": "",
        "basic_pass": "",
    },
    "load_profile": {
        "load_profile_24h": LOAD_PROFILE,
    },
    "tariff": {
        "peak_grid_price_eur_per_kwh": PEAK_GRID_PRICE_EUR_PER_KWH,
        "offpeak_grid_price_eur_per_kwh": OFFPEAK_GRID_PRICE_EUR_PER_KWH,
        "injection_grid_price_eur_per_kwh": INJECTION_GRID_PRICE_EUR_PER_KWH,
        "allow_injection_to_grid": True,
        "max_grid_import_kw": 0.0,
        "optimization_mode": "window_only",
        "night_load_from_battery": False,
        "offpeak_windows_by_dow": [
            [["22:00", "07:00"]],  # Monday
            [["22:00", "07:00"]],  # Tuesday
            [["22:00", "07:00"]],  # Wednesday
            [["22:00", "07:00"]],  # Thursday
            [["22:00", "07:00"]],  # Friday
            [["00:00", "24:00"]],  # Saturday
            [["00:00", "24:00"]],  # Sunday
        ]
    },
    "weather": {
        "store_provider_payloads": False,
        "use_satellite_nowcast_0_6h": False,
        "dynamic_weights": {
            "enabled": False,
            "lookback_days": 30,
            "min_days": 10,
            "db_path": "local_state/planner_history.sqlite",
        }
    },
    "system": {
        "enable_invariant_checks": ENABLE_INVARIANT_CHECKS,
    },
}

EFFECTIVE_CFG = copy.deepcopy(DEFAULT_CONFIG)
OFFPEAK_WINDOWS_BY_DOW: dict[int, list[tuple[str, str]]] = {
    0: [("22:00", "07:00")],
    1: [("22:00", "07:00")],
    2: [("22:00", "07:00")],
    3: [("22:00", "07:00")],
    4: [("22:00", "07:00")],
    5: [("00:00", "24:00")],
    6: [("00:00", "24:00")],
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_CONFIG_STATE_LOCK = threading.RLock()


def deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config_file(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return payload


def migrate_legacy_tilt_config(user_cfg: dict) -> dict:
    migrated_cfg = copy.deepcopy(user_cfg)
    user_pv = migrated_cfg.get("pv")
    if not isinstance(user_pv, dict):
        return migrated_cfg

    if LEGACY_TILT_KEY in user_pv:
        tilt_common = user_pv[LEGACY_TILT_KEY]
        if "tilt_east_deg" not in user_pv:
            user_pv["tilt_east_deg"] = tilt_common
        if "tilt_south_deg" not in user_pv:
            user_pv["tilt_south_deg"] = tilt_common
        del user_pv[LEGACY_TILT_KEY]
    return migrated_cfg


def _validate_time_hhmm(value: str, *, allow_2400_end: bool = False) -> None:
    if not isinstance(value, str):
        raise ValueError(f"Time must be a string, got {value!r}.")
    if value == "24:00":
        if allow_2400_end:
            return
        raise ValueError("'24:00' is only allowed as window end time.")
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format '{value}' (expected HH:MM).")
    hh, mm = parts
    if not (hh.isdigit() and mm.isdigit()):
        raise ValueError(f"Invalid time format '{value}' (expected HH:MM).")
    h = int(hh)
    m = int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid time '{value}' (hour 00-23, minute 00-59).")


def parse_offpeak_windows_by_dow(value) -> dict[int, list[tuple[str, str]]]:
    def parse_day_windows(day_idx: int, day_value) -> list[tuple[str, str]]:
        if not isinstance(day_value, list):
            raise ValueError(f"Invalid tariff windows for {DAY_NAMES[day_idx]} (day index {day_idx}): {day_value!r}")
        parsed_windows: list[tuple[str, str]] = []
        for window in day_value:
            if not (isinstance(window, (list, tuple)) and len(window) == 2):
                raise ValueError(
                    f"Invalid window for {DAY_NAMES[day_idx]} (day index {day_idx}): {window!r}. "
                    "Expected [start, end]."
                )
            start, end = window
            _validate_time_hhmm(start, allow_2400_end=False)
            _validate_time_hhmm(end, allow_2400_end=True)
            if start == end:
                raise ValueError(
                    f"Invalid window for {DAY_NAMES[day_idx]} (day index {day_idx}): {window!r}. "
                    "Start and end cannot be the same."
                )
            parsed_windows.append((start, end))
        return parsed_windows

    by_day: dict[int, list[tuple[str, str]]] = {}
    if isinstance(value, list):
        if len(value) != 7:
            raise ValueError(f"tariff.offpeak_windows_by_dow must contain 7 days, got {len(value)}.")
        for i, day_windows in enumerate(value):
            by_day[i] = parse_day_windows(i, day_windows)
    elif isinstance(value, dict):
        for i in range(7):
            if i not in value and str(i) not in value:
                raise ValueError(f"Missing tariff windows for {DAY_NAMES[i]} (day index {i}).")
            day_windows = value[i] if i in value else value[str(i)]
            by_day[i] = parse_day_windows(i, day_windows)
    else:
        raise ValueError("tariff.offpeak_windows_by_dow must be a list (len 7) or dict with weekday keys.")
    return by_day


def validate_config(cfg: dict) -> None:
    location = cfg["location"]
    pv = cfg["pv"]
    battery = cfg["battery"]
    load_profile = cfg["load_profile"]
    tariff = cfg["tariff"]
    system = cfg.get("system", {})
    weather = cfg.get("weather", {})

    if not (-90.0 <= float(location["latitude"]) <= 90.0):
        raise ValueError("location.latitude must be in [-90, 90].")
    if not (-180.0 <= float(location["longitude"]) <= 180.0):
        raise ValueError("location.longitude must be in [-180, 180].")
    if not (0.0 < float(pv["performance_ratio"]) <= 1.0):
        raise ValueError("pv.performance_ratio must be in (0, 1].")
    if not (0.0 < float(pv["inverter_eff"]) <= 1.0):
        raise ValueError("pv.inverter_eff must be in (0, 1].")
    pv_loss_model = str(pv.get("loss_model", pv.get("pv_loss_model", "split"))).strip().lower()
    if pv_loss_model not in {"split", "combined"}:
        raise ValueError("pv.loss_model (or legacy pv.pv_loss_model) must be either 'split' or 'combined'.")
    if pv_loss_model == "combined" and abs(float(pv["inverter_eff"]) - 1.0) > 1e-9:
        raise ValueError("pv.inverter_eff must be 1.0 when pv.loss_model='combined'.")
    iam_model = str(pv.get("iam_model", "none")).strip().lower()
    if iam_model not in {"none", "ashrae"}:
        raise ValueError("pv.iam_model must be either 'none' or 'ashrae'.")
    iam_ashrae_b_raw = pv.get("iam_ashrae_b", 0.05)
    iam_ashrae_b = 0.05 if iam_ashrae_b_raw is None else float(iam_ashrae_b_raw)
    if not (0.0 <= iam_ashrae_b <= 0.5):
        raise ValueError("pv.iam_ashrae_b must be in [0.0, 0.5].")
    if "albedo" in pv and pv.get("albedo") is not None:
        albedo = float(pv["albedo"])
        if not (0.0 <= albedo <= 1.0):
            raise ValueError("pv.albedo must be in [0.0, 1.0] when set.")
    inverter_ac_model = str(pv.get("inverter_ac_model", "linear")).strip().lower()
    if inverter_ac_model not in {"linear", "pvwatts"}:
        raise ValueError("pv.inverter_ac_model must be either 'linear' or 'pvwatts'.")
    pv_calibration_factor_east_raw = pv.get("pv_calibration_factor_east", 1.0)
    pv_calibration_factor_south_raw = pv.get("pv_calibration_factor_south", 1.0)
    pv_calibration_factor_east = 1.0 if pv_calibration_factor_east_raw is None else float(pv_calibration_factor_east_raw)
    pv_calibration_factor_south = 1.0 if pv_calibration_factor_south_raw is None else float(pv_calibration_factor_south_raw)
    if not (0.7 <= pv_calibration_factor_east <= 1.3):
        raise ValueError("pv.pv_calibration_factor_east must be in [0.7, 1.3].")
    if not (0.7 <= pv_calibration_factor_south <= 1.3):
        raise ValueError("pv.pv_calibration_factor_south must be in [0.7, 1.3].")

    loss_combo = float(pv["performance_ratio"]) * float(pv["inverter_eff"])
    if loss_combo < 0.65 or loss_combo > 0.95:
        warnings.warn(
            "Suspicious PV loss settings: performance_ratio * inverter_eff "
            f"= {loss_combo:.3f} (expected about 0.65..0.95).",
            RuntimeWarning,
            stacklevel=2,
        )
    array_south_panels = int(pv["array_south_panels"])
    array_east_panels = int(pv["array_east_panels"])
    if array_south_panels < 0 or array_east_panels < 0:
        raise ValueError("pv.array_south_panels and pv.array_east_panels must be >= 0.")
    if (array_south_panels + array_east_panels) <= 0:
        raise ValueError("pv.array_*_panels: at least one PV array must have > 0 panels.")
    if int(pv["panel_wp"]) <= 0:
        raise ValueError("pv.panel_wp must be > 0.")
    if not (0.0 <= float(pv["tilt_east_deg"]) <= 90.0):
        raise ValueError("pv.tilt_east_deg must be in [0, 90].")
    if not (0.0 <= float(pv["tilt_south_deg"]) <= 90.0):
        raise ValueError("pv.tilt_south_deg must be in [0, 90].")
    if float(battery["battery_kwh"]) <= 0.0:
        raise ValueError("battery.battery_kwh must be > 0.")

    min_soc_percent = float(battery["min_soc_percent"])
    max_cutoff_soc_percent = float(battery["max_cutoff_soc_percent"])
    if not (0.0 <= min_soc_percent <= 100.0):
        raise ValueError("battery.min_soc_percent must be in [0, 100].")
    if not (0.0 <= max_cutoff_soc_percent <= 100.0):
        raise ValueError("battery.max_cutoff_soc_percent must be in [0, 100].")
    if min_soc_percent > max_cutoff_soc_percent:
        raise ValueError("battery.min_soc_percent must be <= battery.max_cutoff_soc_percent.")

    profile = load_profile["load_profile_24h"]
    if not isinstance(profile, list) or len(profile) != 24:
        raise ValueError("load_profile.load_profile_24h must be a list of 24 numbers.")
    profile_sum = sum(float(v) for v in profile)
    if profile_sum <= 0:
        raise ValueError("load_profile.load_profile_24h must have a positive sum.")

    if "offpeak_windows_by_dow" not in tariff:
        raise ValueError("tariff.offpeak_windows_by_dow is required.")
    if "peak_grid_price_eur_per_kwh" not in tariff:
        raise ValueError("tariff.peak_grid_price_eur_per_kwh is required.")
    if "offpeak_grid_price_eur_per_kwh" not in tariff:
        raise ValueError("tariff.offpeak_grid_price_eur_per_kwh is required.")
    if "injection_grid_price_eur_per_kwh" not in tariff:
        raise ValueError("tariff.injection_grid_price_eur_per_kwh is required.")

    peak_price = float(tariff["peak_grid_price_eur_per_kwh"])
    offpeak_price = float(tariff["offpeak_grid_price_eur_per_kwh"])
    float(tariff["injection_grid_price_eur_per_kwh"])
    if peak_price < 0.0:
        raise ValueError("tariff.peak_grid_price_eur_per_kwh must be >= 0.")
    if offpeak_price < 0.0:
        raise ValueError("tariff.offpeak_grid_price_eur_per_kwh must be >= 0.")

    optimization_mode = str(tariff.get("optimization_mode", "window_only")).strip().lower()
    if optimization_mode not in {"window_only", "price_aware"}:
        raise ValueError("tariff.optimization_mode must be 'window_only' or 'price_aware'.")

    allow_injection_to_grid = tariff.get("allow_injection_to_grid", True)
    if not isinstance(allow_injection_to_grid, bool):
        raise ValueError("tariff.allow_injection_to_grid must be a boolean.")

    max_grid_import_kw_raw = tariff.get("max_grid_import_kw", 0.0)
    try:
        max_grid_import_kw = float(max_grid_import_kw_raw)
    except (TypeError, ValueError):
        raise ValueError("tariff.max_grid_import_kw must be a finite number >= 0.") from None
    if not math.isfinite(max_grid_import_kw) or max_grid_import_kw < 0.0:
        raise ValueError("tariff.max_grid_import_kw must be a finite number >= 0.")

    parse_offpeak_windows_by_dow(tariff["offpeak_windows_by_dow"])

    enable_invariant_checks = system.get("enable_invariant_checks", ENABLE_INVARIANT_CHECKS)
    if not isinstance(enable_invariant_checks, bool):
        raise ValueError("system.enable_invariant_checks must be a boolean.")

    store_provider_payloads = weather.get("store_provider_payloads", False) if isinstance(weather, dict) else False
    if not isinstance(store_provider_payloads, bool):
        raise ValueError("weather.store_provider_payloads must be a boolean.")

    dynamic_weights = weather.get("dynamic_weights", {}) if isinstance(weather, dict) else {}
    enabled = dynamic_weights.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("weather.dynamic_weights.enabled must be a boolean.")
    lookback_days = int(dynamic_weights.get("lookback_days", 30))
    min_days = int(dynamic_weights.get("min_days", 10))
    if lookback_days <= 0:
        raise ValueError("weather.dynamic_weights.lookback_days must be > 0.")
    if min_days <= 0:
        raise ValueError("weather.dynamic_weights.min_days must be > 0.")


def apply_config(cfg: dict) -> None:
    global USE_GEOCODING, ADDRESS_QUERY, LATITUDE, LONGITUDE, TIMEZONE
    global PANEL_WP, ARRAY_SOUTH_PANELS, ARRAY_EAST_PANELS
    global TILT_EAST_DEG, TILT_SOUTH_DEG, AZIMUTH_EAST_DEG, AZIMUTH_SOUTH_DEG
    global PERFORMANCE_RATIO, INVERTER_EFF, PV_LOSS_MODEL, PV_IAM_MODEL, PV_IAM_ASHRAE_B, PV_ALBEDO, INVERTER_AC_MODEL
    global PV_CALIBRATION_FACTOR_EAST, PV_CALIBRATION_FACTOR_SOUTH, INVERTER_AC_KW_LIMIT
    global BATTERY_KWH, MIN_SOC_PERCENT, MAX_CUTOFF_SOC_PERCENT
    global BATTERY_MAX_CHARGE_KW, BATTERY_MAX_DISCHARGE_KW, MAX_AC_CHARGE_KW_HARD_LIMIT
    global LOAD_PROFILE, MIN_SOC, MAX_CUTOFF_SOC, EFFECTIVE_CFG, OFFPEAK_WINDOWS_BY_DOW
    global PEAK_GRID_PRICE_EUR_PER_KWH, OFFPEAK_GRID_PRICE_EUR_PER_KWH, INJECTION_GRID_PRICE_EUR_PER_KWH
    global ENABLE_INVARIANT_CHECKS

    location = cfg["location"]
    pv = cfg["pv"]
    battery = cfg["battery"]
    load_profile = cfg["load_profile"]
    tariff = cfg["tariff"]
    system = cfg.get("system", {})

    USE_GEOCODING = bool(location["use_geocoding"])
    ADDRESS_QUERY = str(location["address_query"])
    LATITUDE = float(location["latitude"])
    LONGITUDE = float(location["longitude"])
    TIMEZONE = str(location["timezone"])

    PANEL_WP = int(pv["panel_wp"])
    ARRAY_SOUTH_PANELS = int(pv["array_south_panels"])
    ARRAY_EAST_PANELS = int(pv["array_east_panels"])
    TILT_EAST_DEG = float(pv["tilt_east_deg"])
    TILT_SOUTH_DEG = float(pv["tilt_south_deg"])
    AZIMUTH_EAST_DEG = float(pv["azimuth_east_deg"])
    AZIMUTH_SOUTH_DEG = float(pv["azimuth_south_deg"])
    PERFORMANCE_RATIO = float(pv["performance_ratio"])
    INVERTER_EFF = 0.97
    PV_LOSS_MODEL = "split"
    PV_IAM_MODEL = "ashrae"
    PV_IAM_ASHRAE_B = 0.05
    PV_ALBEDO = 0.20
    INVERTER_AC_MODEL = "pvwatts"
    base_calibration_factor_east_raw = pv.get("pv_calibration_factor_east", 1.0)
    base_calibration_factor_south_raw = pv.get("pv_calibration_factor_south", 1.0)
    PV_CALIBRATION_FACTOR_EAST = 1.0 if base_calibration_factor_east_raw is None else float(base_calibration_factor_east_raw)
    PV_CALIBRATION_FACTOR_SOUTH = 1.0 if base_calibration_factor_south_raw is None else float(base_calibration_factor_south_raw)
    INVERTER_AC_KW_LIMIT = float(pv["inverter_ac_kw_limit"])

    BATTERY_KWH = float(battery["battery_kwh"])
    MIN_SOC_PERCENT = float(battery["min_soc_percent"])
    MAX_CUTOFF_SOC_PERCENT = float(battery["max_cutoff_soc_percent"])
    BATTERY_MAX_CHARGE_KW = float(battery["battery_max_charge_kw"])
    BATTERY_MAX_DISCHARGE_KW = float(battery["battery_max_discharge_kw"])
    MAX_AC_CHARGE_KW_HARD_LIMIT = float(battery["max_ac_charge_kw_hard_limit"])

    LOAD_PROFILE = [float(v) for v in load_profile["load_profile_24h"]]
    PEAK_GRID_PRICE_EUR_PER_KWH = float(tariff["peak_grid_price_eur_per_kwh"])
    OFFPEAK_GRID_PRICE_EUR_PER_KWH = float(tariff["offpeak_grid_price_eur_per_kwh"])
    INJECTION_GRID_PRICE_EUR_PER_KWH = float(tariff["injection_grid_price_eur_per_kwh"])
    OFFPEAK_WINDOWS_BY_DOW = parse_offpeak_windows_by_dow(tariff["offpeak_windows_by_dow"])
    ENABLE_INVARIANT_CHECKS = bool(system.get("enable_invariant_checks", ENABLE_INVARIANT_CHECKS))
    MIN_SOC = MIN_SOC_PERCENT / 100.0
    MAX_CUTOFF_SOC = MAX_CUTOFF_SOC_PERCENT / 100.0
    EFFECTIVE_CFG = copy.deepcopy(cfg)


def build_effective_config(user_cfg: dict) -> dict:
    migrated_cfg = migrate_legacy_tilt_config(user_cfg)
    merged_cfg = deep_update(copy.deepcopy(DEFAULT_CONFIG), migrated_cfg)
    pv_cfg = merged_cfg.get("pv", {}) if isinstance(merged_cfg.get("pv"), dict) else {}
    if "pv_calibration_factor" in pv_cfg:
        g = float(pv_cfg.get("pv_calibration_factor", 1.0) or 1.0)
        east_rel = float(pv_cfg.get("pv_calibration_factor_east", 1.0) or 1.0)
        south_rel = float(pv_cfg.get("pv_calibration_factor_south", 1.0) or 1.0)
        pv_cfg["pv_calibration_factor_east"] = g * east_rel
        pv_cfg["pv_calibration_factor_south"] = g * south_rel
        del pv_cfg["pv_calibration_factor"]
    loss_model = str(pv_cfg.get("loss_model", pv_cfg.get("pv_loss_model", "split"))).strip().lower()
    pv_cfg["loss_model"] = loss_model
    pv_cfg["pv_loss_model"] = loss_model
    if loss_model == "combined" and abs(float(pv_cfg.get("inverter_eff", 1.0)) - 1.0) > 1e-9:
        warnings.warn(
            "pv.loss_model='combined' already includes inverter losses; forcing pv.inverter_eff=1.0 to avoid double-counting.",
            RuntimeWarning,
            stacklevel=2,
        )
        pv_cfg["inverter_eff"] = 1.0
    validate_config(merged_cfg)
    return merged_cfg


def resolve_pv_loss_multipliers(performance_ratio: float, inverter_eff: float, loss_model: str) -> tuple[float, float]:
    """Return (DC-side PR multiplier, AC-side inverter efficiency multiplier)."""
    mode = str(loss_model).strip().lower()
    if mode == "combined":
        return float(performance_ratio), 1.0
    if mode == "split":
        return float(performance_ratio), float(inverter_eff)
    raise ValueError("loss_model must be 'combined' or 'split'.")


@contextmanager
def applied_config(cfg: dict):
    effective_cfg = build_effective_config(cfg)
    with _CONFIG_STATE_LOCK:
        previous_cfg = get_effective_config()
        apply_config(effective_cfg)
        try:
            yield effective_cfg
        finally:
            apply_config(previous_cfg)


def get_effective_config() -> dict:
    effective_cfg = copy.deepcopy(EFFECTIVE_CFG)
    pv_cfg = effective_cfg.get("pv")
    if isinstance(pv_cfg, dict):
        pv_cfg.pop(LEGACY_TILT_KEY, None)
    return effective_cfg


def save_config_file(cfg: dict, path: Path) -> None:
    cfg_to_save = migrate_legacy_tilt_config(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(cfg_to_save, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def validate_flow_invariants(flows_df: "pd.DataFrame", context: str, *, tol: float = 1e-6) -> None:
    if flows_df.empty:
        return

    soc_lo = (MIN_SOC * 100.0) - tol
    soc_hi = (MAX_CUTOFF_SOC * 100.0) + tol

    def _as_float(row: "pd.Series", col: str) -> float:
        return float(row[col]) if col in row else 0.0

    kwh_cols = [c for c in flows_df.columns if c.endswith("_kwh")]

    for ts, row in flows_df.iterrows():
        ts_text = str(ts)
        soc_start = _as_float(row, "soc_start_pct")
        soc_end = _as_float(row, "soc_end_pct")

        if soc_start < soc_lo or soc_start > soc_hi:
            raise RuntimeError(
                "Invariant failed (SOC bounds): "
                f"context={context}, timestamp={ts_text}, soc_start_pct={soc_start:.6f}, "
                f"expected_range=[{soc_lo:.6f}, {soc_hi:.6f}]"
            )
        if soc_end < soc_lo or soc_end > soc_hi:
            raise RuntimeError(
                "Invariant failed (SOC bounds): "
                f"context={context}, timestamp={ts_text}, soc_end_pct={soc_end:.6f}, "
                f"expected_range=[{soc_lo:.6f}, {soc_hi:.6f}]"
            )

        for col in kwh_cols:
            value = _as_float(row, col)
            if value < -tol:
                raise RuntimeError(
                    "Invariant failed (non-negativity): "
                    f"context={context}, timestamp={ts_text}, {col}={value:.9f}, min_allowed={-tol:.9f}"
                )

        load_kwh = _as_float(row, "load_kwh")
        pv_to_load_kwh = _as_float(row, "pv_to_load_kwh")
        batt_discharge_kwh = _as_float(row, "batt_discharge_kwh")
        grid_import_kwh = _as_float(row, "grid_import_kwh")
        batt_charge_kwh = _as_float(row, "batt_charge_kwh")

        is_pure_night_charge_row = (
            load_kwh <= tol
            and pv_to_load_kwh <= tol
            and batt_discharge_kwh <= tol
            and batt_charge_kwh > tol
        )

        if context in {"full_day", "expensive_hours"} and not is_pure_night_charge_row:
            delivered_from_batt = batt_discharge_kwh * BATTERY_DISCHARGE_EFF
            rhs = pv_to_load_kwh + delivered_from_batt + grid_import_kwh
            diff = load_kwh - rhs
            if abs(diff) > tol:
                raise RuntimeError(
                    "Invariant failed (load balance): "
                    f"context={context}, timestamp={ts_text}, load_kwh={load_kwh:.9f}, "
                    f"pv_to_load_kwh={pv_to_load_kwh:.9f}, delivered_from_batt_kwh={delivered_from_batt:.9f}, "
                    f"grid_import_kwh={grid_import_kwh:.9f}, diff={diff:.9f}, tol={tol:.1e}"
                )

        if context == "night":
            expected_grid_import = batt_charge_kwh / BATTERY_AC_CHARGE_EFF if BATTERY_AC_CHARGE_EFF > 0 else 0.0
            diff = grid_import_kwh - expected_grid_import
            if abs(diff) > tol:
                raise RuntimeError(
                    "Invariant failed (night charging balance): "
                    f"context={context}, timestamp={ts_text}, grid_import_kwh={grid_import_kwh:.9f}, "
                    f"expected_grid_import_kwh={expected_grid_import:.9f}, diff={diff:.9f}, tol={tol:.1e}"
                )


def set_user_config(user_cfg: dict) -> dict:
    merged_cfg = build_effective_config(user_cfg)
    apply_config(merged_cfg)
    return get_effective_config()


USER_CFG = load_config_file(CONFIG_PATH)
EFFECTIVE_CFG = build_effective_config(USER_CFG)
apply_config(EFFECTIVE_CFG)

# Zon-uur indicator (voor "Sun%" in output)
SUN_HOUR_THRESHOLD_KWH = 0.05

# ------------------------------------------------------------
# DALUREN per dag (tariefmodel) (0=ma ... 6=zo).
# Hoog tarief = complement van deze daluren.
# AC laden wordt hiervan afgeleid.
# ------------------------------------------------------------


# ============================================================
# DATA STRUCTS
# ============================================================

@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    elevation_m: float | None = None


@dataclass
class ForecastResult:
    df: "pd.DataFrame"
    sunrise: dt.datetime
    sunset: dt.datetime


@dataclass
class PlannerInputs:
    soc_at_22: float
    yesterday_consumption_kwh: float
    forecast_buffer_soc: float = 0.0


@dataclass
class PlannerOutput:
    location: Location
    tomorrow_date: dt.date
    weather: ForecastResult
    hourly_df: "pd.DataFrame"
    expensive_detail_df: "pd.DataFrame"
    full_day_soc: "pd.Series"
    full_day_flows_df: "pd.DataFrame"
    cutoff_soc: float
    charge_kw: float
    achieved_soc_start: float
    grid_import_expensive_kwh: float
    grid_export_expensive_kwh: float
    curtailed_expensive_kwh: float
    cutoff_note: str
    cutoff_reason: str
    charge_note: str


@dataclass
class ExternalServiceError(RuntimeError):
    service: str
    category: str
    detail: str
    hint: str

    def __str__(self) -> str:
        return f"{self.service} failed [{self.category}]: {self.detail}. Hint: {self.hint}"


def _http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_HTTP_SESSION = _http_session()


def _request_json(
    *,
    service: str,
    url: str,
    params: dict,
    timeout: int = 20,
) -> dict:
    try:
        response = _HTTP_SESSION.get(url, params=params, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise ExternalServiceError(
            service=service,
            category="network_timeout",
            detail=str(exc) or "request timed out",
            hint="Try again; check internet/VPN/firewall.",
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise ExternalServiceError(
            service=service,
            category="network_connection",
            detail=str(exc) or "connection failed",
            hint="Check internet/VPN/firewall and DNS, then retry.",
        ) from exc

    status = response.status_code
    if status == 429:
        raise ExternalServiceError(
            service=service,
            category="http_rate_limited",
            detail="HTTP 429 Too Many Requests",
            hint="Wait 30–60s and retry.",
        )
    if 500 <= status <= 599:
        raise ExternalServiceError(
            service=service,
            category="http_server_error",
            detail=f"HTTP {status}",
            hint="Service temporarily unavailable; retry.",
        )
    if 400 <= status <= 499:
        raise ExternalServiceError(
            service=service,
            category="http_client_error",
            detail=f"HTTP {status}",
            hint="Check query/address format.",
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise ExternalServiceError(
            service=service,
            category="bad_response",
            detail="Invalid JSON payload",
            hint="Retry; if persistent, report the issue.",
        ) from exc

    if not isinstance(data, dict):
        raise ExternalServiceError(
            service=service,
            category="bad_response",
            detail=f"Unexpected payload type: {type(data).__name__}",
            hint="Retry; if persistent, report the issue.",
        )
    return data


# ============================================================
# TIME WINDOW HELPERS (minuten-based, ondersteunt "24:00")
# ============================================================

def to_minutes(hhmm: str) -> int:
    if hhmm == "24:00":
        return 1440
    h, m = map(int, hhmm.split(":"))
    return 60 * h + m


def minute_of_day(t: dt.time) -> int:
    return 60 * t.hour + t.minute  # 0..1439


def normalize_windows(windows: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    # Als er ooit "00:00-24:00" in zit, is de dag volledig dal.
    for s, e in windows:
        if s == "00:00" and e == "24:00":
            return [("00:00", "24:00")]
    return windows


def in_window(tmin: int, start: int, end: int) -> bool:
    # start/end in minuten; end kan 1440 zijn.
    if start < end:
        return start <= tmin < end
    # wraps midnight
    return (tmin >= start) or (tmin < end)


def in_any_window(t: dt.time, windows: List[Tuple[str, str]]) -> bool:
    windows = normalize_windows(windows)
    tmin = minute_of_day(t)
    for s, e in windows:
        smin, emin = to_minutes(s), to_minutes(e)
        if smin == 0 and emin == 1440:
            return True
        if in_window(tmin, smin, emin):
            return True
    return False


def window_duration_hours(s: str, e: str) -> float:
    smin, emin = to_minutes(s), to_minutes(e)
    if smin == 0 and emin == 1440:
        return 24.0
    if smin < emin:
        return (emin - smin) / 60.0
    return ((1440 - smin) + emin) / 60.0


def total_window_hours(windows: List[Tuple[str, str]]) -> float:
    windows = normalize_windows(windows)
    if windows == [("00:00", "24:00")]:
        return 24.0
    return sum(window_duration_hours(s, e) for s, e in windows)


def get_offpeak_windows(for_date: dt.date) -> List[Tuple[str, str]]:
    return normalize_windows(OFFPEAK_WINDOWS_BY_DOW.get(for_date.weekday(), []))


def get_offpeak_windows_for_date(for_date: dt.date, cfg: Optional[dict] = None) -> List[Tuple[str, str]]:
    if cfg is None:
        return get_offpeak_windows(for_date)

    source = cfg.get("offpeak_windows_by_dow")
    if source is None and isinstance(cfg.get("tariff"), dict):
        source = cfg["tariff"].get("offpeak_windows_by_dow")

    if source is None:
        return get_offpeak_windows(for_date)

    parsed = parse_offpeak_windows_by_dow(source)
    return normalize_windows(parsed.get(for_date.weekday(), []))


def get_offpeak_mask_for_date(index: pd.DatetimeIndex, target_date: dt.date, cfg: Optional[dict] = None) -> pd.Series:
    normalized = pd.DatetimeIndex(index)
    if len(normalized) == 0:
        return pd.Series(False, index=normalized, dtype=bool)

    windows = get_offpeak_windows_for_date(target_date, cfg)
    values = [in_any_window(ts.time(), windows) for ts in normalized]
    return pd.Series(values, index=normalized, dtype=bool)


def get_offpeak_mask(index: pd.DatetimeIndex, cfg: Optional[dict] = None) -> pd.Series:
    normalized = pd.DatetimeIndex(index)
    if len(normalized) == 0:
        return pd.Series(False, index=normalized, dtype=bool)

    if isinstance(cfg, dt.date):
        return get_offpeak_mask_for_date(normalized, cfg, None)

    values = [in_any_window(ts.time(), get_offpeak_windows_for_date(ts.date(), cfg)) for ts in normalized]
    return pd.Series(values, index=normalized, dtype=bool)


def get_offpeak_mask_overnight_session(index: pd.DatetimeIndex, charge_date: dt.date, cfg: Optional[dict] = None) -> pd.Series:
    windows = get_offpeak_windows_for_date(charge_date, cfg)
    if len(index) == 0 or not windows:
        return pd.Series(False, index=index, dtype=bool)

    t0 = pd.Timestamp(dt.datetime.combine(charge_date, dt.time(0, 0)), tz=TIMEZONE)
    t1 = t0 + dt.timedelta(days=1)
    t2 = t1 + dt.timedelta(days=1)
    normalized = pd.DatetimeIndex(index)
    mask = pd.Series(False, index=normalized, dtype=bool)

    for start_hhmm, end_hhmm in normalize_windows(windows):
        smin, emin = to_minutes(start_hhmm), to_minutes(end_hhmm)
        if smin == 0 and emin == 1440:
            mask |= (normalized >= t0) & (normalized < t1)
            continue

        start_td = dt.timedelta(minutes=smin)
        end_td = dt.timedelta(minutes=emin)
        if smin < emin:
            mask |= (normalized >= (t0 + start_td)) & (normalized < (t0 + end_td))
            continue

        mask |= (normalized >= (t0 + start_td)) & (normalized < t1)
        mask |= (normalized >= t1) & (normalized < (t1 + end_td))

    mask &= (normalized >= t0) & (normalized < t2)
    return mask


def get_charge_session_index(charge_date: dt.date, cfg: Optional[dict] = None, start_hhmm: str = "22:00") -> pd.DatetimeIndex:
    hh, mm = (int(v) for v in str(start_hhmm).split(":"))
    session_start = pd.Timestamp(dt.datetime.combine(charge_date, dt.time(hh, mm)), tz=TIMEZONE)
    session_end = session_start + dt.timedelta(days=1)
    idx = pd.date_range(session_start, session_end, freq="h", inclusive="left", tz=TIMEZONE)
    mask = get_offpeak_mask(idx, cfg)
    return idx[mask.to_numpy()]


def compute_charging_window_for_target_date(target_date: dt.date, tariff_cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Returns (start_ts, end_ts) tz-aware in TIMEZONE for the off-peak window that should be used
    to charge before target_date.
    """
    target_midnight = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=TIMEZONE)
    charge_date = target_date - dt.timedelta(days=1)
    charge_midnight = pd.Timestamp(dt.datetime.combine(charge_date, dt.time(0, 0)), tz=TIMEZONE)

    def _window_to_interval(base_midnight: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        smin = to_minutes(start_hhmm)
        emin = to_minutes(end_hhmm)
        start_dt = base_midnight + dt.timedelta(minutes=smin)
        if smin == 0 and emin == 1440:
            return start_dt, base_midnight + dt.timedelta(days=1)
        if smin < emin:
            end_dt = base_midnight + dt.timedelta(minutes=emin)
        else:
            end_dt = base_midnight + dt.timedelta(days=1) + dt.timedelta(minutes=emin)
        return start_dt, end_dt

    charge_windows = normalize_windows(get_offpeak_windows_for_date(charge_date, tariff_cfg))
    if ("00:00", "24:00") in charge_windows:
        return charge_midnight, target_midnight

    intervals = [
        _window_to_interval(charge_midnight, start_hhmm, end_hhmm)
        for start_hhmm, end_hhmm in charge_windows
    ]

    crossing_midnight = [(start_ts, end_ts) for start_ts, end_ts in intervals if start_ts < target_midnight < end_ts]
    if crossing_midnight:
        return max(crossing_midnight, key=lambda x: x[0])

    target_windows = normalize_windows(get_offpeak_windows_for_date(target_date, tariff_cfg))
    if target_windows:
        earliest_start_hhmm, earliest_end_hhmm = min(target_windows, key=lambda x: to_minutes(x[0]))
        return _window_to_interval(target_midnight, earliest_start_hhmm, earliest_end_hhmm)

    return target_midnight, target_midnight


def estimate_soc_at_offpeak_start(
    *,
    soc_now_percent: float,
    now_local: dt.datetime,
    offpeak_start: pd.Timestamp,
    effective_daily_kwh: float,
    pv_credit_kwh: float,
    battery_kwh: float,
    min_soc_percent: float,
    used_history: bool,
    pv_credit_available: bool,
) -> tuple[float, float, str, str, dict]:
    """Estimate SOC at off-peak start with TOD load weighting and optional PV credit."""
    now_ts = _to_local_ts(now_local)
    offpeak_ts = _to_local_ts(offpeak_start)
    hours_until = max(0.0, (offpeak_ts - now_ts).total_seconds() / 3600.0)
    load_kwh_window = estimate_window_consumption_kwh(
        start_local=now_local,
        end_local=offpeak_start.to_pydatetime(),
        effective_daily_kwh=float(effective_daily_kwh),
    )
    pv_credit_used = max(0.0, min(float(pv_credit_kwh), load_kwh_window))
    net_kwh = max(0.0, load_kwh_window - pv_credit_used)

    battery_kwh = max(float(battery_kwh), 1e-9)
    energy_now = (float(soc_now_percent) / 100.0) * battery_kwh
    energy_est = energy_now - net_kwh

    energy_floor = (float(min_soc_percent) / 100.0) * battery_kwh
    energy_est = max(energy_floor, energy_est)

    soc_est = (energy_est / battery_kwh) * 100.0
    soc_est = max(0.0, min(100.0, soc_est))

    if hours_until < 2:
        confidence_level = 2
    elif hours_until <= 6:
        confidence_level = 1
    else:
        confidence_level = 0

    start_h = _to_local_ts(now_local).ceil("h")
    end_h = _to_local_ts(offpeak_start).floor("h")
    peak_overlap = False
    if end_h > start_h:
        for ts in pd.date_range(start_h, end_h, freq="h", inclusive="left"):
            if 17 <= ts.hour < 22:
                peak_overlap = True
                break
    if peak_overlap:
        confidence_level = max(0, confidence_level - 1)
    if not used_history:
        confidence_level = max(0, confidence_level - 1)
    daytime = 8 <= now_local.hour < 18 and hours_until > 0
    if daytime and not pv_credit_available:
        confidence_level = 0

    confidence = ["Low", "Medium", "High"][confidence_level]

    history_label = "history" if used_history else "yesterday_only"
    pv_label = "B2pv" if pv_credit_used > 0 else "pv0"
    method = f"{history_label}+tod_weight+{pv_label}"
    debug = {
        "delta_h": float(hours_until),
        "load_kwh_window": float(load_kwh_window),
        "pv_credit_kwh": float(pv_credit_used),
        "effective_daily_kwh_used": float(effective_daily_kwh),
        "used_history": bool(used_history),
        "pv_credit_available": bool(pv_credit_available),
        "peak_overlap": bool(peak_overlap),
    }
    return float(soc_est), float(hours_until), confidence, method, debug


def _tod_weight_for_hour(hour: int) -> float:
    if 0 <= hour < 6:
        return 0.7
    if 6 <= hour < 9:
        return 1.0
    if 9 <= hour < 17:
        return 0.9
    if 17 <= hour < 22:
        return 1.3
    return 1.0


def _to_local_ts(x: object, tz_name: str = TIMEZONE) -> pd.Timestamp:
    """
    Convert x (datetime or Timestamp) into a pandas Timestamp in the app timezone,
    using ZoneInfo to avoid tzinfo-type mismatches (zoneinfo vs dateutil).
    """
    tzinfo = ZoneInfo(tz_name)
    ts = pd.Timestamp(x)
    if ts.tz is None:
        return ts.tz_localize(tzinfo, ambiguous="infer", nonexistent="shift_forward")
    return ts.tz_convert(tzinfo)


def estimate_window_consumption_kwh(
    *,
    start_local: dt.datetime,
    end_local: dt.datetime,
    effective_daily_kwh: float,
) -> float:
    start_h = _to_local_ts(start_local).ceil("h")
    end_h = _to_local_ts(end_local).floor("h")
    if end_h <= start_h:
        return 0.0

    day_weights = sum(_tod_weight_for_hour(hour) for hour in range(24))
    if day_weights <= 0:
        return 0.0
    kwh_per_weight = float(effective_daily_kwh) / day_weights
    window_weight = 0.0
    for ts in pd.date_range(start_h, end_h, freq="h", inclusive="left"):
        window_weight += _tod_weight_for_hour(int(ts.hour))
    return max(0.0, float(window_weight * kwh_per_weight))


def get_charge_session_index_from_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Hourly index of charging decision slots fully inside [start_ts, end_ts).
    """
    start_h = pd.Timestamp(start_ts).ceil("h")
    end_h = pd.Timestamp(end_ts).floor("h")
    if end_h <= start_h:
        return pd.DatetimeIndex([], tz=TIMEZONE)
    return pd.date_range(start_h, end_h, freq="h", inclusive="left")


def get_charge_windows(charge_date: dt.date, cfg: Optional[dict] = None) -> List[Tuple[str, str]]:
    return get_offpeak_windows_for_date(charge_date, cfg)


def import_price_eur_per_kwh(ts: pd.Timestamp, tariff_cfg: dict) -> float:
    offpeak = float(tariff_cfg.get("offpeak_grid_price_eur_per_kwh", 0.0))
    peak = float(tariff_cfg.get("peak_grid_price_eur_per_kwh", 0.0))
    windows = get_offpeak_windows_for_date(ts.date(), tariff_cfg)
    return offpeak if in_any_window(ts.time(), windows) else peak


def tariff_has_meaningful_spread(tariff_cfg: Optional[dict], eps: float = 1e-4) -> bool:
    cfg = tariff_cfg or {}
    peak = float(cfg.get("peak_grid_price_eur_per_kwh", PEAK_GRID_PRICE_EUR_PER_KWH))
    offpeak = float(cfg.get("offpeak_grid_price_eur_per_kwh", OFFPEAK_GRID_PRICE_EUR_PER_KWH))
    return abs(peak - offpeak) > float(eps)


def should_use_battery_for_offpeak_load(tariff_cfg: Optional[dict]) -> bool:
    cfg = tariff_cfg or {}
    mode = str(cfg.get("optimization_mode", "window_only") or "window_only").strip().lower()
    explicit = bool(cfg.get("night_load_from_battery", False))
    # In window_only mode, preserve battery for expensive hours / peak bridging.
    if mode == "window_only":
        return False
    return explicit


def fmt_windows(windows: List[Tuple[str, str]]) -> str:
    if not windows:
        return "none"
    return ", ".join([f"{start}–{end}" for start, end in windows])


def overnight_charge_hours_summary(charge_date: dt.date, cfg: Optional[dict] = None) -> tuple[float, str]:
    target_date = charge_date + dt.timedelta(days=1)
    window_start, window_end = compute_charging_window_for_target_date(target_date, cfg or EFFECTIVE_CFG["tariff"])
    session_idx = get_charge_session_index_from_window(window_start, window_end)
    available_charge_hours = float(len(session_idx))
    return available_charge_hours, f"{window_start.strftime('%H:%M')}–{window_end.strftime('%H:%M')}: {available_charge_hours:.1f}h off-peak"


def complement_windows(windows: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Complement binnen 00:00-24:00. Ondersteunt wrap (bv 22:00-07:00).
    """
    windows = normalize_windows(windows)
    if windows == [("00:00", "24:00")]:
        return []

    intervals: List[Tuple[int, int]] = []
    for s, e in windows:
        smin, emin = to_minutes(s), to_minutes(e)
        if smin == 0 and emin == 1440:
            return []
        if smin < emin:
            intervals.append((smin, emin))
        else:
            intervals.append((smin, 1440))
            intervals.append((0, emin))

    if not intervals:
        return [("00:00", "24:00")]

    intervals.sort()
    merged: List[List[int]] = []
    for a, b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    comp: List[Tuple[int, int]] = []
    prev = 0
    for a, b in merged:
        if a > prev:
            comp.append((prev, a))
        prev = b
    if prev < 1440:
        comp.append((prev, 1440))

    def to_hhmm(m: int) -> str:
        if m == 1440:
            return "24:00"
        return f"{m//60:02d}:{m%60:02d}"

    return [(to_hhmm(a), to_hhmm(b)) for a, b in comp if a != b]


def get_expensive_windows(for_date: dt.date, cfg: Optional[dict] = None) -> List[Tuple[str, str]]:
    # Hoog tarief = complement van daluren
    return complement_windows(get_offpeak_windows_for_date(for_date, cfg))


# ============================================================
# LOAD & PV HELPERS
# ============================================================

def hourly_load_kwh(total_kwh: float) -> List[float]:
    if len(LOAD_PROFILE) != 24:
        raise ValueError("LOAD_PROFILE must have 24 values.")
    s = sum(LOAD_PROFILE)
    if s <= 0:
        raise ValueError("LOAD_PROFILE sum must be > 0.")
    prof = [p / s for p in LOAD_PROFILE]
    return [total_kwh * p for p in prof]


def load_kwh_at(ts: pd.Timestamp, total_kwh: float, dt_h: float = 1.0) -> float:
    loads = hourly_load_kwh(total_kwh)
    return float(loads[ts.hour]) * float(dt_h)


def timestep_hours(index: "pd.DatetimeIndex") -> "pd.Series":
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)
    ts = pd.Series(index=index, data=index)
    dt_h = ts.diff().dt.total_seconds().div(3600.0)
    if len(index) > 1 and pd.notna(dt_h.iloc[1]):
        dt_h.iloc[0] = dt_h.iloc[1]
    else:
        dt_h.iloc[0] = 1.0
    dt_h = pd.to_numeric(dt_h, errors="coerce").fillna(1.0).clip(lower=0.0)
    return dt_h.astype(float)


def build_hourly_load_series(index: "pd.DatetimeIndex", total_kwh: float) -> "pd.Series":
    loads = hourly_load_kwh(total_kwh)
    base = pd.Series([loads[t.hour] for t in index], index=index, dtype=float)
    dt_h = timestep_hours(index)
    weighted = base * dt_h
    wsum = float(weighted.sum())
    if wsum <= 0:
        return pd.Series(0.0, index=index, dtype=float)
    return weighted * (total_kwh / wsum)


def build_cycle_hourly_load_series(
    target_date: dt.date,
    total_consumption_kwh: float,
    tariff_cfg: Optional[dict] = None,
) -> "pd.Series":
    cfg = tariff_cfg or DEFAULT_CONFIG["tariff"]
    windows = normalize_windows(get_offpeak_windows_for_date(target_date, cfg))
    all_day = windows == [("00:00", "24:00")]
    if all_day:
        cycle_start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=TIMEZONE)
    else:
        cycle_start, _ = compute_charging_window_for_target_date(target_date, cfg)
    next_cycle_start = cycle_start + dt.timedelta(hours=24)

    cycle_idx = pd.date_range(cycle_start, next_cycle_start, freq="h", inclusive="left", tz=TIMEZONE)
    cycle_loads = build_hourly_load_series(cycle_idx, total_consumption_kwh)

    if ENABLE_INVARIANT_CHECKS:
        cycle_total = float(cycle_loads.sum())
        if abs(cycle_total - float(total_consumption_kwh)) > 1e-6:
            raise ValueError(
                "Cycle load normalization mismatch: "
                f"sum={cycle_total:.9f} expected={float(total_consumption_kwh):.9f}"
            )
    return cycle_loads


def load_consumption_profile_kwh_per_hour() -> list[float]:
    """Return a normalized 24h load profile for backward compatibility."""
    if len(LOAD_PROFILE) != 24:
        raise ValueError("LOAD_PROFILE must have 24 values.")
    profile = [float(v) for v in LOAD_PROFILE]
    total = float(sum(profile))
    if total <= 0:
        raise ValueError("LOAD_PROFILE sum must be > 0.")
    return [v / total for v in profile]


def build_consumption_forecast(
    profile_kwh_per_hour: list[float],
    total_kwh: float,
    target_date: dt.date,
    tz: str,
) -> "pd.Series":
    """Build a 24h consumption series for target_date using a supplied profile."""
    if len(profile_kwh_per_hour) != 24:
        raise ValueError("profile_kwh_per_hour must contain 24 values.")
    profile = [float(v) for v in profile_kwh_per_hour]
    profile_sum = float(sum(profile))
    if profile_sum <= 0:
        raise ValueError("profile_kwh_per_hour must have a positive sum.")

    start = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=tz)
    end = start + dt.timedelta(days=1)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    normalized = [v / profile_sum for v in profile]
    values = [float(total_kwh) * normalized[ts.hour] for ts in idx]
    return pd.Series(values, index=idx, dtype=float)


def parse_soc_input(raw: str) -> float:
    val = raw.strip().replace(",", ".")
    if not val:
        raise ValueError("SOC input is required")

    soc_percent = float(val)
    if 0.0 <= soc_percent <= 100.0 and not (0.0 < soc_percent < 1.0):
        return soc_percent / 100.0
    raise ValueError("Expected percent 0..100, not fraction 0..1. Example: use 20 not 0.20.")


def ask_required_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("Value is required. Please try again.")
            continue
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("Invalid number. Please try again.")


def ask_required_soc(prompt: str) -> float:
    while True:
        raw = input(prompt)
        try:
            return parse_soc_input(raw)
        except ValueError:
            print("Invalid SOC. Expected percent 0..100, not fraction 0..1. Example: use 20 not 0.20.")


def dc_kwp(panels: int) -> float:
    return (panels * PANEL_WP) / 1000.0



def add_load_and_surplus_columns(
    df: "pd.DataFrame",
    total_consumption_kwh: float,
) -> "pd.DataFrame":
    out = df.copy()
    load_series = build_hourly_load_series(out.index, total_consumption_kwh)
    out["load_kwh"] = load_series.astype(float)
    pv_total = out["pv_total_kwh"].fillna(0.0)
    load = out["load_kwh"].fillna(0.0)
    out["pv_surplus_kwh"] = (pv_total - load).clip(lower=0.0)
    out["pv_deficit_kwh"] = (load - pv_total).clip(lower=0.0)
    return out


def compute_euro_savings_no_battery_vs_plan(
    pv_df: pd.DataFrame,
    flows_df: pd.DataFrame,
    soc_at_22: float,
    charge_kw: float,
    cutoff_soc: float,
    today_date: dt.date,
    tomorrow_date: dt.date,
    total_consumption_kwh: float,
    tariff_cfg: dict,
) -> dict:
    """
    Returns cycle totals for a fixed 24h operational horizon from off-peak start, plus tomorrow detail.
    """
    inj = float(tariff_cfg.get("injection_grid_price_eur_per_kwh", 0.0))

    tomorrow_start = pd.Timestamp(dt.datetime.combine(tomorrow_date, dt.time(0, 0)), tz=TIMEZONE)
    tomorrow_end = tomorrow_start + dt.timedelta(days=1)

    idx_tomorrow = pd.date_range(tomorrow_start, tomorrow_end, freq="h", inclusive="left", tz=TIMEZONE)
    window_start, window_end = compute_charging_window_for_target_date(tomorrow_date, tariff_cfg)
    windows_tom = normalize_windows(get_offpeak_windows_for_date(tomorrow_date, tariff_cfg))
    all_day = windows_tom == [("00:00", "24:00")]
    if all_day:
        cycle_start = tomorrow_start
    else:
        cycle_start = window_start
    cycle_end = cycle_start + dt.timedelta(hours=24)

    idx_cycle = pd.date_range(cycle_start, cycle_end, freq="h", inclusive="left", tz=TIMEZONE)
    if ENABLE_INVARIANT_CHECKS:
        assert len(idx_cycle) == 24, "Cycle horizon must always span 24 hourly slots."

    pv_baseline_col_used = "pv_total_decision_kwh" if "pv_total_decision_kwh" in pv_df.columns else "pv_total_kwh"
    pv_plan_col_used = "pv_total_decision_kwh" if "pv_total_decision_kwh" in pv_df.columns else "pv_total_kwh"

    dt_h_cycle = timestep_hours(idx_cycle)
    pv_cycle = pd.to_numeric(pv_df[pv_baseline_col_used].reindex(idx_cycle), errors="coerce").fillna(0.0).astype(float)
    load_cycle = pd.Series(
        [load_kwh_at(ts, total_consumption_kwh, float(dt_h_cycle.loc[ts])) for ts in idx_cycle],
        index=idx_cycle,
        dtype=float,
    )
    base_import_cycle = (load_cycle - pv_cycle).clip(lower=0.0)
    base_export_cycle = (pv_cycle - load_cycle).clip(lower=0.0)
    base_price_cycle = pd.Series([import_price_eur_per_kwh(ts, tariff_cfg) for ts in idx_cycle], index=idx_cycle)
    grid_only_import_cycle = load_cycle
    grid_only_cost_cycle_series = grid_only_import_cycle * base_price_cycle
    base_cost_cycle = base_import_cycle * base_price_cycle - base_export_cycle * inj

    plan_import_cycle = pd.Series(0.0, index=idx_cycle, dtype=float)
    plan_export_cycle = pd.Series(0.0, index=idx_cycle, dtype=float)
    plan_soc_end_cycle = pd.Series(np.nan, index=idx_cycle, dtype=float)

    idx_pre = idx_cycle[idx_cycle < tomorrow_start]
    idx_post = idx_cycle[idx_cycle >= tomorrow_start]

    night_df = simulate_night_charging_series(
        soc_at_22,
        charge_kw,
        cutoff_soc,
        session_start=window_start,
        session_end=window_end,
        total_consumption_kwh=total_consumption_kwh,
        tariff_cfg=tariff_cfg,
    )
    if len(idx_pre) > 0:
        plan_import_cycle.loc[idx_pre] = night_df["grid_import_kwh"].reindex(idx_pre).fillna(0.0).astype(float)
        if "grid_export_kwh" in night_df.columns:
            plan_export_cycle.loc[idx_pre] = night_df["grid_export_kwh"].reindex(idx_pre).fillna(0.0).astype(float)
        if "soc_end_pct" in night_df.columns:
            plan_soc_end_cycle.loc[idx_pre] = pd.to_numeric(
                night_df["soc_end_pct"].reindex(idx_pre), errors="coerce"
            )

    if len(idx_post) > 0:
        plan_import_cycle.loc[idx_post] = flows_df["grid_import_kwh"].reindex(idx_post).fillna(0.0).astype(float)
        plan_export_cycle.loc[idx_post] = flows_df["grid_export_kwh"].reindex(idx_post).fillna(0.0).astype(float)
        if "soc_end_pct" in flows_df.columns:
            plan_soc_end_cycle.loc[idx_post] = pd.to_numeric(
                flows_df["soc_end_pct"].reindex(idx_post), errors="coerce"
            )

    plan_cost_cycle_cash_series = plan_import_cycle * base_price_cycle - plan_export_cycle * inj
    plan_cost_cycle_cash = plan_cost_cycle_cash_series
    hourly_savings_cycle = (base_cost_cycle - plan_cost_cycle_cash).reindex(idx_cycle).fillna(0.0)
    hourly_benefit_vs_grid_only_cycle_cash = (
        grid_only_cost_cycle_series - plan_cost_cycle_cash_series
    ).reindex(idx_cycle).fillna(0.0)

    dt_h_tom = timestep_hours(idx_tomorrow)
    pv_tom = pd.to_numeric(pv_df[pv_baseline_col_used].reindex(idx_tomorrow), errors="coerce").fillna(0.0).astype(float)
    load_tom = pd.Series(
        [load_kwh_at(ts, total_consumption_kwh, float(dt_h_tom.loc[ts])) for ts in idx_tomorrow],
        index=idx_tomorrow,
        dtype=float,
    )

    if ENABLE_INVARIANT_CHECKS:
        small_eps = 1e-6
        assert abs(float(load_cycle.sum()) - float(total_consumption_kwh)) < small_eps, (
            "Cycle load mismatch: expected total_consumption_kwh over cycle horizon."
        )
    base_import_tom = (load_tom - pv_tom).clip(lower=0.0)
    base_export_tom = (pv_tom - load_tom).clip(lower=0.0)
    price_tom = pd.Series([import_price_eur_per_kwh(ts, tariff_cfg) for ts in idx_tomorrow], index=idx_tomorrow)
    base_cost_tom = base_import_tom * price_tom - base_export_tom * inj

    plan_import_tom = flows_df["grid_import_kwh"].reindex(idx_tomorrow).fillna(0.0).astype(float)
    plan_export_tom = flows_df["grid_export_kwh"].reindex(idx_tomorrow).fillna(0.0).astype(float)
    plan_cost_tom = plan_import_tom * price_tom - plan_export_tom * inj
    grid_only_cost_tom = load_tom * price_tom
    isystem_cost_tom_cash = plan_cost_tom
    hourly_benefit_vs_grid_only_tomorrow_cash = (grid_only_cost_tom - isystem_cost_tom_cash).reindex(idx_tomorrow).fillna(0.0)

    hourly_savings = (base_cost_tom - plan_cost_tom).reindex(idx_tomorrow).fillna(0.0)

    baseline_cycle = float(base_cost_cycle.sum())
    plan_cycle_cash = float(plan_cost_cycle_cash.sum())
    plan_cycle_cash_recomputed = float(plan_cost_cycle_cash_series.sum())
    grid_only_cost_eur_cycle = float(grid_only_cost_cycle_series.sum())
    baseline_tom = float(base_cost_tom.sum())
    plan_tom = float(plan_cost_tom.sum())
    grid_only_cost_eur_tomorrow = float(grid_only_cost_tom.sum())
    isystem_cost_eur_tomorrow_cash = float(isystem_cost_tom_cash.sum())

    cycle_start_soc = max(0.0, min(1.0, float(soc_at_22)))
    cycle_end_soc: float | None = None
    cycle_terminal_row_ts = cycle_end - dt.timedelta(hours=1)
    if cycle_terminal_row_ts in plan_soc_end_cycle.index:
        try:
            soc_pct = float(plan_soc_end_cycle.loc[cycle_terminal_row_ts])
        except Exception:
            soc_pct = float("nan")
        if np.isfinite(soc_pct):
            cycle_end_soc = max(0.0, min(1.0, soc_pct / 100.0))

    terminal_battery_value_eur_cycle = 0.0
    savings_cycle_terminal_value_applied = False
    cycle_stored_energy_delta_kwh = 0.0
    cycle_soc_delta_pct = 0.0
    if cycle_end_soc is not None:
        cycle_stored_energy_delta_kwh = (cycle_end_soc - cycle_start_soc) * BATTERY_KWH
        cycle_soc_delta_pct = (cycle_end_soc - cycle_start_soc) * 100.0
        replacement_price = float(import_price_eur_per_kwh(cycle_end, tariff_cfg))
        charge_eff = float(BATTERY_AC_CHARGE_EFF)
        replacement_cost_per_stored_kwh = replacement_price / charge_eff if charge_eff > 1e-9 else 0.0
        terminal_battery_value_eur_cycle = cycle_stored_energy_delta_kwh * replacement_cost_per_stored_kwh
        savings_cycle_terminal_value_applied = True

    plan_cycle = plan_cycle_cash - terminal_battery_value_eur_cycle
    savings_cycle = baseline_cycle - plan_cycle
    isystem_cost_eur_cycle = plan_cycle
    benefit_vs_grid_only_eur_cycle = grid_only_cost_eur_cycle - isystem_cost_eur_cycle
    benefit_vs_grid_only_eur_tomorrow_cash = grid_only_cost_eur_tomorrow - isystem_cost_eur_tomorrow_cash

    hourly_savings_list = [float(hourly_savings.loc[ts]) for ts in idx_tomorrow]
    hourly_benefit_vs_grid_only_tomorrow_cash_list = [
        float(hourly_benefit_vs_grid_only_tomorrow_cash.loc[ts]) for ts in idx_tomorrow
    ]
    if len(hourly_savings_list) != 24:
        hourly_savings_list = (hourly_savings_list + [0.0] * 24)[:24]
    if len(hourly_benefit_vs_grid_only_tomorrow_cash_list) != 24:
        hourly_benefit_vs_grid_only_tomorrow_cash_list = (hourly_benefit_vs_grid_only_tomorrow_cash_list + [0.0] * 24)[:24]

    def _safe_numeric(value: float) -> float:
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            return float(value)
        return 0.0

    baseline_cycle = _safe_numeric(baseline_cycle)
    plan_cycle_cash = _safe_numeric(plan_cycle_cash)
    plan_cycle_cash_recomputed = _safe_numeric(plan_cycle_cash_recomputed)
    plan_cycle = _safe_numeric(plan_cycle)
    isystem_cost_eur_cycle = _safe_numeric(isystem_cost_eur_cycle)
    grid_only_cost_eur_cycle = _safe_numeric(grid_only_cost_eur_cycle)
    benefit_vs_grid_only_eur_cycle = _safe_numeric(benefit_vs_grid_only_eur_cycle)
    baseline_tom = _safe_numeric(baseline_tom)
    plan_tom = _safe_numeric(plan_tom)
    grid_only_cost_eur_tomorrow = _safe_numeric(grid_only_cost_eur_tomorrow)
    isystem_cost_eur_tomorrow_cash = _safe_numeric(isystem_cost_eur_tomorrow_cash)
    benefit_vs_grid_only_eur_tomorrow_cash = _safe_numeric(benefit_vs_grid_only_eur_tomorrow_cash)
    terminal_battery_value_eur_cycle = _safe_numeric(terminal_battery_value_eur_cycle)
    cycle_stored_energy_delta_kwh = _safe_numeric(cycle_stored_energy_delta_kwh)
    cycle_soc_delta_pct = _safe_numeric(cycle_soc_delta_pct)
    cycle_start_soc_pct = _safe_numeric(cycle_start_soc * 100.0)
    cycle_end_soc_pct = _safe_numeric((cycle_end_soc if cycle_end_soc is not None else cycle_start_soc) * 100.0)
    savings_cycle = _safe_numeric(savings_cycle)
    savings_tom = _safe_numeric(float(baseline_tom - plan_tom))
    hourly_savings_cycle_list = [float(hourly_savings_cycle.loc[ts]) for ts in idx_cycle]
    hourly_benefit_vs_grid_only_cycle_cash_list = [
        float(hourly_benefit_vs_grid_only_cycle_cash.loc[ts]) for ts in idx_cycle
    ]
    if len(hourly_savings_cycle_list) != 24:
        hourly_savings_cycle_list = (hourly_savings_cycle_list + [0.0] * 24)[:24]
    if len(hourly_benefit_vs_grid_only_cycle_cash_list) != 24:
        hourly_benefit_vs_grid_only_cycle_cash_list = (hourly_benefit_vs_grid_only_cycle_cash_list + [0.0] * 24)[:24]
    hourly_savings_cycle_hour_labels = [ts.strftime("%H:%M") for ts in idx_cycle][:24]
    if len(hourly_savings_cycle_hour_labels) != 24:
        hourly_savings_cycle_hour_labels = [f"{h:02d}:00" for h in range(24)]

    if ENABLE_INVARIANT_CHECKS:
        assert len(hourly_benefit_vs_grid_only_tomorrow_cash_list) == 24
        assert len(hourly_benefit_vs_grid_only_cycle_cash_list) == 24

    result = {
        "baseline_cost_eur_total": baseline_cycle,
        "plan_cost_eur_total": plan_cycle,
        "savings_eur_total": savings_cycle,
        "baseline_cost_eur_cycle": baseline_cycle,
        "plan_cost_eur_cycle": plan_cycle,
        "savings_eur_cycle": savings_cycle,
        "baseline_cost_eur_cycle_cash": baseline_cycle,
        "plan_cost_eur_cycle_cash": plan_cycle_cash,
        "grid_only_cost_eur_cycle": grid_only_cost_eur_cycle,
        "isystem_cost_eur_cycle_cash": plan_cycle_cash,
        "isystem_cost_eur_cycle": isystem_cost_eur_cycle,
        "benefit_vs_grid_only_eur_cycle": benefit_vs_grid_only_eur_cycle,
        "terminal_battery_value_eur_cycle": terminal_battery_value_eur_cycle,
        "plan_cost_eur_cycle_adjusted": plan_cycle,
        "baseline_cost_eur_tomorrow": baseline_tom,
        "plan_cost_eur_tomorrow": plan_tom,
        "savings_eur_tomorrow": savings_tom,
        "grid_only_cost_eur_tomorrow": grid_only_cost_eur_tomorrow,
        "isystem_cost_eur_tomorrow_cash": isystem_cost_eur_tomorrow_cash,
        "benefit_vs_grid_only_eur_tomorrow_cash": benefit_vs_grid_only_eur_tomorrow_cash,
        "hourly_savings_eur_cycle": hourly_savings_cycle_list,
        "hourly_benefit_vs_grid_only_eur_cycle_cash": hourly_benefit_vs_grid_only_cycle_cash_list,
        "hourly_benefit_cycle_hour_labels": hourly_savings_cycle_hour_labels,
        "hourly_savings_cycle_hour_labels": hourly_savings_cycle_hour_labels,
        "hourly_savings_eur_tomorrow": hourly_savings_list,
        "hourly_benefit_vs_grid_only_eur_tomorrow_cash": hourly_benefit_vs_grid_only_tomorrow_cash_list,
        "savings_horizon_kind": "offpeak_cycle",
        "savings_horizon_start_iso": cycle_start.isoformat(),
        "savings_horizon_end_iso": cycle_end.isoformat(),
        "savings_cycle_inventory_adjusted": True,
        "savings_cycle_terminal_value_applied": bool(savings_cycle_terminal_value_applied),
        "savings_horizon_label": (
            "off-peak start -> next off-peak start"
        ),
        "savings_horizon_detail": (
            "Cycle (off-peak start → next off-peak start): "
            f"{cycle_start.strftime('%H:%M')} → {cycle_end.strftime('%H:%M')}"
        ),
        "savings_hourly_detail_scope": "tomorrow_00_24",
        "savings_cycle_start_soc_percent_used": float(soc_at_22 * 100.0),
        "cycle_start_soc_pct": cycle_start_soc_pct,
        "cycle_end_soc_pct": cycle_end_soc_pct,
        "cycle_soc_delta_pct": cycle_soc_delta_pct,
        "cycle_stored_energy_delta_kwh": cycle_stored_energy_delta_kwh,
        "pv_baseline_col_used": str(pv_baseline_col_used),
        "pv_plan_col_used": str(pv_plan_col_used),
        "savings_night_load_from_battery_used": bool(should_use_battery_for_offpeak_load(tariff_cfg)),
        "savings_cycle_window_start_local": cycle_start.isoformat(),
        "savings_cycle_window_end_local": cycle_end.isoformat(),
    }

    diagnostics_defaults = {
        "baseline_cost_eur_cycle": 0.0,
        "plan_cost_eur_cycle_cash": 0.0,
        "terminal_battery_value_eur_cycle": 0.0,
        "plan_cost_eur_cycle": 0.0,
        "savings_eur_cycle": 0.0,
        "grid_only_cost_eur_cycle": 0.0,
        "isystem_cost_eur_cycle_cash": 0.0,
        "isystem_cost_eur_cycle": 0.0,
        "benefit_vs_grid_only_eur_cycle": 0.0,
        "grid_only_cost_eur_tomorrow": 0.0,
        "isystem_cost_eur_tomorrow_cash": 0.0,
        "benefit_vs_grid_only_eur_tomorrow_cash": 0.0,
        "hourly_benefit_vs_grid_only_eur_cycle_cash": [0.0] * 24,
        "hourly_benefit_vs_grid_only_eur_tomorrow_cash": [0.0] * 24,
        "hourly_benefit_cycle_hour_labels": [f"{h:02d}:00" for h in range(24)],
        "cycle_start_soc_pct": 0.0,
        "cycle_end_soc_pct": 0.0,
        "cycle_soc_delta_pct": 0.0,
        "cycle_stored_energy_delta_kwh": 0.0,
        "savings_cycle_terminal_value_applied": False,
        "savings_horizon_label": "off-peak start -> next off-peak start",
    }
    for key, default_value in diagnostics_defaults.items():
        if key not in result or result[key] is None:
            result[key] = default_value
    return result


def quick_sanity_checks() -> None:
    try:
        assert ARRAY_EAST_PANELS >= 0 and ARRAY_SOUTH_PANELS >= 0
        assert (ARRAY_EAST_PANELS + ARRAY_SOUTH_PANELS) > 0
        assert 0 < PERFORMANCE_RATIO <= 1
        assert 0 < INVERTER_EFF <= 1
        assert PV_LOSS_MODEL in {"split", "combined"}
        assert 0.7 <= PV_CALIBRATION_FACTOR_EAST <= 1.3
        assert 0.7 <= PV_CALIBRATION_FACTOR_SOUTH <= 1.3
        assert 0 < BATTERY_AC_CHARGE_EFF <= 1
        assert 0 < BATTERY_PV_CHARGE_EFF <= 1
        assert 0 < BATTERY_DISCHARGE_EFF <= 1
        assert BATTERY_MAX_CHARGE_KW > 0
        assert BATTERY_MAX_DISCHARGE_KW > 0
        assert 0 <= MIN_SOC < 1
        assert MIN_SOC <= MAX_CUTOFF_SOC <= 1
        assert len(LOAD_PROFILE) == 24
        assert sum(LOAD_PROFILE) > 0
    except AssertionError as exc:
        raise SystemExit(
            "Sanity check failed: verify panel counts, efficiencies, SOC limits and LOAD_PROFILE config."
        ) from exc


# ============================================================
# OPEN-METEO (geocode + weerdata)
# ============================================================

def geocode_address(query: str) -> Location:
    loc, _ = geocode_address_full(query)
    return loc


def geocode_address_full(query: str) -> tuple[Location, str | None]:
    service = "open-meteo-geocode"
    url = "https://geocoding-api.open-meteo.com/v1/search"
    candidates = _build_geocode_query_candidates(query)
    res = None
    request_errors: list[ExternalServiceError] = []

    for candidate in candidates:
        params = {"name": candidate, "count": 1, "language": "en", "format": "json"}
        try:
            data = _request_json(service=service, url=url, params=params)
        except ExternalServiceError as exc:
            request_errors.append(exc)
            continue

        results = data.get("results")
        if results:
            res = results[0]
            break

    if res is None:
        if request_errors and len(request_errors) == len(candidates):
            last = request_errors[-1]
            hint_query = candidates[-1] if len(candidates) > 1 else "Lembeek, Belgium"
            raise ExternalServiceError(
                service=service,
                category=last.category,
                detail=f"{last.detail}. Geocoding failed for '{query}'. Try '{hint_query}'.",
                hint=last.hint,
            ) from last
        hint = candidates[-1] if len(candidates) > 1 else "Lembeek, Belgium"
        raise ExternalServiceError(
            service=service,
            category="bad_response",
            detail=f"Geocoding returned no results for '{query}'.",
            hint=f"Try '{hint}'.",
        )

    try:
        loc = Location(
            name=res.get("name", query),
            latitude=float(res["latitude"]),
            longitude=float(res["longitude"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalServiceError(
            service=service,
            category="bad_response",
            detail="Geocoding response is missing expected location fields.",
            hint="Retry; if persistent, report the issue.",
        ) from exc
    return loc, res.get("timezone")


def compose_address(street: str, house_number: str, postal_code: str, city: str, country: str) -> str:
    street_clean = " ".join(str(street or "").split())
    house_number_clean = " ".join(str(house_number or "").split())
    postal_code_clean = " ".join(str(postal_code or "").split())
    city_clean = " ".join(str(city or "").split())
    country_clean = " ".join(str(country or "").split())

    line1 = " ".join(part for part in [street_clean, house_number_clean] if part).strip()
    line2 = " ".join(part for part in [postal_code_clean, city_clean] if part).strip()

    return ", ".join(part for part in [line1, line2, country_clean] if part)


def resolve_location_from_structured_address(
    street: str,
    house_number: str,
    postal_code: str,
    city: str,
    country: str,
) -> dict:
    address_query = compose_address(street, house_number, postal_code, city, country)
    if not address_query:
        raise RuntimeError("Please provide at least Street/City/Country before lookup.")

    try:
        loc, timezone = geocode_address_full(address_query)
    except Exception as exc:
        raise RuntimeError(f"Could not resolve '{address_query}'. {exc}") from exc

    timezone_use = str(timezone or TIMEZONE)
    try:
        ZoneInfo(timezone_use)
    except Exception:
        timezone_use = TIMEZONE

    return {
        "address_query": address_query,
        "latitude": float(loc.latitude),
        "longitude": float(loc.longitude),
        "timezone": timezone_use,
        "address_structured": {
            "street": " ".join(str(street or "").split()),
            "house_number": " ".join(str(house_number or "").split()),
            "postal_code": " ".join(str(postal_code or "").split()),
            "city": " ".join(str(city or "").split()),
            "country": " ".join(str(country or "").split()),
        },
    }


def _build_geocode_query_candidates(query: str) -> list[str]:
    cleaned = query.strip()
    if not cleaned:
        return [query]

    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    candidates = [cleaned]

    if parts:
        first_no_number = " ".join(tok for tok in parts[0].split() if not any(ch.isdigit() for ch in tok)).strip()
        if first_no_number and first_no_number != parts[0]:
            candidates.append(", ".join([first_no_number, *parts[1:]]))

    if len(parts) >= 2:
        locality_tokens = [tok for tok in parts[-2].split() if not any(ch.isdigit() for ch in tok)]
        if locality_tokens:
            locality = " ".join(locality_tokens)
            candidates.append(", ".join([locality, parts[-1]]))

    if len(parts) >= 3:
        candidates.append(", ".join(parts[-2:]))

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _fetch_weather_payload(loc: Location, target_date: dt.date, tz_use: str) -> ForecastResult:
    service = "open-meteo-weather"
    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "timezone": tz_use,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timeformat": "iso8601",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "hourly": ",".join([
            "temperature_2m",
            "cloud_cover",
            "shortwave_radiation",
            "direct_normal_irradiance",
            "diffuse_radiation",
            "wind_speed_10m",
        ]),
        "daily": ",".join(["sunrise", "sunset"]),
    }

    data = _request_json(service=service, url=url, params=params)

    hourly = data.get("hourly")
    if not isinstance(hourly, dict) or not hourly.get("time"):
        raise ExternalServiceError(
            service=service,
            category="bad_response",
            detail="Open-Meteo response is missing hourly forecast data.",
            hint="Retry; if persistent, report the issue.",
        )

    times = pd.to_datetime(hourly["time"], errors="coerce")
    if times.isna().all():
        raise ExternalServiceError(
            service=service,
            category="bad_response",
            detail="Open-Meteo returned invalid hourly timestamps.",
            hint="Retry; if persistent, report the issue.",
        )
    if getattr(times, "tz", None) is None:
        times = times.tz_localize(tz_use)
    else:
        times = times.tz_convert(tz_use)

    required_hourly_keys = [
        "temperature_2m",
        "cloud_cover",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
    ]
    missing_keys = [key for key in required_hourly_keys if key not in hourly]
    if missing_keys:
        raise ExternalServiceError(
            service=service,
            category="bad_response",
            detail=f"Open-Meteo hourly data missing keys: {', '.join(missing_keys)}",
            hint="Retry; if persistent, report the issue.",
        )

    df = pd.DataFrame({
        "time": times,
        "temp_air_c": hourly["temperature_2m"],
        "cloud_cover_pct": hourly["cloud_cover"],
        "ghi_wm2": hourly["shortwave_radiation"],
        "dni_wm2": hourly["direct_normal_irradiance"],
        "dhi_wm2": hourly["diffuse_radiation"],
        "wind_speed_ms": hourly.get("wind_speed_10m", [1.0] * len(times)),
    }).set_index("time")

    df = normalize_hourly_forecast_index(
        df[["temp_air_c", "ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_cover_pct", "wind_speed_ms"]],
        target_date,
        tz_use,
    )

    daily = data.get("daily")
    sunrise_list = daily.get("sunrise") if isinstance(daily, dict) else None
    sunset_list = daily.get("sunset") if isinstance(daily, dict) else None

    top_keys = list(data.keys())
    daily_keys = list(daily.keys()) if isinstance(daily, dict) else None

    def _raise_daily_shape_error(reason: str) -> None:
        raise ExternalServiceError(
            service=service,
            category="bad_response",
            detail=(
                "Open-Meteo response is missing daily sunrise/sunset data "
                f"(expected daily.sunrise[0], daily.sunset[0]; {reason}). "
                f"Keys: top={top_keys}, daily={daily_keys}"
            ),
            hint="Retry; if persistent, report the issue.",
        )

    if not isinstance(daily, dict):
        _raise_daily_shape_error("daily is missing or not an object")
    if sunrise_list is None:
        _raise_daily_shape_error("daily.sunrise is missing")
    if sunset_list is None:
        _raise_daily_shape_error("daily.sunset is missing")
    if not isinstance(sunrise_list, (list, tuple)):
        _raise_daily_shape_error("daily.sunrise is not list-like")
    if not isinstance(sunset_list, (list, tuple)):
        _raise_daily_shape_error("daily.sunset is not list-like")
    if not sunrise_list:
        _raise_daily_shape_error("daily.sunrise is empty")
    if not sunset_list:
        _raise_daily_shape_error("daily.sunset is empty")

    sunrise_raw = sunrise_list[0]
    sunset_raw = sunset_list[0]
    if not sunrise_raw:
        _raise_daily_shape_error("daily.sunrise[0] is missing/falsy")
    if not sunset_raw:
        _raise_daily_shape_error("daily.sunset[0] is missing/falsy")

    sunrise_ts = pd.to_datetime(sunrise_raw, errors="coerce")
    sunset_ts = pd.to_datetime(sunset_raw, errors="coerce")
    if pd.isna(sunrise_ts) or pd.isna(sunset_ts):
        raise ExternalServiceError(
            service=service,
            category="bad_response",
            detail=(
                "Open-Meteo returned invalid daily sunrise/sunset timestamps: "
                f"sunrise={sunrise_raw!r}, sunset={sunset_raw!r}"
            ),
            hint="Retry; if persistent, report the issue.",
        )
    if sunrise_ts.tzinfo is None:
        sunrise_ts = sunrise_ts.tz_localize(tz_use)
    else:
        sunrise_ts = sunrise_ts.tz_convert(tz_use)
    if sunset_ts.tzinfo is None:
        sunset_ts = sunset_ts.tz_localize(tz_use)
    else:
        sunset_ts = sunset_ts.tz_convert(tz_use)

    sunrise = sunrise_ts.to_pydatetime()
    sunset = sunset_ts.to_pydatetime()
    return ForecastResult(df=df, sunrise=sunrise, sunset=sunset)


def fetch_weather_for_date(loc: Location, target_date: dt.date, tz: str | None = None) -> ForecastResult:
    tz_use = tz or TIMEZONE
    return _fetch_weather_payload(loc, target_date, tz_use)


def fetch_tomorrow_weather(loc: Location, tz: str | None = None) -> ForecastResult:
    tz_use = tz or TIMEZONE
    today_local = dt.datetime.now(ZoneInfo(tz_use)).date()
    tomorrow = today_local + dt.timedelta(days=1)
    return fetch_weather_for_date(loc, tomorrow, tz=tz_use)


def local_day_hourly_index(target_date: dt.date, tzname: str) -> pd.DatetimeIndex:
    """Return the canonical tz-aware hourly index for a local calendar day.

    The returned index naturally contains 23/24/25 rows on DST start/normal/end
    days and preserves duplicate wall-clock hours on fall-back days via timezone-
    aware timestamps.
    """
    tz = ZoneInfo(str(tzname))
    day_start = dt.datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz)
    next_day = day_start + dt.timedelta(days=1)
    return pd.date_range(start=day_start, end=next_day, freq="h", inclusive="left")


def build_local_day_hour_index(day: dt.date, tz: str) -> tuple[pd.DatetimeIndex, bool]:
    """Backward-compatible wrapper around :func:`local_day_hourly_index`."""
    idx = local_day_hourly_index(day, tz)
    return idx, len(idx) != 24


def normalize_hourly_forecast_index(df: "pd.DataFrame", date: dt.date, tz: str) -> "pd.DataFrame":
    if df.empty:
        raise RuntimeError("Open-Meteo hourly forecast is empty.")

    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce")
    if idx.isna().all():
        raise RuntimeError("Open-Meteo hourly forecast index is invalid.")
    if idx.tz is None:
        idx = idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    else:
        idx = idx.tz_convert(tz)

    out.index = idx
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()

    expected_index = local_day_hourly_index(date, tz)
    out = out.reindex(expected_index)

    irr_cols = [c for c in ["ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_cover_pct"] if c in out.columns]
    for col in irr_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0)

    if "temp_air_c" in out.columns:
        out["temp_air_c"] = pd.to_numeric(out["temp_air_c"], errors="coerce").ffill().bfill().fillna(10.0)

    if "wind_speed_ms" in out.columns:
        out["wind_speed_ms"] = pd.to_numeric(out["wind_speed_ms"], errors="coerce").fillna(1.0).clip(lower=0.0)

    return out


# ============================================================
# PV forecast (pvlib-only)
# ============================================================

def irradiance_sanity_warnings(
    df: "pd.DataFrame",
    loc: "Location",
    tz: str,
    *,
    model_id: str = "forecast",
) -> list[str]:
    warnings_out: list[str] = []
    if "ghi_wm2" not in df.columns or df.empty:
        return warnings_out

    ghi = pd.to_numeric(df.get("ghi_wm2"), errors="coerce")
    if ghi.notna().sum() == 0:
        return warnings_out

    max_ghi = float(ghi.max(skipna=True)) if ghi.notna().any() else 0.0
    sustained_mask = ghi > IRRADIANCE_HOURLY_MAX_WM2
    sustained_count = int(sustained_mask.sum())
    extreme_count = int((ghi > IRRADIANCE_HOURLY_EXTREME_WM2).sum())

    if sustained_count > 0:
        sample_hours = [str(ts) for ts in ghi.index[sustained_mask][:5]]
        warnings_out.append(
            f"irradiance anomaly model={model_id}: hourly_ghi_exceeds={IRRADIANCE_HOURLY_MAX_WM2:.0f}W/m² "
            f"count={sustained_count} extreme_count={extreme_count} max_ghi={max_ghi:.1f} sample_hours={sample_hours}"
        )

    if PVLIB_AVAILABLE and getattr(df.index, "tz", None) is not None:
        try:
            pvloc = pvlib.location.Location(latitude=loc.latitude, longitude=loc.longitude, tz=tz)
            clear = pvloc.get_clearsky(df.index.tz_convert(tz), model="ineichen")
            clear_ghi = pd.to_numeric(clear.get("ghi"), errors="coerce").reindex(df.index).fillna(0.0).clip(lower=0.0)
            measured_wh = float(ghi.fillna(0.0).clip(lower=0.0).sum())
            clear_wh = float(clear_ghi.sum())
            ratio = measured_wh / max(clear_wh, 1.0)
            if measured_wh > clear_wh * IRRADIANCE_DAILY_CLEARSKY_FACTOR:
                warnings_out.append(
                    f"irradiance anomaly model={model_id}: daily_ghi_integral_whm2={measured_wh:.1f} "
                    f"clear_sky_whm2={clear_wh:.1f} ratio={ratio:.2f} limit={IRRADIANCE_DAILY_CLEARSKY_FACTOR:.2f}"
                )
        except Exception:
            pass

    return warnings_out

def _integrate_hourly_power_trapezoid(power_kw: "pd.Series") -> "pd.Series":
    """
    Convert power at hour-start timestamps to hourly kWh using trapezoid:
    e[t] = 0.5*(p[t] + p[t+1]) for all but last; last uses p[last].
    Preserve NaN (if p[t] is NaN => e[t] NaN). Clamp negatives to 0.
    """
    p = pd.to_numeric(power_kw, errors="coerce")
    next_p = p.shift(-1)
    energy = 0.5 * (p + next_p)
    if len(energy) > 0:
        energy.iloc[-1] = p.iloc[-1]
    energy = energy.where(p.notna(), np.nan)
    return energy.clip(lower=0.0)


def _apply_last_resort_ghi(provider_ghi: "pd.Series", ghi_candidate: "pd.Series", allow_mask: "pd.Series") -> "pd.Series":
    provider = pd.to_numeric(provider_ghi, errors="coerce")
    candidate = pd.to_numeric(ghi_candidate, errors="coerce")
    mask = pd.to_numeric(allow_mask, errors="coerce").fillna(0).astype(bool)
    return provider.where(provider.notna(), candidate.where(mask, np.nan))


def estimate_pv_with_pvlib(
    df: "pd.DataFrame",
    loc: Location,
    tz: str | None = None,
    *,
    allow_synthetic_ghi_mask: "pd.Series | None" = None,
) -> Tuple["pd.Series", "pd.Series", "pd.Series", "pd.Series"]:
    tz_use = tz or TIMEZONE
    pvloc = pvlib.location.Location(latitude=loc.latitude, longitude=loc.longitude, tz=tz_use)
    times = df.index
    if getattr(times, "tz", None) is None:
        raise ValueError("Forecast times index must be timezone-aware before pvlib calculations.")
    times = times.tz_convert(tz_use)

    df_local = df.copy()
    df_local.index = times
    avail = df_local.notna().any(axis=1)

    min15_attr = df_local.attrs.get("minutely_15_df") if hasattr(df_local, "attrs") else None
    if isinstance(min15_attr, pd.DataFrame) and not min15_attr.empty:
        try:
            min15 = min15_attr.copy()
            min15_idx = pd.to_datetime(min15.index, errors="coerce")
            if min15_idx.tz is None:
                min15_idx = min15_idx.tz_localize(tz_use)
            else:
                min15_idx = min15_idx.tz_convert(tz_use)
            min15.index = min15_idx

            min15_weather = pd.DataFrame(index=min15.index)
            for col in ["ghi_wm2", "dni_wm2", "dhi_wm2"]:
                if col in min15.columns:
                    min15_weather[col] = pd.to_numeric(min15[col], errors="coerce")
            hourly_temp = pd.to_numeric(df_local.get("temp_air_c"), errors="coerce") if "temp_air_c" in df_local.columns else pd.Series(10.0, index=df_local.index)
            hourly_wind = pd.to_numeric(df_local.get("wind_speed_ms"), errors="coerce") if "wind_speed_ms" in df_local.columns else pd.Series(1.0, index=df_local.index)
            hourly_cloud = pd.to_numeric(df_local.get("cloud_cover_pct"), errors="coerce") if "cloud_cover_pct" in df_local.columns else pd.Series(np.nan, index=df_local.index)
            min15_weather["temp_air_c"] = hourly_temp.reindex(min15.index, method="ffill").bfill().fillna(10.0)
            min15_weather["wind_speed_ms"] = hourly_wind.reindex(min15.index, method="ffill").bfill().fillna(1.0)
            min15_weather["cloud_cover_pct"] = hourly_cloud.reindex(min15.index, method="ffill")

            allow15 = None
            if allow_synthetic_ghi_mask is not None:
                allow15 = pd.to_numeric(allow_synthetic_ghi_mask.reindex(min15.index, method="ffill"), errors="coerce").fillna(False).astype(bool)

            min15_weather.attrs = {}
            e15, s15, t15u, t15c = estimate_pv_with_pvlib(
                min15_weather,
                loc,
                tz=tz_use,
                allow_synthetic_ghi_mask=allow15,
            )

            bucket = min15.index.ceil("h") - pd.Timedelta(hours=1)
            def _agg_hour(x: pd.Series) -> pd.Series:
                return pd.to_numeric(x, errors="coerce").groupby(bucket).sum(min_count=1).reindex(df_local.index)

            return _agg_hour(e15), _agg_hour(s15), _agg_hour(t15u), _agg_hour(t15c)
        except Exception as exc:
            warnings.warn(f"15-min PV aggregation failed, falling back to hourly path: {exc}", RuntimeWarning)

    solpos = pvloc.get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    def cloud_transmittance_from_cover(cloud_cover_pct: "pd.Series") -> "pd.Series":
        cloud_fraction = (pd.to_numeric(cloud_cover_pct, errors="coerce") / 100.0).clip(lower=0.0, upper=1.0)
        cloud_fraction = cloud_fraction.fillna(0.0)
        trans = 1.0 - (CLOUD_ATTENUATION_WEIGHT * (cloud_fraction ** CLOUD_ATTENUATION_EXPONENT))
        return trans.clip(lower=CLOUD_TRANSMITTANCE_MIN, upper=1.0)

    def derive_irradiance_from_ghi(ghi_in: "pd.Series") -> Tuple["pd.Series", "pd.Series", "pd.Series"]:
        ghi_s = pd.to_numeric(ghi_in, errors="coerce").reindex(df_local.index).clip(lower=0.0)
        repair_method = IRR_REPAIR_METHOD.lower()
        if repair_method == "erbs":
            decomp = pvlib.irradiance.erbs(ghi_s.fillna(0.0), solpos["apparent_zenith"], times)
            dni_s = pd.to_numeric(decomp["dni"], errors="coerce").fillna(0.0).clip(lower=0.0)
            dhi_s = pd.to_numeric(decomp["dhi"], errors="coerce").fillna(0.0).clip(lower=0.0)
        else:
            decomp = pvlib.irradiance.disc(ghi_s.fillna(0.0), solpos["apparent_zenith"], times)
            dni_s = pd.to_numeric(decomp["dni"], errors="coerce").fillna(0.0).clip(lower=0.0)
            cos_zen_local = pd.to_numeric(solpos["apparent_zenith"], errors="coerce").apply(
                lambda z: max(0.0, math.cos(math.radians(z))) if pd.notna(z) else 0.0
            )
            dhi_s = (ghi_s.fillna(0.0) - (dni_s * cos_zen_local)).clip(lower=0.0)
        return ghi_s.astype(float), dni_s.astype(float), dhi_s.astype(float)

    irradiance_cols = ["ghi_wm2", "dni_wm2", "dhi_wm2"]
    cs = pvloc.get_clearsky(times, model="ineichen")
    daylight = pd.to_numeric(cs.get("ghi"), errors="coerce").reindex(df_local.index).fillna(0.0) > 20.0

    provider_ghi = pd.to_numeric(df_local.get("ghi_wm2"), errors="coerce") if "ghi_wm2" in df_local.columns else pd.Series(np.nan, index=df_local.index)
    cloud_cover = pd.to_numeric(df_local.get("cloud_cover_pct"), errors="coerce") if "cloud_cover_pct" in df_local.columns else pd.Series(np.nan, index=df_local.index)
    trans = cloud_transmittance_from_cover(cloud_cover) if "cloud_cover_pct" in df_local.columns else pd.Series(1.0, index=df_local.index)
    ghi_candidate = pd.to_numeric(cs.get("ghi"), errors="coerce").reindex(df_local.index) * trans
    if allow_synthetic_ghi_mask is None:
        allow_mask = pd.Series(True, index=df_local.index)
    else:
        allow_mask = pd.to_numeric(allow_synthetic_ghi_mask.reindex(df_local.index), errors="coerce").fillna(False).astype(bool)
    ghi_fallback = ghi_candidate.where(allow_mask, np.nan)
    ghi_final = _apply_last_resort_ghi(provider_ghi, ghi_fallback.where(cloud_cover.notna(), np.nan), allow_mask)
    ghi_final = pd.to_numeric(ghi_final, errors="coerce").clip(lower=0.0)

    has_inst = ("ghi_inst_wm2" in df_local.columns) and pd.to_numeric(df_local.get("ghi_inst_wm2"), errors="coerce").notna().any()
    if has_inst:
        ghi = pd.to_numeric(df_local.get("ghi_inst_wm2"), errors="coerce").reindex(df_local.index).clip(lower=0.0)
        dni_inst = pd.to_numeric(df_local.get("dni_inst_wm2"), errors="coerce") if "dni_inst_wm2" in df_local.columns else pd.Series(np.nan, index=df_local.index)
        dhi_inst = pd.to_numeric(df_local.get("dhi_inst_wm2"), errors="coerce") if "dhi_inst_wm2" in df_local.columns else pd.Series(np.nan, index=df_local.index)
        if dni_inst.notna().any() and dhi_inst.notna().any():
            dni = dni_inst
            dhi = dhi_inst
        else:
            ghi, dni, dhi = derive_irradiance_from_ghi(ghi)
        missing_inputs = ghi.isna() & daylight
    elif "dni_wm2" in df_local.columns and "dhi_wm2" in df_local.columns and provider_ghi.notna().any():
        dni = pd.to_numeric(df_local["dni_wm2"], errors="coerce")
        dhi = pd.to_numeric(df_local["dhi_wm2"], errors="coerce")
        ghi = ghi_final
        missing_inputs = ghi_final.isna() & daylight
    else:
        ghi, dni, dhi = derive_irradiance_from_ghi(ghi_final)
        missing_inputs = ghi_final.isna() & daylight

    zenith = pd.to_numeric(solpos["apparent_zenith"], errors="coerce").reindex(df_local.index)
    cos_zenith = zenith.apply(lambda z: math.cos(math.radians(z)) if pd.notna(z) else 0.0)
    cos_zenith = cos_zenith.clip(lower=0.0, upper=1.0).fillna(0.0)

    def irradiance_consistency_stats(ghi_s: "pd.Series", dni_s: "pd.Series", dhi_s: "pd.Series") -> Tuple[float | None, float | None, "pd.Series"]:
        expected_ghi_s = (dhi_s + (dni_s * cos_zenith)).fillna(0.0)
        valid = (ghi_s >= IRR_MIN_GHI_WM2) & (cos_zenith > 0.05)
        if not bool(valid.any()):
            return None, None, valid
        rel_err_s = ((expected_ghi_s - ghi_s).abs() / ghi_s.clip(lower=1.0)).fillna(0.0)
        rel_err_valid = rel_err_s[valid]
        if rel_err_valid.empty:
            return None, None, valid
        return float(rel_err_valid.median()), float((rel_err_valid > IRR_REL_ERR_POINT_THRESHOLD).mean()), valid

    median_rel_err, fraction_bad_points, _ = irradiance_consistency_stats(ghi, dni, dhi)
    if median_rel_err is not None and fraction_bad_points is not None:
        if (median_rel_err > IRR_REL_ERR_MEDIAN_THRESHOLD) or (fraction_bad_points > IRR_BAD_POINT_FRACTION):
            repair_method = IRR_REPAIR_METHOD.lower()
            if repair_method == "erbs":
                erbs_out = pvlib.irradiance.erbs(ghi, solpos["apparent_zenith"], times)
                dni = pd.to_numeric(erbs_out["dni"], errors="coerce").reindex(df_local.index).fillna(0.0).clip(lower=0.0)
                dhi = pd.to_numeric(erbs_out["dhi"], errors="coerce").reindex(df_local.index).fillna(0.0).clip(lower=0.0)
            else:
                disc_out = pvlib.irradiance.disc(ghi, solpos["apparent_zenith"], times)
                dni = pd.to_numeric(disc_out["dni"], errors="coerce").reindex(df_local.index).fillna(0.0).clip(lower=0.0)
                dhi = (ghi - (dni * cos_zenith)).fillna(0.0).clip(lower=0.0)
                repair_method = "disc"
            dni = pd.to_numeric(dni, errors="coerce").reindex(df_local.index).astype(float).clip(lower=0.0)
            dhi = pd.to_numeric(dhi, errors="coerce").reindex(df_local.index).astype(float).clip(lower=0.0)
            print(
                f"Irradiance repair applied (median_rel_err={median_rel_err:.3f}, bad_fraction={fraction_bad_points:.3f}) "
                f"using method={repair_method}"
            )

    ghi = ghi.clip(lower=0.0)
    dni = dni.clip(lower=0.0)
    dhi = dhi.clip(lower=0.0)

    wind_raw = df_local["wind_speed_ms"] if "wind_speed_ms" in df_local.columns else pd.Series(1.0, index=df_local.index)
    wind_speed = pd.to_numeric(wind_raw, errors="coerce").fillna(1.0).clip(lower=0.0)
    temp_air = pd.to_numeric(df_local["temp_air_c"], errors="coerce").ffill().bfill().fillna(10.0)

    dt_h = timestep_hours(df_local.index)
    dc_pr_multiplier, ac_inv_eff_multiplier = resolve_pv_loss_multipliers(
        PERFORMANCE_RATIO,
        INVERTER_EFF,
        PV_LOSS_MODEL,
    )

    def array_energy(tilt: float, az: float, pdc0_kw: float) -> Tuple["pd.Series", "pd.Series"]:
        irradiance_kwargs = {}
        if PV_ALBEDO is not None:
            irradiance_kwargs["albedo"] = PV_ALBEDO
        irr = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=az,
            solar_zenith=solpos["apparent_zenith"],
            solar_azimuth=solpos["azimuth"],
            dni=dni.fillna(0).clip(lower=0),
            ghi=ghi.fillna(0).clip(lower=0),
            dhi=dhi.fillna(0).clip(lower=0),
            dni_extra=dni_extra,
            model="haydavies",
            **irradiance_kwargs,
        )
        poa = irr["poa_global"].fillna(0).clip(lower=0)
        if PV_IAM_MODEL == "ashrae":
            aoi = pvlib.irradiance.aoi(
                surface_tilt=tilt,
                surface_azimuth=az,
                solar_zenith=solpos["apparent_zenith"],
                solar_azimuth=solpos["azimuth"],
            )
            iam_modifier = pvlib.iam.ashrae(aoi, b=PV_IAM_ASHRAE_B)
            poa = (poa * pd.to_numeric(iam_modifier, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)).fillna(0.0)
        temp_model_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
        temp_cell = pvlib.temperature.sapm_cell(
            poa_global=poa,
            temp_air=temp_air,
            wind_speed=wind_speed,
            a=float(temp_model_params["a"]),
            b=float(temp_model_params["b"]),
            deltaT=float(temp_model_params["deltaT"]),
        )
        dc_w = pvlib.pvsystem.pvwatts_dc(
            poa, pdc0=pdc0_kw * 1000.0, gamma_pdc=PV_GAMMA_PDC, temp_cell=temp_cell
        )
        dc_kw = ((dc_w / 1000.0) * dc_pr_multiplier).clip(lower=0)
        dc_kwh = (dc_kw * dt_h).fillna(0).clip(lower=0)
        return dc_kw.astype(float), dc_kwh.astype(float)

    east_dc_kw, east_dc_kwh = array_energy(
        TILT_EAST_DEG, AZIMUTH_EAST_DEG, dc_kwp(ARRAY_EAST_PANELS)
    )
    south_dc_kw, south_dc_kwh = array_energy(
        TILT_SOUTH_DEG, AZIMUTH_SOUTH_DEG, dc_kwp(ARRAY_SOUTH_PANELS)
    )

    east_dc_kwh = (east_dc_kwh * PV_CALIBRATION_FACTOR_EAST).fillna(0).clip(lower=0)
    south_dc_kwh = (south_dc_kwh * PV_CALIBRATION_FACTOR_SOUTH).fillna(0).clip(lower=0)
    east_dc_kw = (east_dc_kwh / dt_h.replace(0.0, float("nan"))).fillna(0).clip(lower=0)
    south_dc_kw = (south_dc_kwh / dt_h.replace(0.0, float("nan"))).fillna(0).clip(lower=0)

    total_dc_kw_unclipped = (east_dc_kw + south_dc_kw).fillna(0).clip(lower=0)
    if INVERTER_AC_MODEL == "pvwatts":
        total_pdc0_w = (
            dc_kwp(ARRAY_EAST_PANELS) + dc_kwp(ARRAY_SOUTH_PANELS)
        ) * 1000.0
        total_ac_kw_unclipped = pd.Series(
            pvlib.inverter.pvwatts(
                pdc=total_dc_kw_unclipped * 1000.0,
                pdc0=total_pdc0_w,
                eta_inv_nom=ac_inv_eff_multiplier,
            ),
            index=total_dc_kw_unclipped.index,
            dtype=float,
        ) / 1000.0
        total_ac_kw_unclipped = total_ac_kw_unclipped.fillna(0.0).clip(lower=0.0)
    else:
        total_ac_kw_unclipped = (total_dc_kw_unclipped * ac_inv_eff_multiplier).fillna(0).clip(lower=0)

    total_ac_kwh_unclipped = (total_ac_kw_unclipped * dt_h).fillna(0.0).clip(lower=0.0)
    total_ac_kw_clipped = total_ac_kw_unclipped.clip(lower=0.0, upper=INVERTER_AC_KW_LIMIT)
    total_ac_kwh_clipped = (total_ac_kw_clipped * dt_h).fillna(0.0).clip(lower=0.0)

    dc_eps = 1e-9
    east_share = (east_dc_kw / total_dc_kw_unclipped.clip(lower=dc_eps)).fillna(0.0).clip(lower=0.0, upper=1.0)
    south_share = (south_dc_kw / total_dc_kw_unclipped.clip(lower=dc_eps)).fillna(0.0).clip(lower=0.0, upper=1.0)
    share_sum = (east_share + south_share).replace(0.0, float("nan"))
    east_share = (east_share / share_sum).fillna(0.0)
    south_share = (south_share / share_sum).fillna(0.0)

    east_ac_kw_clipped = (total_ac_kw_clipped * east_share).fillna(0.0).clip(lower=0.0)
    south_ac_kw_clipped = (total_ac_kw_clipped * south_share).fillna(0.0).clip(lower=0.0)
    if has_inst:
        east_ac_kwh_clipped = _integrate_hourly_power_trapezoid(east_ac_kw_clipped)
        south_ac_kwh_clipped = _integrate_hourly_power_trapezoid(south_ac_kw_clipped)
        total_ac_kwh_unclipped = _integrate_hourly_power_trapezoid(total_ac_kw_unclipped)
        total_ac_kwh_clipped = _integrate_hourly_power_trapezoid(total_ac_kw_clipped)
    else:
        east_ac_kwh_clipped = (east_ac_kw_clipped * dt_h).fillna(0.0).clip(lower=0.0)
        south_ac_kwh_clipped = (south_ac_kw_clipped * dt_h).fillna(0.0).clip(lower=0.0)

    east_ac_kwh_clipped = east_ac_kwh_clipped.where(avail)
    south_ac_kwh_clipped = south_ac_kwh_clipped.where(avail)
    total_ac_kwh_clipped = total_ac_kwh_clipped.where(avail)
    total_ac_kwh_unclipped = total_ac_kwh_unclipped.where(avail)
    east_ac_kwh_clipped = east_ac_kwh_clipped.mask(missing_inputs, np.nan)
    south_ac_kwh_clipped = south_ac_kwh_clipped.mask(missing_inputs, np.nan)
    total_ac_kwh_clipped = total_ac_kwh_clipped.mask(missing_inputs, np.nan)
    total_ac_kwh_unclipped = total_ac_kwh_unclipped.mask(missing_inputs, np.nan)
    east_ac_kwh_clipped = east_ac_kwh_clipped.where(daylight | east_ac_kwh_clipped.isna(), 0.0)
    south_ac_kwh_clipped = south_ac_kwh_clipped.where(daylight | south_ac_kwh_clipped.isna(), 0.0)
    total_ac_kwh_clipped = total_ac_kwh_clipped.where(daylight | total_ac_kwh_clipped.isna(), 0.0)
    total_ac_kwh_unclipped = total_ac_kwh_unclipped.where(daylight | total_ac_kwh_unclipped.isna(), 0.0)

    return (
        east_ac_kwh_clipped.astype(float),
        south_ac_kwh_clipped.astype(float),
        total_ac_kwh_unclipped.astype(float),
        total_ac_kwh_clipped.astype(float),
    )

def build_pv_forecast(
    df: "pd.DataFrame",
    loc: Location,
    tz: str | None = None,
    *,
    allow_synthetic_ghi_mask: "pd.Series | None" = None,
) -> "pd.DataFrame":
    if not PVLIB_AVAILABLE:
        raise SystemExit("pvlib is required. Install with: pip install pvlib")

    (
        east_ac_kwh_clipped,
        south_ac_kwh_clipped,
        total_ac_kwh_unclipped,
        total_ac_kwh_clipped,
    ) = estimate_pv_with_pvlib(
        df,
        loc,
        tz=tz,
        allow_synthetic_ghi_mask=allow_synthetic_ghi_mask,
    )

    out = df.copy()
    dt_h = timestep_hours(out.index)

    out["pv_east_kwh"] = east_ac_kwh_clipped.where(east_ac_kwh_clipped.isna() | (east_ac_kwh_clipped >= 0.0))
    out["pv_south_kwh"] = south_ac_kwh_clipped.where(south_ac_kwh_clipped.isna() | (south_ac_kwh_clipped >= 0.0))
    out["pv_total_unclipped_kwh"] = total_ac_kwh_unclipped.where(total_ac_kwh_unclipped.isna() | (total_ac_kwh_unclipped >= 0.0))
    out["pv_total_kwh"] = total_ac_kwh_clipped.where(total_ac_kwh_clipped.isna() | (total_ac_kwh_clipped >= 0.0))

    out["pv_total_unclipped_kw"] = (out["pv_total_unclipped_kwh"] / dt_h.replace(0.0, float("nan"))).clip(lower=0.0)
    out["pv_total_kw"] = (out["pv_total_kwh"] / dt_h.replace(0.0, float("nan"))).clip(lower=0.0, upper=INVERTER_AC_KW_LIMIT)

    # Legacy aliases kept for backward compatibility with existing UI/flow logic.
    out["pv_dc_available_kwh"] = out["pv_total_unclipped_kwh"]
    out["pv_ac_limited_kwh"] = out["pv_total_kwh"]
    out["pv_dc_available_kw"] = out["pv_total_unclipped_kw"]
    out["pv_ac_limited_kw"] = out["pv_total_kw"]

    out["pv_clipped_kwh"] = (out["pv_total_unclipped_kwh"] - out["pv_total_kwh"]).clip(lower=0.0)

    out["pv_east_kw"] = (out["pv_east_kwh"] / dt_h.replace(0.0, float("nan"))).clip(lower=0.0)
    out["pv_south_kw"] = (out["pv_south_kwh"] / dt_h.replace(0.0, float("nan"))).clip(lower=0.0)

    out = ensure_pv_columns(out, split_ratio=(0.5, 0.5))
    validate_pv_outputs(out)
    out.attrs["pv_method"] = "pvlib"
    return out


def ensure_pv_columns(df: "pd.DataFrame", *, prefer_split: bool = True, split_ratio: tuple[float, float] = (0.0, 1.0)) -> "pd.DataFrame":
    """
    Enforces presence of PV columns expected by UI and detailed simulation.
    split_ratio = (east_ratio, south_ratio) used only when split missing.
    """
    out = df.copy()

    if "pv_total_kwh" not in out.columns:
        if "pv_kwh" in out.columns:
            out["pv_total_kwh"] = out["pv_kwh"]
        else:
            out["pv_total_kwh"] = 0.0

    if "pv_total_unclipped_kwh" not in out.columns:
        if "pv_dc_available_kwh" in out.columns:
            out["pv_total_unclipped_kwh"] = out["pv_dc_available_kwh"]
        else:
            out["pv_total_unclipped_kwh"] = out["pv_total_kwh"]

    if "pv_dc_available_kwh" not in out.columns:
        out["pv_dc_available_kwh"] = out["pv_total_unclipped_kwh"]
    if "pv_ac_limited_kwh" not in out.columns:
        out["pv_ac_limited_kwh"] = out["pv_total_kwh"]

    split_missing = ("pv_east_kwh" not in out.columns) or ("pv_south_kwh" not in out.columns)
    if split_missing and prefer_split:
        e_ratio, s_ratio = split_ratio
        total = pd.to_numeric(out["pv_total_kwh"], errors="coerce")
        out["pv_east_kwh"] = (total * float(e_ratio)).astype(float)
        out["pv_south_kwh"] = (total * float(s_ratio)).astype(float)
    elif split_missing:
        out["pv_east_kwh"] = 0.0
        out["pv_south_kwh"] = pd.to_numeric(out["pv_total_kwh"], errors="coerce")

    if "pv_clipped_kwh" not in out.columns:
        out["pv_clipped_kwh"] = (out["pv_total_unclipped_kwh"] - out["pv_total_kwh"]).clip(lower=0.0)

    out["pv_total_kwh"] = pd.to_numeric(out["pv_total_kwh"], errors="coerce").clip(lower=0.0)
    out["pv_total_unclipped_kwh"] = pd.to_numeric(out["pv_total_unclipped_kwh"], errors="coerce").clip(lower=0.0)
    out["pv_total_unclipped_kwh"] = out["pv_total_unclipped_kwh"].combine_first(out["pv_total_kwh"])
    both = out["pv_total_unclipped_kwh"].notna() & out["pv_total_kwh"].notna()
    out.loc[both, "pv_total_unclipped_kwh"] = out.loc[both, ["pv_total_unclipped_kwh", "pv_total_kwh"]].max(axis=1)
    out["pv_east_kwh"] = pd.to_numeric(out["pv_east_kwh"], errors="coerce").clip(lower=0.0)
    out["pv_south_kwh"] = pd.to_numeric(out["pv_south_kwh"], errors="coerce").clip(lower=0.0)

    out["pv_dc_available_kwh"] = out["pv_total_unclipped_kwh"]
    out["pv_ac_limited_kwh"] = out["pv_total_kwh"]
    out["pv_clipped_kwh"] = (out["pv_total_unclipped_kwh"] - out["pv_total_kwh"]).clip(lower=0.0)
    return out


def validate_pv_outputs(out: "pd.DataFrame") -> None:
    if not (0 < PERFORMANCE_RATIO <= 1.0):
        raise ValueError("PERFORMANCE_RATIO must be within (0, 1].")
    if not (0 < INVERTER_EFF <= 1.0):
        raise ValueError("INVERTER_EFF must be within (0, 1].")
    if PV_LOSS_MODEL not in {"split", "combined"}:
        raise ValueError("PV_LOSS_MODEL must be either 'split' or 'combined'.")
    if not (0.7 <= PV_CALIBRATION_FACTOR_EAST <= 1.3):
        raise ValueError("PV_CALIBRATION_FACTOR_EAST must be within [0.7, 1.3].")
    if not (0.7 <= PV_CALIBRATION_FACTOR_SOUTH <= 1.3):
        raise ValueError("PV_CALIBRATION_FACTOR_SOUTH must be within [0.7, 1.3].")

    eps = 1e-9
    for col in ["pv_east_kwh", "pv_south_kwh", "pv_total_unclipped_kwh", "pv_total_kwh", "pv_clipped_kwh"]:
        if col in out.columns and (out[col].fillna(0.0) < -eps).any():
            raise ValueError(f"PV output column '{col}' contains negative energy values.")

    if "pv_total_unclipped_kwh" in out.columns and "pv_total_kwh" in out.columns:
        exceeds = out["pv_total_kwh"].fillna(0.0) > out["pv_total_unclipped_kwh"].fillna(0.0) + eps
        if bool(exceeds.any()):
            raise ValueError("pv_total_kwh cannot exceed pv_total_unclipped_kwh.")

    if "pv_clipped_kwh" in out.columns and (out["pv_clipped_kwh"].fillna(0.0) < -eps).any():
        raise ValueError("pv_clipped_kwh cannot be negative.")


def _soft_daylight_factor_from_elevation(elev_deg: float) -> float:
    """Map solar elevation to a soft daylight gating factor in [0, 1]."""
    elev = float(elev_deg)
    if elev <= -3.0:
        return 0.0
    if elev >= 6.0:
        return 1.0
    return (elev + 3.0) / 9.0


def compute_solar_elevation_series(index: "pd.DatetimeIndex", loc: Location) -> "pd.Series":
    if not PVLIB_AVAILABLE:
        raise RuntimeError("pvlib is required for solar elevation daylight gating")
    latitude = float(loc.latitude)
    longitude = float(loc.longitude)
    solpos = pvlib.solarposition.get_solarposition(index, latitude, longitude, altitude=loc.elevation_m)
    elev = pd.to_numeric(solpos.get("apparent_elevation"), errors="coerce")
    return pd.Series(elev.values, index=index, dtype=float)


def _apply_soft_daylight_factor_and_twilight_clamp(df: "pd.DataFrame", factor: "pd.Series") -> "pd.DataFrame":
    out = df.copy()
    pv_cols = [
        col for col in out.columns
        if isinstance(col, str) and col.startswith("pv_") and col.endswith("_kwh")
    ]
    if not pv_cols:
        return out

    aligned_factor = pd.to_numeric(factor.reindex(out.index), errors="coerce")
    for col in pv_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        gated = values * aligned_factor
        twilight_mask = (aligned_factor < 0.25) & gated.notna() & (gated.abs() < 0.01)
        gated.loc[twilight_mask] = 0.0
        out[col] = gated
    return out


def apply_soft_daylight_gating(df: "pd.DataFrame", loc: Location) -> "pd.DataFrame":
    elev = compute_solar_elevation_series(df.index, loc)
    factor = elev.apply(_soft_daylight_factor_from_elevation)
    return _apply_soft_daylight_factor_and_twilight_clamp(df, factor)


def apply_daylight_clamp(df: "pd.DataFrame", sunrise: dt.datetime, sunset: dt.datetime) -> "pd.DataFrame":
    """Backward-compatible hard clamp kept for legacy callers/tests."""
    out = df.copy()
    _, _, daylight_mask = normalize_daylight_window(out.index, sunrise, sunset)
    pv_cols = [
        col for col in out.columns
        if isinstance(col, str) and col.startswith("pv_") and col.endswith("_kwh")
    ]
    for col in pv_cols:
        out.loc[~daylight_mask, col] = 0.0
    return out


def align_timestamp_to_index_tz(value: dt.datetime, index: "pd.DatetimeIndex") -> "pd.Timestamp":
    ts = pd.Timestamp(value)
    tz = index.tz
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def normalize_daylight_window(
    df_index: "pd.DatetimeIndex",
    sunrise: dt.datetime,
    sunset: dt.datetime
) -> Tuple["pd.Timestamp", "pd.Timestamp", "pd.Series"]:
    if getattr(df_index, "tz", None) is None:
        raise ValueError("Forecast times index must be timezone-aware for daylight normalization.")

    sunrise_ts = align_timestamp_to_index_tz(sunrise, df_index)
    sunset_ts = align_timestamp_to_index_tz(sunset, df_index)
    daylight_mask = pd.Series((df_index >= sunrise_ts) & (df_index <= sunset_ts), index=df_index)
    return sunrise_ts, sunset_ts, daylight_mask


def add_sun_percent(out: "pd.DataFrame", sunrise: dt.datetime, sunset: dt.datetime) -> "pd.DataFrame":
    df = out.copy()
    _, _, daylight_mask = normalize_daylight_window(df.index, sunrise, sunset)
    max_pv = df.loc[daylight_mask, "pv_total_kwh"].max() if daylight_mask.any() else 0.0
    df["sun_percent"] = 0
    if max_pv and max_pv > 0:
        df.loc[daylight_mask, "sun_percent"] = (df.loc[daylight_mask, "pv_total_kwh"] / max_pv * 100.0).round(0).astype(int)
    df.loc[~daylight_mask, "sun_percent"] = 0
    df.loc[df["pv_total_kwh"] < 0.001, "sun_percent"] = 0
    return df


def validate_array_orientation_logic(df: "pd.DataFrame", sunrise: dt.datetime, sunset: dt.datetime) -> None:
    _, _, daylight_mask = normalize_daylight_window(df.index, sunrise, sunset)
    day = df.loc[daylight_mask].copy()
    if day.empty:
        return

    east_day = day["pv_east_kwh"].fillna(0.0)
    south_day = day["pv_south_kwh"].fillna(0.0)

    if east_day.max() > 0 and south_day.max() > 0:
        east_peak = east_day.idxmax().hour
        south_peak = south_day.idxmax().hour
        if east_peak > south_peak:
            print("Warning: East peak occurs after South peak. Check azimuth/tilt or weather input.")

    total_east = float(east_day.sum())
    total_south = float(south_day.sum())
    if total_south > 0:
        ratio = total_east / total_south
        if not (0.35 <= ratio <= 0.95):
            print(
                "Warning: East/South daily energy ratio is outside expected range (0.35-0.95). "
                "Check panel configuration or weather data."
            )


# ============================================================
# CORE: timing-aware SOC & charge plan
# ============================================================

def compute_soc_low_timing_aware(
    df: "pd.DataFrame",
    total_consumption_kwh: float,
    for_date: dt.date,
    buffer_soc: float = 0.0,
    tariff_cfg: Optional[dict] = None,
    pv_col: str = "pv_total_kwh",
) -> float:
    expensive_windows = get_expensive_windows(for_date, tariff_cfg)
    if not expensive_windows:
        return MIN_SOC

    loads = build_hourly_load_series(df.index, total_consumption_kwh)
    cum = 0.0
    max_cum = 0.0

    optimization_mode = str((tariff_cfg or {}).get("optimization_mode", "window_only")).strip().lower()
    optimization_mode = optimization_mode if optimization_mode in {"window_only", "price_aware"} else "window_only"
    offpeak_price = float((tariff_cfg or {}).get("offpeak_grid_price_eur_per_kwh", OFFPEAK_GRID_PRICE_EUR_PER_KWH))
    break_even_price = offpeak_price / max(1e-9, BATTERY_AC_CHARGE_EFF * BATTERY_DISCHARGE_EFF)

    for ts in df.index:
        if not in_any_window(ts.time(), expensive_windows):
            continue
        pv = float(df.loc[ts, pv_col])
        load = float(loads.loc[ts])
        net = load - pv  # + tekort, - overschot
        if optimization_mode == "price_aware":
            hour_price = import_price_eur_per_kwh(ts, tariff_cfg or DEFAULT_CONFIG["tariff"])
            if hour_price <= 0.0:
                net = 0.0
            elif hour_price <= break_even_price:
                net = 0.0
            else:
                net *= (hour_price - break_even_price) / hour_price
        cum += net
        if cum > max_cum:
            max_cum = cum

    required_from_batt_kwh = max(0.0, max_cum) / BATTERY_DISCHARGE_EFF
    soc_low = MIN_SOC + (required_from_batt_kwh / BATTERY_KWH)
    soc_low = min(max(soc_low, MIN_SOC), 1.0)
    soc_low = min(1.0, soc_low + float(buffer_soc))
    return soc_low


def run_forecast_pipeline(
    cfg: dict,
    target_date: dt.date,
    soc_at_22_percent: float,
    yesterday_kwh: float,
    buffer_percent: float,
    user_max_ac_kw: float,
) -> PlannerOutput:
    with applied_config(cfg) as effective_cfg:
        quick_sanity_checks()

        loc_cfg = effective_cfg.get("location", {})
        tz = str(loc_cfg["timezone"])
        loc = Location(
            name=str(loc_cfg.get("address_query") or loc_cfg.get("name") or "Configured"),
            latitude=float(loc_cfg["latitude"]),
            longitude=float(loc_cfg["longitude"]),
            elevation_m=float(loc_cfg["elevation_m"]) if loc_cfg.get("elevation_m") is not None else None,
        )

        weather = fetch_weather_for_date(loc, target_date, tz=tz)
        pv = build_pv_forecast(weather.df, loc, tz=tz)
        pv = apply_soft_daylight_gating(pv, loc).sort_index()
        pv = add_sun_percent(pv, weather.sunrise, weather.sunset)
        pv = add_load_and_surplus_columns(pv, yesterday_kwh)

        tariff_cfg = effective_cfg.get("tariff", DEFAULT_CONFIG["tariff"])
        soc_low = compute_soc_low_timing_aware(pv, yesterday_kwh, target_date, tariff_cfg=tariff_cfg)
        _, soc_high = compute_soc_high_headroom(
            pv,
            yesterday_kwh,
            target_date,
            sunrise=weather.sunrise,
            sunset=weather.sunset,
        )
        cutoff_soc_raw, cutoff_reason = choose_cutoff_soc(target_date, soc_low, soc_high, tariff_cfg=tariff_cfg)
        cutoff_soc = cutoff_soc_raw + (float(buffer_percent) / 100.0)

        old_cutoff_soc = cutoff_soc
        cutoff_soc = min(max(cutoff_soc, MIN_SOC), MAX_CUTOFF_SOC)
        cutoff_note = (
            f"Cutoff capped to {MAX_CUTOFF_SOC_PERCENT:.1f}% (was {old_cutoff_soc*100:.1f}%)."
            if abs(cutoff_soc - old_cutoff_soc) > 1e-9 else ""
        )

        charge_date = target_date - dt.timedelta(days=1)
        _, charge_kw, charge_note, achieved_soc_start = plan_charge_power(
            soc_at_22_percent / 100.0,
            cutoff_soc,
            charge_date,
            user_cap_kw=user_max_ac_kw,
            tariff_cfg=tariff_cfg,
        )

        detail_df, grid_import, grid_export, _, _ = simulate_expensive_hours_detailed(
            pv, yesterday_kwh, achieved_soc_start, target_date, tariff_cfg=tariff_cfg
        )
        full_soc, full_flows = simulate_full_day_soc(
            pv,
            yesterday_kwh,
            soc_at_22_percent / 100.0,
            charge_kw,
            cutoff_soc,
            target_date,
            tariff_cfg=tariff_cfg,
        )

        return PlannerOutput(
            location=loc,
            tomorrow_date=target_date,
            weather=weather,
            hourly_df=pv,
            expensive_detail_df=detail_df,
            full_day_soc=full_soc,
            full_day_flows_df=full_flows,
            cutoff_soc=cutoff_soc,
            charge_kw=charge_kw,
            achieved_soc_start=achieved_soc_start,
            grid_import_expensive_kwh=float(grid_import),
            grid_export_expensive_kwh=float(grid_export),
            curtailed_expensive_kwh=float(detail_df["curtailed_kwh"].sum()) if not detail_df.empty else 0.0,
            cutoff_note=cutoff_note,
            cutoff_reason=cutoff_reason,
            charge_note=charge_note,
        )


def run_detailed_plan(
    target_date: dt.date,
    weather: ForecastResult,
    pv_df: "pd.DataFrame",
    consumption_kwh: "pd.Series",
    soc_at_22_percent: float,
    buffer_percent: float,
    max_ac_charge_power_kw: float,
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.Series", float, float, str, str, bool, str]:
    """Legacy backend API entrypoint retained for compatibility."""
    total_consumption_kwh = float(pd.to_numeric(consumption_kwh, errors="coerce").fillna(0.0).sum())

    pv = pv_df.copy()
    if "load_kwh" not in pv.columns:
        pv = add_load_and_surplus_columns(pv, total_consumption_kwh)

    tariff_cfg = EFFECTIVE_CFG.get("tariff", DEFAULT_CONFIG["tariff"]) if isinstance(EFFECTIVE_CFG, dict) else DEFAULT_CONFIG["tariff"]
    # Use the same PV series for both soc_low and soc_high to keep cutoff decision internally consistent.
    pv_col_for_planning = "pv_total_decision_kwh" if "pv_total_decision_kwh" in pv.columns else "pv_total_kwh"
    soc_low = compute_soc_low_timing_aware(pv, total_consumption_kwh, target_date, tariff_cfg=tariff_cfg, pv_col=pv_col_for_planning)
    _, soc_high = compute_soc_high_headroom(
        pv,
        total_consumption_kwh,
        target_date,
        sunrise=weather.sunrise,
        sunset=weather.sunset,
        pv_col=pv_col_for_planning,
    )
    cutoff_soc_raw, cutoff_reason = choose_cutoff_soc(target_date, soc_low, soc_high, tariff_cfg=tariff_cfg)
    cutoff_soc = min(max(cutoff_soc_raw + (float(buffer_percent) / 100.0), MIN_SOC), MAX_CUTOFF_SOC)

    charge_date = target_date - dt.timedelta(days=1)
    _, charge_kw, charge_note, achieved_soc_start = plan_charge_power(
        soc_at_22_percent / 100.0,
        cutoff_soc,
        charge_date,
        user_cap_kw=max_ac_charge_power_kw,
        tariff_cfg=tariff_cfg,
    )
    target_date_for_window = charge_date + dt.timedelta(days=1)
    window_start, window_end = compute_charging_window_for_target_date(target_date_for_window, tariff_cfg)
    available_charge_hours = float(len(get_charge_session_index_from_window(window_start, window_end)))
    required_grid_kwh = 0.0
    if available_charge_hours > 0 and cutoff_soc > (soc_at_22_percent / 100.0) + 1e-9 and tariff_has_meaningful_spread(tariff_cfg):
        soc_at_22_kwh = (soc_at_22_percent / 100.0) * BATTERY_KWH
        target_soc_kwh = cutoff_soc * BATTERY_KWH
        required_batt_kwh = max(0.0, target_soc_kwh - soc_at_22_kwh)
        required_grid_kwh = required_batt_kwh / BATTERY_AC_CHARGE_EFF
    recommended_allowed_ac_kw = (required_grid_kwh / available_charge_hours) if available_charge_hours > 0 else 0.0
    charge_effective_cap_kw, charge_limit_reason_raw = _compute_charge_limit_metadata(
        recommended_kw=recommended_allowed_ac_kw,
        user_cap_kw=max_ac_charge_power_kw,
    )

    detail_df, _, _, _, _ = simulate_expensive_hours_detailed(
        pv,
        total_consumption_kwh,
        achieved_soc_start,
        target_date,
        tariff_cfg=tariff_cfg,
        pv_col=pv_col_for_planning,
    )
    soc_series, flows_df = simulate_full_day_soc(
        pv,
        total_consumption_kwh,
        soc_at_22_percent / 100.0,
        charge_kw,
        cutoff_soc,
        target_date,
        tariff_cfg=tariff_cfg,
        pv_col=pv_col_for_planning,
    )
    flows_df.attrs["charge_effective_cap_kw"] = float(charge_effective_cap_kw)
    flows_df.attrs["charge_limit_reason_raw"] = str(charge_limit_reason_raw)
    charge_warning_text = str(charge_note) if str(charge_note).startswith("Warning") else ""
    charge_target_reachable = not bool(charge_warning_text)
    return detail_df, flows_df, soc_series, float(charge_kw), float(cutoff_soc), str(cutoff_reason), str(charge_note), bool(charge_target_reachable), str(charge_warning_text)

def compute_soc_high_headroom(
    df: "pd.DataFrame",
    total_consumption_kwh: float,
    for_date: dt.date,
    sunrise: dt.datetime,
    sunset: dt.datetime,
    pv_col: str = "pv_total_kwh",
) -> Tuple[float, float]:
    """
    Headroom doel: hoeveel PV-overschot verwacht je BINNEN daglichturen.
    Hoe meer overschot, hoe lager je bij start van hoog tarief wil zitten om injectie te vermijden.
    """
    _ = for_date
    if pv_col not in df.columns:
        raise KeyError(f"compute_soc_high_headroom missing pv_col={pv_col!r} in df columns")

    _, _, daylight_mask = normalize_daylight_window(df.index, sunrise, sunset)
    if not bool(daylight_mask.any()):
        return 0.0, 1.0

    loads = build_hourly_load_series(df.index, total_consumption_kwh)
    dt_h = timestep_hours(df.index)
    surplus_sum_ac = 0.0
    stored_kwh_sum = 0.0

    for ts in df.index:
        if not bool(daylight_mask.loc[ts]):
            continue

        pv_ac_kwh = float(df.loc[ts, pv_col])
        load_ac_kwh = float(loads.loc[ts])
        surplus_ac_kwh = max(0.0, pv_ac_kwh - load_ac_kwh)
        surplus_sum_ac += surplus_ac_kwh

        stored_candidate_kwh = surplus_ac_kwh * BATTERY_PV_CHARGE_EFF
        max_store_kwh = float(BATTERY_MAX_CHARGE_KW) * float(dt_h.loc[ts])
        stored_kwh_sum += min(stored_candidate_kwh, max_store_kwh)

    soc_high = 1.0 - (stored_kwh_sum / BATTERY_KWH)
    soc_high = min(max(soc_high, MIN_SOC), 1.0)
    return surplus_sum_ac, soc_high


def choose_cutoff_soc(for_date: dt.date, soc_low: float, soc_high: float, tariff_cfg: Optional[dict] = None) -> Tuple[float, str]:
    if not tariff_has_meaningful_spread(tariff_cfg or DEFAULT_CONFIG["tariff"]):
        return MIN_SOC, "No active arbitrage target: flat tariff / no meaningful spread; keep cutoff at MIN_SOC for PV headroom."

    expensive_windows = get_expensive_windows(for_date, tariff_cfg)
    if not expensive_windows:
        return MIN_SOC, "No expensive hours (all off-peak): keep cutoff low for maximum headroom."

    if soc_low <= soc_high + 1e-9:
        return soc_low, "OK: bridge expensive hours and keep headroom to reduce export."
    return soc_high, (
        "CONFLICT: required SOC to bridge expensive hours is higher than PV headroom target. "
        "Using headroom target to avoid morning PV export; expect some grid import later if PV underperforms."
    )


def plan_charge_power(
    soc_start: float,
    soc_cutoff: float,
    charge_date: dt.date,
    user_cap_kw: Optional[float] = None,
    tariff_cfg: Optional[dict] = None,
) -> Tuple[float, float, str, float]:
    cfg = tariff_cfg or DEFAULT_CONFIG["tariff"]
    target_date = charge_date + dt.timedelta(days=1)
    window_start, window_end = compute_charging_window_for_target_date(target_date, cfg)
    session_idx = get_charge_session_index_from_window(window_start, window_end)
    available_charge_hours = float(len(session_idx))
    if available_charge_hours <= 0:
        return 0.0, 0.0, "No off-peak hours available in configured charging windows.", soc_start

    soc_start = max(min(soc_start, 1.0), 0.0)
    soc_cutoff = max(min(soc_cutoff, 1.0), 0.0)

    if soc_cutoff <= soc_start + 1e-9:
        return 0.0, 0.0, "No AC charging needed (cutoff already reached).", soc_start

    if not tariff_has_meaningful_spread(cfg):
        return 0.0, 0.0, "No active arbitrage AC charging: flat tariff / no meaningful spread.", soc_start

    soc_at_22_kwh = soc_start * BATTERY_KWH
    target_soc_kwh = soc_cutoff * BATTERY_KWH
    required_batt_kwh = max(0.0, target_soc_kwh - soc_at_22_kwh)
    required_grid_kwh = required_batt_kwh / BATTERY_AC_CHARGE_EFF
    recommended_allowed_ac_kw = required_grid_kwh / available_charge_hours
    effective_cap_kw, _ = _compute_charge_limit_metadata(
        recommended_kw=recommended_allowed_ac_kw,
        user_cap_kw=user_cap_kw,
    )
    allowed_ac_kw = min(recommended_allowed_ac_kw, effective_cap_kw)

    if recommended_allowed_ac_kw > allowed_ac_kw + 1e-9:
        achievable_grid_kwh = allowed_ac_kw * available_charge_hours
        achievable_batt_kwh = achievable_grid_kwh * BATTERY_AC_CHARGE_EFF
        achievable_soc = min(1.0, soc_start + achievable_batt_kwh / BATTERY_KWH)
        note = (
            f"Warning: Needed ≈ {recommended_allowed_ac_kw:.2f} kW over {available_charge_hours:.1f}h, "
            f"but cap is {allowed_ac_kw:.2f} kW. "
            f"Cutoff may be unreachable; achievable SOC ≈ {achievable_soc*100:.1f}%."
        )
        return required_grid_kwh, allowed_ac_kw, note, achievable_soc

    return required_grid_kwh, allowed_ac_kw, f"Automatically computed over {available_charge_hours:.1f}h charging window.", soc_cutoff


def _compute_charge_limit_metadata(recommended_kw: float, user_cap_kw: Optional[float]) -> Tuple[float, str]:
    user_cap = MAX_AC_CHARGE_KW_HARD_LIMIT if user_cap_kw is None else max(float(user_cap_kw), 0.0)
    caps = {
        "user": float(user_cap),
        "inverter": float(INVERTER_AC_KW_LIMIT),
        "battery": float(BATTERY_MAX_CHARGE_KW),
    }
    effective_cap_kw = float(min(caps.values()))
    if float(recommended_kw) <= effective_cap_kw + 1e-9:
        return effective_cap_kw, "none"
    limiter_reason = min(caps.items(), key=lambda item: item[1])[0]
    return effective_cap_kw, str(limiter_reason)


# ============================================================
# DETAIL SIMULATIE: per uur binnen hoog-tarief uren
# ============================================================

def simulate_expensive_hours_detailed(
    df: "pd.DataFrame",
    total_consumption_kwh: float,
    start_soc: float,
    for_date: dt.date,
    tariff_cfg: Optional[dict] = None,
    pv_col: str = "pv_total_kwh",
) -> Tuple["pd.DataFrame", float, float, float, bool]:
    if pv_col not in df.columns:
        raise KeyError(f"simulate_expensive_hours_detailed missing pv_col='{pv_col}' in df columns")

    expensive_windows = get_expensive_windows(for_date, tariff_cfg or DEFAULT_CONFIG["tariff"])
    if not expensive_windows:
        detail = df.copy().iloc[0:0]
        return detail, 0.0, 0.0, start_soc, False

    loads = build_hourly_load_series(df.index, total_consumption_kwh)
    dt_h = timestep_hours(df.index)

    energy = max(min(start_soc, 1.0), 0.0) * BATTERY_KWH
    min_energy = MIN_SOC * BATTERY_KWH
    max_energy = MAX_CUTOFF_SOC * BATTERY_KWH

    rows = []
    grid_import_total = 0.0
    grid_export_total = 0.0
    hit_min = False
    import_with_high_soc_due_to_power_limit = False
    allow_injection = bool((tariff_cfg or {}).get("allow_injection_to_grid", True))
    blocked_export_kwh_total = 0.0

    def _get_float(frame: "pd.DataFrame", ts: pd.Timestamp, col: str, default: float = 0.0) -> float:
        if col in frame.columns:
            try:
                value = frame.loc[ts, col]
                return float(value) if pd.notna(value) else float(default)
            except Exception:
                return float(default)
        return float(default)

    for ts in df.index:
        if not in_any_window(ts.time(), expensive_windows):
            continue

        step_h = float(dt_h.loc[ts])
        pv_selected = _get_float(df, ts, pv_col, default=0.0)
        if pv_col == "pv_total_kwh":
            pv_ac_limited = _get_float(df, ts, "pv_ac_limited_kwh", default=pv_selected)
        else:
            pv_ac_limited = pv_selected

        pv_unclipped = _get_float(
            df,
            ts,
            "pv_dc_available_kwh",
            default=_get_float(
                df,
                ts,
                "pv_total_unclipped_kwh",
                default=pv_ac_limited,
            ),
        )
        pv_unclipped = max(pv_unclipped, pv_ac_limited)
        load = float(loads.loc[ts])

        soc_start_pct = (energy / BATTERY_KWH * 100.0) if BATTERY_KWH > 0 else 0.0

        pv_to_load = min(pv_ac_limited, load)
        remaining_load = max(0.0, load - pv_to_load)
        pv_after_load = max(0.0, pv_ac_limited - pv_to_load)
        overflow = max(0.0, pv_unclipped - pv_ac_limited)

        batt_charge_kwh = 0.0
        batt_discharge_kwh = 0.0
        grid_import = 0.0
        grid_export = 0.0
        curtailed = 0.0

        pv_for_storage = pv_after_load + overflow
        if pv_for_storage > 0:
            room = max(0.0, max_energy - energy)
            charge_power_limited = BATTERY_MAX_CHARGE_KW * step_h
            pv_limited_store = pv_for_storage * BATTERY_PV_CHARGE_EFF
            store = min(room, pv_limited_store, charge_power_limited)
            energy += store
            batt_charge_kwh = store
            pv_used_for_batt = store / BATTERY_PV_CHARGE_EFF if BATTERY_PV_CHARGE_EFF > 0 else 0.0
            pv_after_batt = max(0.0, pv_for_storage - pv_used_for_batt)

            export_limit = max(0.0, (INVERTER_AC_KW_LIMIT * step_h) - pv_to_load)
            effective_export_limit = export_limit if allow_injection else 0.0
            grid_export = min(pv_after_batt, effective_export_limit)
            if not allow_injection:
                blocked_export_kwh_total += min(pv_after_batt, export_limit)
            curtailed = max(0.0, pv_after_batt - grid_export)
            grid_export_total += grid_export

        if remaining_load > 0:
            available = max(0.0, energy - min_energy)
            discharge_power_limited = BATTERY_MAX_DISCHARGE_KW * step_h
            needed_from_batt = remaining_load / BATTERY_DISCHARGE_EFF if BATTERY_DISCHARGE_EFF > 0 else remaining_load
            discharge = min(available, needed_from_batt, discharge_power_limited)

            energy -= discharge
            batt_discharge_kwh = discharge

            delivered = discharge * BATTERY_DISCHARGE_EFF
            grid_import = max(0.0, remaining_load - delivered)
            grid_import_total += grid_import

            max_deliverable = BATTERY_MAX_DISCHARGE_KW * step_h * BATTERY_DISCHARGE_EFF
            if remaining_load - max_deliverable > 1e-9:
                print(
                    f"Warning: {ts.strftime('%Y-%m-%d %H:%M %Z')} expensive-hour deficit ({remaining_load:.2f} kWh) "
                    f"exceeds battery discharge power limit ({BATTERY_MAX_DISCHARGE_KW * step_h:.2f} kWh pre-eff)."
                )
                if soc_start_pct > (MIN_SOC_PERCENT + 5.0) and grid_import > 1e-9:
                    import_with_high_soc_due_to_power_limit = True

            if energy <= min_energy + 1e-9 and grid_import > 0:
                hit_min = True

        soc_end_pct = (energy / BATTERY_KWH * 100.0) if BATTERY_KWH > 0 else soc_start_pct

        rows.append({
            "time": ts,
            "pv_dc_available_kwh": pv_unclipped,
            "pv_ac_limited_kwh": pv_ac_limited,
            "pv_overflow_kwh": overflow,
            "load_kwh": load,
            "pv_to_load_kwh": pv_to_load,
            "surplus_kwh": max(0.0, pv_ac_limited - load),
            "deficit_kwh": max(0.0, load - pv_ac_limited),
            "batt_charge_kwh": batt_charge_kwh,
            "batt_discharge_kwh": batt_discharge_kwh,
            "grid_import_kwh": grid_import,
            "grid_export_kwh": grid_export,
            "curtailed_kwh": curtailed,
            "soc_start_pct": soc_start_pct,
            "soc_end_pct": soc_end_pct
        })

    detail_df = pd.DataFrame(rows).set_index("time")

    if "pv_total_unclipped_kwh" not in detail_df.columns and "pv_dc_available_kwh" in detail_df.columns:
        detail_df["pv_total_unclipped_kwh"] = detail_df["pv_dc_available_kwh"]
    if "pv_total_kwh" not in detail_df.columns and "pv_ac_limited_kwh" in detail_df.columns:
        detail_df["pv_total_kwh"] = detail_df["pv_ac_limited_kwh"]

    pv_total_unclipped = detail_df.get("pv_total_unclipped_kwh")
    if pv_total_unclipped is None:
        pv_total_unclipped = pd.Series(0.0, index=detail_df.index)
    detail_df["pv_total_unclipped_kwh"] = pd.to_numeric(pv_total_unclipped, errors="coerce").fillna(0.0)

    pv_total = detail_df.get("pv_total_kwh")
    if pv_total is None:
        pv_total = pd.Series(0.0, index=detail_df.index)
    detail_df["pv_total_kwh"] = pd.to_numeric(pv_total, errors="coerce").fillna(0.0)
    detail_df["pv_clipped_kwh"] = (detail_df["pv_total_unclipped_kwh"] - detail_df["pv_total_kwh"]).clip(lower=0.0)

    if "pv_surplus_kwh" not in detail_df.columns and "surplus_kwh" in detail_df.columns:
        detail_df["pv_surplus_kwh"] = detail_df["surplus_kwh"]
    if "pv_deficit_kwh" not in detail_df.columns and "deficit_kwh" in detail_df.columns:
        detail_df["pv_deficit_kwh"] = detail_df["deficit_kwh"]

    if "pv_east_kwh" in df.columns and "pv_south_kwh" in df.columns:
        detail_df["pv_east_kwh"] = pd.to_numeric(df.reindex(detail_df.index)["pv_east_kwh"], errors="coerce").fillna(0.0)
        detail_df["pv_south_kwh"] = pd.to_numeric(df.reindex(detail_df.index)["pv_south_kwh"], errors="coerce").fillna(0.0)
    else:
        split_total = max(ARRAY_EAST_PANELS + ARRAY_SOUTH_PANELS, 1)
        east_ratio = ARRAY_EAST_PANELS / split_total
        south_ratio = ARRAY_SOUTH_PANELS / split_total
        detail_df["pv_east_kwh"] = detail_df["pv_total_kwh"] * east_ratio
        detail_df["pv_south_kwh"] = detail_df["pv_total_kwh"] * south_ratio
    if ENABLE_INVARIANT_CHECKS:
        validate_flow_invariants(detail_df, "expensive_hours")
    detail_df.attrs["allow_injection_to_grid"] = allow_injection
    detail_df.attrs["export_blocked_by_policy"] = bool(not allow_injection)
    detail_df.attrs["blocked_export_kwh_total"] = float(blocked_export_kwh_total)
    if import_with_high_soc_due_to_power_limit:
        print(
            "Warning: expensive-hour imports occurred while SOC was high due to battery power limits "
            "(discharge kW cap), not available energy."
        )
    end_soc = (energy / BATTERY_KWH) if BATTERY_KWH > 0 else start_soc
    return detail_df, grid_import_total, grid_export_total, end_soc, hit_min



def simulate_night_charging_series(
    soc_at_22: float,
    charge_kw: float,
    cutoff_soc: float,
    session_start: Optional[pd.Timestamp] = None,
    session_end: Optional[pd.Timestamp] = None,
    total_consumption_kwh: float = 0.0,
    tariff_cfg: Optional[dict] = None,
    tomorrow_date: Optional[dt.date] = None,
    precomputed_loads: Optional[pd.Series] = None,
) -> "pd.DataFrame":
    cfg = tariff_cfg or DEFAULT_CONFIG["tariff"]
    if session_start is None or session_end is None:
        if tomorrow_date is None:
            raise ValueError("tomorrow_date is required when session_start/session_end are not provided.")
        session_start, session_end = compute_charging_window_for_target_date(tomorrow_date, cfg)
    idx = pd.date_range(session_start, session_end, freq="h", inclusive="left", tz=TIMEZONE)
    if precomputed_loads is not None:
        loads = pd.to_numeric(precomputed_loads.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    else:
        if tomorrow_date is None:
            tomorrow_date = session_end.date()
        cycle_loads = build_cycle_hourly_load_series(tomorrow_date, total_consumption_kwh, tariff_cfg=cfg)
        loads = pd.to_numeric(cycle_loads.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    energy = max(0.0, min(1.0, soc_at_22)) * BATTERY_KWH
    min_energy = MIN_SOC * BATTERY_KWH
    max_energy = MAX_CUTOFF_SOC * BATTERY_KWH
    charge_cutoff_energy = min(MAX_CUTOFF_SOC, max(MIN_SOC, cutoff_soc)) * BATTERY_KWH
    night_load_from_battery = should_use_battery_for_offpeak_load(cfg)
    max_grid_import_kw = float((cfg or {}).get("max_grid_import_kw", 0.0))
    grid_import_cap_active = max_grid_import_kw > 0.0
    grid_import_cap_binding_events = 0
    grid_import_cap_load_exceeds_events = 0
    grid_import_cap_limited_charge_kwh_total = 0.0

    rows = []
    for ts in idx:
        step_h = 1.0
        soc_start_pct = (energy / BATTERY_KWH * 100.0) if BATTERY_KWH > 0 else 0.0
        load = float(loads.loc[ts])
        remaining_load = load
        batt_discharge_kwh = 0.0
        batt_charge_kwh = 0.0
        charging_grid_import = 0.0

        if night_load_from_battery and remaining_load > 0:
            available = max(0.0, energy - min_energy)
            discharge_power_limited = BATTERY_MAX_DISCHARGE_KW * step_h
            needed_from_batt = remaining_load / BATTERY_DISCHARGE_EFF if BATTERY_DISCHARGE_EFF > 0 else remaining_load
            discharge = min(available, needed_from_batt, discharge_power_limited)
            energy -= discharge
            batt_discharge_kwh = discharge
            delivered = discharge * BATTERY_DISCHARGE_EFF
            remaining_load = max(0.0, remaining_load - delivered)

        if bool(get_offpeak_mask(pd.DatetimeIndex([ts]), cfg).iloc[0]) and energy < charge_cutoff_energy - 1e-9:
            charge_grid_kwh = min(charge_kw * step_h, BATTERY_MAX_CHARGE_KW * step_h)
            room = max(0.0, min(max_energy, charge_cutoff_energy) - energy)
            charge_to_battery = min(room, charge_grid_kwh * BATTERY_AC_CHARGE_EFF)
            if grid_import_cap_active:
                cap_import_kwh = max_grid_import_kw * step_h
                load_import_kwh = remaining_load
                if load_import_kwh > cap_import_kwh + 1e-9:
                    grid_import_cap_load_exceeds_events += 1
                charge_import_headroom_kwh = max(0.0, cap_import_kwh - load_import_kwh)
                max_charge_to_battery_by_gridcap = charge_import_headroom_kwh * BATTERY_AC_CHARGE_EFF
                unclamped_charge_to_battery = charge_to_battery
                charge_to_battery = min(charge_to_battery, max_charge_to_battery_by_gridcap)
                if unclamped_charge_to_battery - charge_to_battery > 1e-9:
                    grid_import_cap_binding_events += 1
                    grid_import_cap_limited_charge_kwh_total += (unclamped_charge_to_battery - charge_to_battery)
            if charge_to_battery > 0:
                charging_grid_import = charge_to_battery / BATTERY_AC_CHARGE_EFF if BATTERY_AC_CHARGE_EFF > 0 else 0.0
                energy = min(max_energy, energy + charge_to_battery)
                batt_charge_kwh = charge_to_battery

        grid_import = remaining_load + charging_grid_import
        soc_end_pct = (energy / BATTERY_KWH * 100.0) if BATTERY_KWH > 0 else soc_start_pct
        rows.append({
            "ts_local": ts,
            "load_kwh": float(load),
            "grid_import_kwh": float(grid_import),
            "batt_discharge_kwh": float(batt_discharge_kwh),
            "batt_charge_kwh": float(batt_charge_kwh),
            "soc_start_pct": float(soc_start_pct),
            "soc_end_pct": float(soc_end_pct),
            "pv_to_load_kwh": 0.0,
            "grid_export_kwh": 0.0,
            "curtailed_kwh": 0.0,
        })

    night_df = pd.DataFrame(rows).set_index("ts_local")
    night_df.attrs["grid_import_cap_active"] = bool(grid_import_cap_active)
    night_df.attrs["grid_import_cap_binding_events"] = int(grid_import_cap_binding_events)
    night_df.attrs["grid_import_cap_load_exceeds_events"] = int(grid_import_cap_load_exceeds_events)
    night_df.attrs["grid_import_cap_limited_charge_kwh_total"] = float(grid_import_cap_limited_charge_kwh_total)
    if ENABLE_INVARIANT_CHECKS:
        validate_flow_invariants(night_df, "night")
    return night_df


def simulate_full_day_soc(
    df: "pd.DataFrame",
    total_consumption_kwh: float,
    soc_at_22: float,
    charge_kw: float,
    cutoff_soc: float,
    tomorrow_date: dt.date,
    tariff_cfg: Optional[dict] = None,
    pv_col: str = "pv_total_kwh",
) -> Tuple["pd.Series", "pd.DataFrame"]:
    if pv_col not in df.columns:
        raise KeyError(f"simulate_full_day_soc missing pv_col={pv_col!r} in df columns")

    tomorrow_start = pd.Timestamp(dt.datetime.combine(tomorrow_date, dt.time(0, 0)), tz=TIMEZONE)
    tomorrow_end = tomorrow_start + dt.timedelta(days=1)
    tomorrow_idx = pd.date_range(tomorrow_start, tomorrow_end, freq="h", inclusive="left", tz=TIMEZONE)

    cfg = tariff_cfg or DEFAULT_CONFIG["tariff"]
    window_start, window_end = compute_charging_window_for_target_date(tomorrow_date, cfg)

    cycle_loads = build_cycle_hourly_load_series(tomorrow_date, total_consumption_kwh, tariff_cfg=cfg)
    night_loads = pd.to_numeric(cycle_loads.reindex(pd.date_range(window_start, window_end, freq="h", inclusive="left", tz=TIMEZONE)), errors="coerce").fillna(0.0).astype(float)

    night_df = simulate_night_charging_series(
        soc_at_22,
        charge_kw,
        cutoff_soc,
        session_start=window_start,
        session_end=window_end,
        total_consumption_kwh=total_consumption_kwh,
        tariff_cfg=cfg,
        precomputed_loads=night_loads,
    )
    day_start_ts = window_end
    soc_day_start = float(night_df.iloc[-1]["soc_end_pct"]) / 100.0 if not night_df.empty else soc_at_22

    day_idx = pd.date_range(day_start_ts, tomorrow_end, freq="h", inclusive="left", tz=TIMEZONE)
    dt_h = timestep_hours(day_idx)
    loads = pd.to_numeric(cycle_loads.reindex(day_idx), errors="coerce").fillna(0.0).astype(float)
    pv_total = pd.to_numeric(df[pv_col].reindex(day_idx), errors="coerce").fillna(0.0)
    pv_unclipped = pd.to_numeric(df.get("pv_total_unclipped_kwh", pv_total).reindex(day_idx), errors="coerce")
    pv_unclipped = pv_unclipped.combine_first(pv_total).fillna(0.0)

    if ENABLE_INVARIANT_CHECKS:
        load_cycle_total_kwh = float(cycle_loads.sum())
        load_night_kwh = float(night_loads.sum())
        load_day_kwh = float(loads.sum())
        if abs(load_cycle_total_kwh - float(total_consumption_kwh)) > 1e-6:
            raise ValueError("Cycle load total does not match target consumption.")
        if abs((load_night_kwh + load_day_kwh) - float(total_consumption_kwh)) > 1e-6:
            raise ValueError("Night + day load total does not match target consumption.")

    energy = max(0.0, min(1.0, soc_day_start)) * BATTERY_KWH
    min_energy = MIN_SOC * BATTERY_KWH
    max_energy = MAX_CUTOFF_SOC * BATTERY_KWH
    charge_cutoff_energy = min(MAX_CUTOFF_SOC, max(MIN_SOC, cutoff_soc)) * BATTERY_KWH

    rows = []
    econ_eps = 1e-6
    econ_cfg = cfg if isinstance(cfg, dict) else {}
    inj_raw = econ_cfg.get("injection_grid_price_eur_per_kwh")
    peak_raw = econ_cfg.get("peak_grid_price_eur_per_kwh")
    offpeak_raw = econ_cfg.get("offpeak_grid_price_eur_per_kwh")
    try:
        export_price_now = float(inj_raw)
        peak_price = float(peak_raw)
        offpeak_price = float(offpeak_raw)
        pv_surplus_store_econ_enabled = True
    except Exception:
        export_price_now = 0.0
        peak_price = 0.0
        offpeak_price = 0.0
        pv_surplus_store_econ_enabled = False

    use_battery_for_offpeak_load = should_use_battery_for_offpeak_load(cfg)
    if not get_expensive_windows(tomorrow_date, cfg):
        # No expensive hours exist (all-day off-peak). Stored PV still displaces
        # future grid import at off-peak price.
        use_battery_for_offpeak_load = True
    future_expensive_exists: dict[pd.Timestamp, bool] = {}
    has_future_expensive = False
    for ts in reversed(day_idx):
        is_expensive = in_any_window(ts.time(), get_expensive_windows(ts.date(), cfg))
        has_future_expensive = bool(has_future_expensive or is_expensive)
        future_expensive_exists[ts] = has_future_expensive

    pv_surplus_export_preferred_kwh = 0.0
    pv_surplus_store_preferred_kwh = 0.0
    pv_store_vs_export_decisions_count = 0
    allow_injection = bool(econ_cfg.get("allow_injection_to_grid", True))
    blocked_export_kwh_total = 0.0
    max_grid_import_kw = float((cfg or {}).get("max_grid_import_kw", 0.0))
    grid_import_cap_active = max_grid_import_kw > 0.0
    grid_import_cap_binding_events = int(night_df.attrs.get("grid_import_cap_binding_events", 0))
    grid_import_cap_load_exceeds_events = int(night_df.attrs.get("grid_import_cap_load_exceeds_events", 0))
    grid_import_cap_limited_charge_kwh_total = float(night_df.attrs.get("grid_import_cap_limited_charge_kwh_total", 0.0))
    for ts in day_idx:
        step_h = float(dt_h.loc[ts])
        pv_ac_limited = float(pv_total.loc[ts])
        pv_unclip = float(max(pv_unclipped.loc[ts], pv_ac_limited))
        load = float(loads.loc[ts])

        soc_start_pct = (energy / BATTERY_KWH * 100.0) if BATTERY_KWH > 0 else 0.0
        pv_to_load = min(pv_ac_limited, load)
        remaining_load = max(0.0, load - pv_to_load)
        pv_after_load = max(0.0, pv_ac_limited - pv_to_load)
        overflow = max(0.0, pv_unclip - pv_ac_limited)

        batt_charge_kwh = 0.0
        batt_discharge_kwh = 0.0
        charging_grid_import = 0.0
        grid_export = 0.0
        curtailed = 0.0
        pv_export_preferred_kwh = 0.0
        pv_store_preferred_kwh = 0.0

        pv_for_storage = pv_after_load + overflow
        if pv_for_storage > 0:
            room = max(0.0, max_energy - energy)
            charge_power_limited = BATTERY_MAX_CHARGE_KW * step_h
            pv_storage_headroom = room / BATTERY_PV_CHARGE_EFF if BATTERY_PV_CHARGE_EFF > 0 else 0.0
            pv_charge_power_cap = charge_power_limited / BATTERY_PV_CHARGE_EFF if BATTERY_PV_CHARGE_EFF > 0 else 0.0
            max_pv_to_store = max(0.0, min(pv_for_storage, pv_storage_headroom, pv_charge_power_cap))

            prefer_export = False
            if allow_injection and pv_surplus_store_econ_enabled and max_pv_to_store > 0:
                expected_displacement_price = peak_price if future_expensive_exists.get(ts, False) else offpeak_price
                stored_value_per_kwh_pv = expected_displacement_price * BATTERY_PV_CHARGE_EFF * BATTERY_DISCHARGE_EFF
                prefer_export = export_price_now >= (stored_value_per_kwh_pv + econ_eps)
                pv_store_vs_export_decisions_count += 1
                if prefer_export:
                    pv_export_preferred_kwh = max_pv_to_store
                    pv_surplus_export_preferred_kwh += max_pv_to_store
                else:
                    pv_store_preferred_kwh = max_pv_to_store
                    pv_surplus_store_preferred_kwh += max_pv_to_store

            if prefer_export:
                store = 0.0
                pv_after_batt = pv_for_storage
            else:
                pv_limited_store = pv_for_storage * BATTERY_PV_CHARGE_EFF
                store = min(room, pv_limited_store, charge_power_limited)
                energy += store
                batt_charge_kwh += store
                pv_used_for_batt = store / BATTERY_PV_CHARGE_EFF if BATTERY_PV_CHARGE_EFF > 0 else 0.0
                pv_after_batt = max(0.0, pv_for_storage - pv_used_for_batt)

            export_limit = max(0.0, (INVERTER_AC_KW_LIMIT * step_h) - pv_to_load)
            effective_export_limit = export_limit if allow_injection else 0.0
            grid_export = min(pv_after_batt, effective_export_limit)
            if not allow_injection:
                blocked_export_kwh_total += min(pv_after_batt, export_limit)
            curtailed = max(0.0, pv_after_batt - grid_export)

        offpeak = in_any_window(ts.time(), get_offpeak_windows_for_date(ts.date(), cfg))
        allow_batt_for_load = (not offpeak) or use_battery_for_offpeak_load

        if allow_batt_for_load and remaining_load > 0:
            available = max(0.0, energy - min_energy)
            discharge_power_limited = BATTERY_MAX_DISCHARGE_KW * step_h
            needed_from_batt = remaining_load / BATTERY_DISCHARGE_EFF if BATTERY_DISCHARGE_EFF > 0 else remaining_load
            discharge = min(available, needed_from_batt, discharge_power_limited)
            energy -= discharge
            batt_discharge_kwh = discharge
            delivered = discharge * BATTERY_DISCHARGE_EFF
            remaining_load = max(0.0, remaining_load - delivered)

        if offpeak and energy < charge_cutoff_energy - 1e-9:
            charge_grid_kwh = min(charge_kw * step_h, BATTERY_MAX_CHARGE_KW * step_h)
            room = max(0.0, min(max_energy, charge_cutoff_energy) - energy)
            charge_to_battery = min(room, charge_grid_kwh * BATTERY_AC_CHARGE_EFF)
            if grid_import_cap_active:
                cap_import_kwh = max_grid_import_kw * step_h
                load_import_kwh = remaining_load
                if load_import_kwh > cap_import_kwh + 1e-9:
                    grid_import_cap_load_exceeds_events += 1
                charge_import_headroom_kwh = max(0.0, cap_import_kwh - load_import_kwh)
                max_charge_to_battery_by_gridcap = charge_import_headroom_kwh * BATTERY_AC_CHARGE_EFF
                unclamped_charge_to_battery = charge_to_battery
                charge_to_battery = min(charge_to_battery, max_charge_to_battery_by_gridcap)
                if unclamped_charge_to_battery - charge_to_battery > 1e-9:
                    grid_import_cap_binding_events += 1
                    grid_import_cap_limited_charge_kwh_total += (unclamped_charge_to_battery - charge_to_battery)
            if charge_to_battery > 0:
                charging_grid_import = charge_to_battery / BATTERY_AC_CHARGE_EFF if BATTERY_AC_CHARGE_EFF > 0 else 0.0
                energy = min(max_energy, energy + charge_to_battery)
                batt_charge_kwh += charge_to_battery

        grid_import = remaining_load + charging_grid_import
        soc_end_pct = (energy / BATTERY_KWH * 100.0) if BATTERY_KWH > 0 else soc_start_pct
        rows.append({
            "time": ts,
            "load_kwh": load,
            "pv_to_load_kwh": pv_to_load,
            "batt_charge_kwh": batt_charge_kwh,
            "batt_discharge_kwh": batt_discharge_kwh,
            "grid_import_kwh": grid_import,
            "grid_export_kwh": grid_export,
            "curtailed_kwh": curtailed,
            "pv_export_preferred_kwh": pv_export_preferred_kwh,
            "pv_store_preferred_kwh": pv_store_preferred_kwh,
            "soc_start_pct": soc_start_pct,
            "soc_end_pct": soc_end_pct,
        })

    day_flows_df = pd.DataFrame(rows).set_index("time")
    night_tomorrow_df = night_df[(night_df.index >= tomorrow_start) & (night_df.index < day_start_ts)]
    flows_df = pd.concat([night_tomorrow_df, day_flows_df]).sort_index()
    flows_df = flows_df[~flows_df.index.duplicated(keep="last")]
    flows_df = flows_df.reindex(tomorrow_idx).fillna(0.0)
    if ENABLE_INVARIANT_CHECKS and not flows_df.index.is_unique:
        raise ValueError("full_day flows contains duplicate timestamps")
    flows_df.attrs["pv_surplus_store_econ_enabled"] = bool(pv_surplus_store_econ_enabled)
    flows_df.attrs["pv_surplus_export_preferred_kwh"] = float(pv_surplus_export_preferred_kwh)
    flows_df.attrs["pv_surplus_store_preferred_kwh"] = float(pv_surplus_store_preferred_kwh)
    flows_df.attrs["pv_store_vs_export_decisions_count"] = int(pv_store_vs_export_decisions_count)
    flows_df.attrs["allow_injection_to_grid"] = allow_injection
    flows_df.attrs["export_blocked_by_policy"] = bool(not allow_injection)
    flows_df.attrs["blocked_export_kwh_total"] = float(blocked_export_kwh_total)
    flows_df.attrs["max_grid_import_kw"] = float(max_grid_import_kw)
    flows_df.attrs["grid_import_cap_active"] = bool(grid_import_cap_active)
    flows_df.attrs["grid_import_cap_binding_events"] = int(grid_import_cap_binding_events)
    flows_df.attrs["grid_import_cap_load_exceeds_events"] = int(grid_import_cap_load_exceeds_events)
    flows_df.attrs["grid_import_cap_limited_charge_kwh_total"] = float(grid_import_cap_limited_charge_kwh_total)
    soc_series = pd.Series(pd.to_numeric(flows_df["soc_end_pct"], errors="coerce").fillna(0.0).values, index=tomorrow_idx, name="soc_percent")
    if ENABLE_INVARIANT_CHECKS:
        validate_flow_invariants(flows_df, "full_day")
    return soc_series, flows_df


def run_planner(inputs: PlannerInputs) -> PlannerOutput:
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    cfg = get_effective_config()
    return run_forecast_pipeline(
        cfg=cfg,
        target_date=tomorrow,
        soc_at_22_percent=float(inputs.soc_at_22) * 100.0,
        yesterday_kwh=float(inputs.yesterday_consumption_kwh),
        buffer_percent=float(inputs.forecast_buffer_soc) * 100.0,
        user_max_ac_kw=float(cfg["battery"].get("max_ac_charge_kw_hard_limit", MAX_AC_CHARGE_KW_HARD_LIMIT)),
    )

# ============================================================
# OUTPUT
# ============================================================

def print_hourly_pv(df: "pd.DataFrame", sunrise: dt.datetime, sunset: dt.datetime) -> None:
    df = df.sort_index()

    print("\nHour | PV_EAST | PV_SOUTH | PV_UNCL | PV_CLIP | PV_TOTAL | LOAD  | SURPLUS | DEFICIT | Sun%")
    print("-----+--------+----------+---------+---------+----------+-------+---------+---------+-----")

    for ts, row in df.iterrows():
        print(
            f"{ts.strftime('%H:00')} | "
            f"{float(row['pv_east_kwh']):>6.2f} | {float(row['pv_south_kwh']):>8.2f} | {float(row.get('pv_total_unclipped_kwh', row['pv_total_kwh'])):>7.2f} | "
            f"{float(row.get('pv_clipped_kwh', 0.0)):>7.2f} | {float(row['pv_total_kwh']):>8.2f} | "
            f"{float(row['load_kwh']):>5.2f} | {float(row['pv_surplus_kwh']):>7.2f} | {float(row['pv_deficit_kwh']):>7.2f} | "
            f"{int(row.get('sun_percent', 0)):>3d}%"
        )

    print("\nDay totals")
    print(f"- Total PV EAST:  {float(df['pv_east_kwh'].sum()):.2f} kWh")
    print(f"- Total PV SOUTH: {float(df['pv_south_kwh'].sum()):.2f} kWh")
    print(f"- Total PV unclipped: {float(df.get('pv_total_unclipped_kwh', df['pv_total_kwh']).sum()):.2f} kWh")
    print(f"- Total PV clipped:   {float(df.get('pv_clipped_kwh', 0.0).sum()):.2f} kWh")
    print(f"- Total PV:           {float(df['pv_total_kwh'].sum()):.2f} kWh")
    print(f"- Total LOAD est: {float(df['load_kwh'].sum()):.2f} kWh")
    print(f"- Total SURPLUS:  {float(df['pv_surplus_kwh'].sum()):.2f} kWh")
    print(f"- Total DEFICIT:  {float(df['pv_deficit_kwh'].sum()):.2f} kWh")

    if PRINT_MODE == "detail":
        _, _, daylight_mask = normalize_daylight_window(df.index, sunrise, sunset)
        daylight = daylight_mask.values
        daylight_hours = int(daylight.sum())
        if daylight_hours > 0:
            sun_hours = int((df.loc[daylight, "pv_total_kwh"] >= SUN_HOUR_THRESHOLD_KWH).sum())
            sun_pct = (sun_hours / daylight_hours * 100.0)
            print(f"- Sun hours (PV_TOTAL >= {SUN_HOUR_THRESHOLD_KWH} kWh/h): {sun_hours} h ({sun_pct:.1f}%)")


def print_expensive_hourly_flow(detail_df: "pd.DataFrame", for_date: dt.date) -> None:
    if PRINT_MODE != "detail":
        return

    expensive_windows = get_expensive_windows(for_date)
    if detail_df is None or len(detail_df) == 0 or not expensive_windows:
        print("\n(No expensive-hour details: no EXPENSIVE windows on this day.)")
        return

    print("\n=======================================")
    print("HOURLY DETAIL (EXPENSIVE HOURS)")
    print("=======================================")
    print("Hour | PV_E | PV_S | PV_T | Load | Sur+ | Def- | Batt+ | Batt- | Grid+ | Grid- | SOC% start -> end")
    print("-----+------+------+------+------+------+------+-------+-------+-------+-------+------------------")

    for ts, r in detail_df.iterrows():
        print(
            f"{ts.strftime('%H:00')} | "
            f"{r['pv_east_kwh']:>4.2f} | {r['pv_south_kwh']:>4.2f} | {r['pv_total_kwh']:>4.2f} | "
            f"{r['load_kwh']:>4.2f} | {r['surplus_kwh']:>4.2f} | {r['deficit_kwh']:>4.2f} | "
            f"{r['batt_charge_kwh']:>5.2f} | {r['batt_discharge_kwh']:>5.2f} | "
            f"{r['grid_import_kwh']:>5.2f} | {r['grid_export_kwh']:>5.2f} | "
            f"{r['soc_start_pct']:>6.1f}% -> {r['soc_end_pct']:>6.1f}%"
        )

    print("\nExpensive-hour totals:")
    print(f"- Total grid import: {detail_df['grid_import_kwh'].sum():.2f} kWh")
    print(f"- Total grid export: {detail_df['grid_export_kwh'].sum():.2f} kWh")


def print_fusionsolar_actions(cutoff_soc: float, charge_kw: float, cutoff_note: str = "") -> None:
    print(format_fusionsolar_actions(cutoff_soc, charge_kw))
    if cutoff_note:
        print(cutoff_note)


def format_fusionsolar_actions(cutoff_soc: float, charge_kw: float) -> str:
    return (
        f"Allowed AC charge power (kW): {charge_kw:.2f}\n"
        f"AC charge cutoff SOC (%): {cutoff_soc*100:.1f}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    quick_sanity_checks()

    soc_at_22_percent = ask_required_soc("Battery SOC at 22:00 from FusionSolar (%): ") * 100.0

    while True:
        yesterday_consumption_kwh = ask_required_float("Total consumption yesterday from FusionSolar (kWh): ")
        if yesterday_consumption_kwh > 0:
            break
        print("Consumption must be > 0.")

    tomorrow = dt.date.today() + dt.timedelta(days=1)
    cfg = get_effective_config()
    out = run_forecast_pipeline(
        cfg=cfg,
        target_date=tomorrow,
        soc_at_22_percent=soc_at_22_percent,
        yesterday_kwh=yesterday_consumption_kwh,
        buffer_percent=0.0,
        user_max_ac_kw=float(cfg["battery"].get("max_ac_charge_kw_hard_limit", MAX_AC_CHARGE_KW_HARD_LIMIT)),
    )

    print(f"Location: {out.location.latitude:.5f}, {out.location.longitude:.5f}")
    print("Inputs:")
    print(f"- SOC at 22:00 (%): {soc_at_22_percent:.1f}")
    print(f"- Yesterday consumption (kWh): {yesterday_consumption_kwh:.2f}")
    print(f"- Inverter AC limit (kW): {INVERTER_AC_KW_LIMIT:.2f}")
    print(f"- Battery max charge/discharge (kW): {BATTERY_MAX_CHARGE_KW:.2f}/{BATTERY_MAX_DISCHARGE_KW:.2f}")
    print(
        f"Sunrise/Sunset: {out.weather.sunrise.strftime('%Y-%m-%d %H:%M %Z')} / "
        f"{out.weather.sunset.strftime('%Y-%m-%d %H:%M %Z')}"
    )

    print_hourly_pv(out.hourly_df, out.weather.sunrise, out.weather.sunset)
    print_expensive_hourly_flow(out.expensive_detail_df, tomorrow)
    print_fusionsolar_actions(out.cutoff_soc, out.charge_kw, out.cutoff_note)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
