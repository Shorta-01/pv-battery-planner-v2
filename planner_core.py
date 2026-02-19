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
PV_IAM_MODEL = "none"
PV_IAM_ASHRAE_B = 0.05
PV_ALBEDO: float | None = None
INVERTER_AC_MODEL = "linear"
PV_CALIBRATION_FACTOR = 1.00
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
        "pv_calibration_factor": PV_CALIBRATION_FACTOR,
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
    "load_profile": {
        "load_profile_24h": LOAD_PROFILE,
    },
    "tariff": {
        "peak_grid_price_eur_per_kwh": PEAK_GRID_PRICE_EUR_PER_KWH,
        "offpeak_grid_price_eur_per_kwh": OFFPEAK_GRID_PRICE_EUR_PER_KWH,
        "injection_grid_price_eur_per_kwh": INJECTION_GRID_PRICE_EUR_PER_KWH,
        "optimization_mode": "window_only",
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
    iam_ashrae_b = float(pv.get("iam_ashrae_b", 0.05))
    if not (0.0 <= iam_ashrae_b <= 0.5):
        raise ValueError("pv.iam_ashrae_b must be in [0.0, 0.5].")
    if "albedo" in pv and pv.get("albedo") is not None:
        albedo = float(pv["albedo"])
        if not (0.0 <= albedo <= 1.0):
            raise ValueError("pv.albedo must be in [0.0, 1.0] when set.")
    inverter_ac_model = str(pv.get("inverter_ac_model", "linear")).strip().lower()
    if inverter_ac_model not in {"linear", "pvwatts"}:
        raise ValueError("pv.inverter_ac_model must be either 'linear' or 'pvwatts'.")
    pv_calibration_factor = float(pv.get("pv_calibration_factor", 1.0))
    if not (0.7 <= pv_calibration_factor <= 1.3):
        raise ValueError("pv.pv_calibration_factor must be in [0.7, 1.3].")
    pv_calibration_factor_east = float(pv.get("pv_calibration_factor_east", 1.0))
    pv_calibration_factor_south = float(pv.get("pv_calibration_factor_south", 1.0))
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
    if int(pv["array_south_panels"]) <= 0 or int(pv["array_east_panels"]) <= 0:
        raise ValueError("pv.array_south_panels and pv.array_east_panels must be > 0.")
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
    global PV_CALIBRATION_FACTOR, PV_CALIBRATION_FACTOR_EAST, PV_CALIBRATION_FACTOR_SOUTH, INVERTER_AC_KW_LIMIT
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
    INVERTER_EFF = float(pv["inverter_eff"])
    PV_LOSS_MODEL = str(pv.get("loss_model", pv.get("pv_loss_model", "split"))).strip().lower()
    PV_IAM_MODEL = str(pv.get("iam_model", "none")).strip().lower()
    PV_IAM_ASHRAE_B = float(pv.get("iam_ashrae_b", 0.05))
    PV_ALBEDO = None if pv.get("albedo") is None else float(pv.get("albedo"))
    INVERTER_AC_MODEL = str(pv.get("inverter_ac_model", "linear")).strip().lower()
    PV_CALIBRATION_FACTOR = float(pv.get("pv_calibration_factor", 1.0))
    base_calibration_factor_east = float(pv.get("pv_calibration_factor_east", 1.0))
    base_calibration_factor_south = float(pv.get("pv_calibration_factor_south", 1.0))
    PV_CALIBRATION_FACTOR_EAST = PV_CALIBRATION_FACTOR * base_calibration_factor_east
    PV_CALIBRATION_FACTOR_SOUTH = PV_CALIBRATION_FACTOR * base_calibration_factor_south
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


def get_offpeak_mask(index: pd.DatetimeIndex, target_date: dt.date, cfg: Optional[dict] = None) -> pd.Series:
    windows = get_offpeak_windows_for_date(target_date, cfg)
    if len(index) == 0 or not windows:
        return pd.Series(False, index=index, dtype=bool)

    t0 = pd.Timestamp(dt.datetime.combine(target_date, dt.time(0, 0)), tz=TIMEZONE)
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


def get_charge_session_index(charge_date: dt.date) -> pd.DatetimeIndex:
    start = pd.Timestamp(dt.datetime.combine(charge_date, dt.time(0, 0)), tz=TIMEZONE)
    end = start + dt.timedelta(days=2)
    idx = pd.date_range(start, end, freq="h", inclusive="left", tz=TIMEZONE)
    mask = get_offpeak_mask(idx, charge_date)
    return idx[mask.to_numpy()]


def get_charge_windows(charge_date: dt.date) -> List[Tuple[str, str]]:
    return get_offpeak_windows(charge_date)


def import_price_eur_per_kwh(ts: pd.Timestamp, tariff_cfg: dict) -> float:
    offpeak = float(tariff_cfg.get("offpeak_grid_price_eur_per_kwh", 0.0))
    peak = float(tariff_cfg.get("peak_grid_price_eur_per_kwh", 0.0))
    windows = get_offpeak_windows(ts.date())
    return offpeak if in_any_window(ts.time(), windows) else peak


def fmt_windows(windows: List[Tuple[str, str]]) -> str:
    if not windows:
        return "none"
    return ", ".join([f"{start}–{end}" for start, end in windows])


def overnight_charge_hours_summary(charge_date: dt.date) -> tuple[float, str]:
    session_idx = get_charge_session_index(charge_date)
    available_charge_hours = float(len(session_idx))
    return available_charge_hours, f"{fmt_windows(get_charge_windows(charge_date))}: {available_charge_hours:.1f}h off-peak"


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


def get_expensive_windows(for_date: dt.date) -> List[Tuple[str, str]]:
    # Hoog tarief = complement van daluren
    return complement_windows(get_offpeak_windows(for_date))


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
    Returns:
      - baseline_cost_eur_total (includes configured pre-tomorrow off-peak charge session + tomorrow 00–24)
      - plan_cost_eur_total     (includes configured pre-tomorrow off-peak charge session + tomorrow 00–24)
      - savings_eur_total       (= baseline − plan)
      - baseline_cost_eur_tomorrow (tomorrow 00–24 only)
      - plan_cost_eur_tomorrow     (tomorrow 00–24 only)
      - savings_eur_tomorrow       (tomorrow 00–24 only)
      - hourly_savings_eur_tomorrow: list[24] aligned to 00:00..23:00
    """
    inj = float(tariff_cfg.get("injection_grid_price_eur_per_kwh", 0.0))

    tomorrow_start = pd.Timestamp(dt.datetime.combine(tomorrow_date, dt.time(0, 0)), tz=TIMEZONE)
    tomorrow_end = tomorrow_start + dt.timedelta(days=1)

    idx_tomorrow = pd.date_range(tomorrow_start, tomorrow_end, freq="h", inclusive="left", tz=TIMEZONE)
    charge_session_idx = get_charge_session_index(today_date)
    idx_tonight = charge_session_idx[charge_session_idx < tomorrow_start]

    dt_h_tomorrow = timestep_hours(idx_tomorrow)
    dt_h_tonight = timestep_hours(idx_tonight)

    pv_tomorrow = pv_df["pv_total_kwh"].reindex(idx_tomorrow).fillna(0.0).astype(float)
    load_tomorrow = pd.Series(
        [load_kwh_at(ts, total_consumption_kwh, float(dt_h_tomorrow.loc[ts])) for ts in idx_tomorrow],
        index=idx_tomorrow,
        dtype=float,
    )
    load_tonight = pd.Series(
        [load_kwh_at(ts, total_consumption_kwh, float(dt_h_tonight.loc[ts])) for ts in idx_tonight],
        index=idx_tonight,
        dtype=float,
    )

    base_import_tom = (load_tomorrow - pv_tomorrow).clip(lower=0.0)
    base_export_tom = (pv_tomorrow - load_tomorrow).clip(lower=0.0)
    base_price_tom = pd.Series([import_price_eur_per_kwh(ts, tariff_cfg) for ts in idx_tomorrow], index=idx_tomorrow)
    base_cost_tom = base_import_tom * base_price_tom - base_export_tom * inj

    plan_import_raw = flows_df["grid_import_kwh"].reindex(idx_tomorrow).fillna(0.0).astype(float)
    plan_export_raw = flows_df["grid_export_kwh"].reindex(idx_tomorrow).fillna(0.0).astype(float)
    offpeak_tomorrow_mask = get_offpeak_mask(idx_tomorrow, tomorrow_date, tariff_cfg)

    plan_import_tom = plan_import_raw.copy()
    plan_export_tom = plan_export_raw.copy()
    plan_import_tom[offpeak_tomorrow_mask] = base_import_tom[offpeak_tomorrow_mask] + plan_import_raw[offpeak_tomorrow_mask]
    plan_export_tom[offpeak_tomorrow_mask] = base_export_tom[offpeak_tomorrow_mask]

    plan_cost_tom = plan_import_tom * base_price_tom - plan_export_tom * inj

    night_df = simulate_night_charging_series(soc_at_22, charge_kw, cutoff_soc, tomorrow_date)
    charge_tonight = night_df["grid_import_kwh"].reindex(idx_tonight).fillna(0.0).astype(float)

    base_price_ton = pd.Series([import_price_eur_per_kwh(ts, tariff_cfg) for ts in idx_tonight], index=idx_tonight)
    base_cost_ton = load_tonight * base_price_ton
    plan_cost_ton = (load_tonight + charge_tonight) * base_price_ton

    baseline_total = float(base_cost_ton.sum() + base_cost_tom.sum())
    plan_total = float(plan_cost_ton.sum() + plan_cost_tom.sum())
    baseline_tom = float(base_cost_tom.sum())
    plan_tom = float(plan_cost_tom.sum())
    hourly_savings = (base_cost_tom - plan_cost_tom).reindex(idx_tomorrow).fillna(0.0)

    return {
        "baseline_cost_eur_total": baseline_total,
        "plan_cost_eur_total": plan_total,
        "savings_eur_total": baseline_total - plan_total,
        "baseline_cost_eur_tomorrow": baseline_tom,
        "plan_cost_eur_tomorrow": plan_tom,
        "savings_eur_tomorrow": baseline_tom - plan_tom,
        "hourly_savings_eur_tomorrow": [float(hourly_savings.loc[ts]) for ts in idx_tomorrow],
    }


def quick_sanity_checks() -> None:
    try:
        assert ARRAY_EAST_PANELS > 0 and ARRAY_SOUTH_PANELS > 0
        assert 0 < PERFORMANCE_RATIO <= 1
        assert 0 < INVERTER_EFF <= 1
        assert PV_LOSS_MODEL in {"split", "combined"}
        assert 0.7 <= PV_CALIBRATION_FACTOR <= 1.3
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

    day_start = pd.Timestamp(dt.datetime.combine(date, dt.time(0, 0)), tz=tz)
    next_day = pd.Timestamp(dt.datetime.combine(date + dt.timedelta(days=1), dt.time(0, 0)), tz=tz)
    expected_index = pd.date_range(day_start, next_day, freq="h", inclusive="left")
    out = out.reindex(expected_index)

    irr_cols = [c for c in ["ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_cover_pct"] if c in out.columns]
    for col in irr_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

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

def estimate_pv_with_pvlib(
    df: "pd.DataFrame",
    loc: Location,
    tz: str | None = None,
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

    solpos = pvloc.get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    def cloud_transmittance_from_cover(cloud_cover_pct: "pd.Series") -> "pd.Series":
        cloud_fraction = (pd.to_numeric(cloud_cover_pct, errors="coerce") / 100.0).clip(lower=0.0, upper=1.0)
        cloud_fraction = cloud_fraction.fillna(0.0)
        trans = 1.0 - (CLOUD_ATTENUATION_WEIGHT * (cloud_fraction ** CLOUD_ATTENUATION_EXPONENT))
        return trans.clip(lower=CLOUD_TRANSMITTANCE_MIN, upper=1.0)

    def derive_irradiance_from_ghi(ghi_in: "pd.Series") -> Tuple["pd.Series", "pd.Series", "pd.Series"]:
        ghi_s = pd.to_numeric(ghi_in, errors="coerce").reindex(df_local.index).fillna(0.0).clip(lower=0.0)
        repair_method = IRR_REPAIR_METHOD.lower()
        if repair_method == "erbs":
            decomp = pvlib.irradiance.erbs(ghi_s, solpos["apparent_zenith"], times)
            dni_s = pd.to_numeric(decomp["dni"], errors="coerce").fillna(0.0).clip(lower=0.0)
            dhi_s = pd.to_numeric(decomp["dhi"], errors="coerce").fillna(0.0).clip(lower=0.0)
        else:
            decomp = pvlib.irradiance.disc(ghi_s, solpos["apparent_zenith"], times)
            dni_s = pd.to_numeric(decomp["dni"], errors="coerce").fillna(0.0).clip(lower=0.0)
            cos_zen_local = pd.to_numeric(solpos["apparent_zenith"], errors="coerce").apply(
                lambda z: max(0.0, math.cos(math.radians(z))) if pd.notna(z) else 0.0
            )
            dhi_s = (ghi_s - (dni_s * cos_zen_local)).fillna(0.0).clip(lower=0.0)
        return ghi_s.astype(float), dni_s.astype(float), dhi_s.astype(float)

    irradiance_cols = ["ghi_wm2", "dni_wm2", "dhi_wm2"]
    missing_irr_cols = [c for c in irradiance_cols if c not in df_local.columns]
    irr_nan_ratio = float(df_local[irradiance_cols].isna().mean().mean()) if not missing_irr_cols else 1.0
    irradiance_anomalies = irradiance_sanity_warnings(df_local, loc, tz_use, model_id="pv_input")
    use_clearsky = bool(missing_irr_cols) or irr_nan_ratio > 0.5 or bool(irradiance_anomalies)
    if irradiance_anomalies:
        print(
            "Irradiance sanity fallback to clear-sky triggered: "
            + " | ".join(irradiance_anomalies)
        )

    if use_clearsky:
        provider_ghi = pd.to_numeric(df_local.get("ghi_wm2"), errors="coerce") if "ghi_wm2" in df_local.columns else pd.Series(np.nan, index=df_local.index)
        ghi_coverage = float(provider_ghi.notna().mean()) if len(provider_ghi) else 0.0
        if ghi_coverage >= 0.5:
            ghi, dni, dhi = derive_irradiance_from_ghi(provider_ghi)
        else:
            cs = pvloc.get_clearsky(times, model="ineichen")
            if "cloud_cover_pct" in df_local.columns:
                trans = cloud_transmittance_from_cover(df_local["cloud_cover_pct"])
                ghi_cloud = cs["ghi"] * trans
                if "ghi_wm2" in df_local.columns:
                    ghi_provider = pd.to_numeric(df_local["ghi_wm2"], errors="coerce")
                    daylight = cs["ghi"] > 20.0
                    valid_bias = daylight & ghi_provider.notna()
                    if bool(valid_bias.any()):
                        bias = (ghi_provider[valid_bias] / cs["ghi"][valid_bias].clip(lower=1.0)).median()
                        if pd.notna(bias):
                            bias = float(np.clip(bias, 0.6, 1.2))
                            ghi_cloud = ghi_cloud * bias
                ghi, dni, dhi = derive_irradiance_from_ghi(ghi_cloud)
            else:
                ghi, dni, dhi = derive_irradiance_from_ghi(cs["ghi"])
    else:
        ghi = pd.to_numeric(df_local["ghi_wm2"], errors="coerce")
        dni = pd.to_numeric(df_local["dni_wm2"], errors="coerce")
        dhi = pd.to_numeric(df_local["dhi_wm2"], errors="coerce")

    ghi = pd.to_numeric(ghi, errors="coerce").reindex(df_local.index).astype(float).fillna(0.0).clip(lower=0.0)
    dni = pd.to_numeric(dni, errors="coerce").reindex(df_local.index).astype(float).fillna(0.0).clip(lower=0.0)
    dhi = pd.to_numeric(dhi, errors="coerce").reindex(df_local.index).astype(float).fillna(0.0).clip(lower=0.0)

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
            dni = pd.to_numeric(dni, errors="coerce").reindex(df_local.index).astype(float).fillna(0.0).clip(lower=0.0)
            dhi = pd.to_numeric(dhi, errors="coerce").reindex(df_local.index).astype(float).fillna(0.0).clip(lower=0.0)
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
        temp_cell = pvlib.temperature.faiman(
            poa_global=poa,
            temp_air=temp_air,
            wind_speed=wind_speed,
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

    total_dc_kw = (east_dc_kw + south_dc_kw).fillna(0).clip(lower=0)
    if INVERTER_AC_MODEL == "pvwatts":
        total_kwp = max(dc_kwp(ARRAY_EAST_PANELS) + dc_kwp(ARRAY_SOUTH_PANELS), 1e-9)
        east_pdc0 = INVERTER_AC_KW_LIMIT * (dc_kwp(ARRAY_EAST_PANELS) / total_kwp)
        south_pdc0 = INVERTER_AC_KW_LIMIT * (dc_kwp(ARRAY_SOUTH_PANELS) / total_kwp)

        east_ac_kw_unclipped = pd.Series(
            pvlib.inverter.pvwatts(
                pdc=east_dc_kw * 1000.0,
                pdc0=(east_pdc0 * 1000.0) / max(ac_inv_eff_multiplier, 1e-6),
                eta_inv_nom=ac_inv_eff_multiplier,
            ),
            index=east_dc_kw.index,
            dtype=float,
        ) / 1000.0
        south_ac_kw_unclipped = pd.Series(
            pvlib.inverter.pvwatts(
                pdc=south_dc_kw * 1000.0,
                pdc0=(south_pdc0 * 1000.0) / max(ac_inv_eff_multiplier, 1e-6),
                eta_inv_nom=ac_inv_eff_multiplier,
            ),
            index=south_dc_kw.index,
            dtype=float,
        ) / 1000.0
        east_ac_kw_unclipped = east_ac_kw_unclipped.fillna(0.0).clip(lower=0.0)
        south_ac_kw_unclipped = south_ac_kw_unclipped.fillna(0.0).clip(lower=0.0)
    else:
        east_ac_kw_unclipped = (east_dc_kw * ac_inv_eff_multiplier).fillna(0).clip(lower=0)
        south_ac_kw_unclipped = (south_dc_kw * ac_inv_eff_multiplier).fillna(0).clip(lower=0)

    east_ac_kwh_unclipped = (east_ac_kw_unclipped * dt_h).fillna(0.0).clip(lower=0.0)
    south_ac_kwh_unclipped = (south_ac_kw_unclipped * dt_h).fillna(0.0).clip(lower=0.0)
    total_ac_kw_unclipped = (east_ac_kw_unclipped + south_ac_kw_unclipped).fillna(0.0).clip(lower=0.0)
    total_ac_kwh_unclipped = (east_ac_kwh_unclipped + south_ac_kwh_unclipped).fillna(0.0).clip(lower=0.0)

    clip_scale = (INVERTER_AC_KW_LIMIT / total_ac_kw_unclipped.replace(0.0, float("nan"))).fillna(1.0).clip(upper=1.0)
    east_ac_kw_clipped = (east_ac_kw_unclipped * clip_scale).fillna(0.0).clip(lower=0.0)
    south_ac_kw_clipped = (south_ac_kw_unclipped * clip_scale).fillna(0.0).clip(lower=0.0)
    total_ac_kw_clipped = (east_ac_kw_clipped + south_ac_kw_clipped).fillna(0.0).clip(lower=0.0, upper=INVERTER_AC_KW_LIMIT)

    east_ac_kwh_clipped = (east_ac_kw_clipped * dt_h).fillna(0.0).clip(lower=0.0)
    south_ac_kwh_clipped = (south_ac_kw_clipped * dt_h).fillna(0.0).clip(lower=0.0)
    total_ac_kwh_clipped = (east_ac_kwh_clipped + south_ac_kwh_clipped).fillna(0.0).clip(lower=0.0)

    east_ac_kwh_clipped = east_ac_kwh_clipped.where(avail)
    south_ac_kwh_clipped = south_ac_kwh_clipped.where(avail)
    total_ac_kwh_clipped = total_ac_kwh_clipped.where(avail)
    total_ac_kwh_unclipped = total_ac_kwh_unclipped.where(avail)

    return (
        east_ac_kwh_clipped.astype(float),
        south_ac_kwh_clipped.astype(float),
        total_ac_kwh_unclipped.astype(float),
        total_ac_kwh_clipped.astype(float),
    )

def build_pv_forecast(df: "pd.DataFrame", loc: Location, tz: str | None = None) -> "pd.DataFrame":
    if not PVLIB_AVAILABLE:
        raise SystemExit("pvlib is required. Install with: pip install pvlib")

    (
        east_ac_kwh_clipped,
        south_ac_kwh_clipped,
        total_ac_kwh_unclipped,
        total_ac_kwh_clipped,
    ) = estimate_pv_with_pvlib(df, loc, tz=tz)

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
        total = pd.to_numeric(out["pv_total_kwh"], errors="coerce").fillna(0.0)
        out["pv_east_kwh"] = (total * float(e_ratio)).astype(float)
        out["pv_south_kwh"] = (total * float(s_ratio)).astype(float)
    elif split_missing:
        out["pv_east_kwh"] = 0.0
        out["pv_south_kwh"] = pd.to_numeric(out["pv_total_kwh"], errors="coerce").fillna(0.0)

    if "pv_clipped_kwh" not in out.columns:
        out["pv_clipped_kwh"] = (out["pv_total_unclipped_kwh"] - out["pv_total_kwh"]).clip(lower=0.0)

    out["pv_total_kwh"] = pd.to_numeric(out["pv_total_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["pv_total_unclipped_kwh"] = pd.to_numeric(out["pv_total_unclipped_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["pv_total_unclipped_kwh"] = np.maximum(out["pv_total_unclipped_kwh"], out["pv_total_kwh"])
    out["pv_east_kwh"] = pd.to_numeric(out["pv_east_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["pv_south_kwh"] = pd.to_numeric(out["pv_south_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)

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
    if not (0.7 <= PV_CALIBRATION_FACTOR <= 1.3):
        raise ValueError("PV_CALIBRATION_FACTOR must be within [0.7, 1.3].")

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


def apply_daylight_clamp(df: "pd.DataFrame", sunrise: dt.datetime, sunset: dt.datetime) -> "pd.DataFrame":
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
) -> float:
    expensive_windows = get_expensive_windows(for_date)
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
        pv = float(df.loc[ts, "pv_total_kwh"])
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
        )

        weather = fetch_weather_for_date(loc, target_date, tz=tz)
        pv = build_pv_forecast(weather.df, loc, tz=tz)
        pv = apply_daylight_clamp(pv, weather.sunrise, weather.sunset).sort_index()
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
        cutoff_soc_raw, cutoff_reason = choose_cutoff_soc(target_date, soc_low, soc_high)
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
        )

        detail_df, grid_import, grid_export, _, _ = simulate_expensive_hours_detailed(
            pv, yesterday_kwh, achieved_soc_start, target_date
        )
        full_soc, full_flows = simulate_full_day_soc(
            pv,
            yesterday_kwh,
            soc_at_22_percent / 100.0,
            charge_kw,
            cutoff_soc,
            target_date,
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

def compute_soc_high_headroom(
    df: "pd.DataFrame",
    total_consumption_kwh: float,
    for_date: dt.date,
    sunrise: dt.datetime,
    sunset: dt.datetime,
) -> Tuple[float, float]:
    """
    Headroom doel: hoeveel PV-overschot verwacht je BINNEN daglichturen.
    Hoe meer overschot, hoe lager je bij start van hoog tarief wil zitten om injectie te vermijden.
    """
    _ = for_date

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

        pv_ac_kwh = float(df.loc[ts, "pv_total_kwh"])
        load_ac_kwh = float(loads.loc[ts])
        surplus_ac_kwh = max(0.0, pv_ac_kwh - load_ac_kwh)
        surplus_sum_ac += surplus_ac_kwh

        stored_candidate_kwh = surplus_ac_kwh * BATTERY_PV_CHARGE_EFF
        max_store_kwh = float(BATTERY_MAX_CHARGE_KW) * float(dt_h.loc[ts])
        stored_kwh_sum += min(stored_candidate_kwh, max_store_kwh)

    soc_high = 1.0 - (stored_kwh_sum / BATTERY_KWH)
    soc_high = min(max(soc_high, MIN_SOC), 1.0)
    return surplus_sum_ac, soc_high


def choose_cutoff_soc(for_date: dt.date, soc_low: float, soc_high: float) -> Tuple[float, str]:
    expensive_windows = get_expensive_windows(for_date)
    if not expensive_windows:
        return MIN_SOC, "No expensive hours (all off-peak): keep cutoff low for maximum headroom."

    if soc_low <= soc_high + 1e-9:
        return soc_low, "OK: bridge expensive hours and keep headroom to reduce export."
    return soc_high, (
        "CONFLICT: required SOC to bridge expensive hours is higher than PV headroom target. "
        "Using headroom target to avoid morning PV export; expect some grid import later if PV underperforms."
    )


def plan_charge_power(soc_start: float, soc_cutoff: float, charge_date: dt.date, user_cap_kw: Optional[float] = None) -> Tuple[float, float, str, float]:
    session_idx = get_charge_session_index(charge_date)
    available_charge_hours = float(len(session_idx))
    if available_charge_hours <= 0:
        return 0.0, 0.0, "No off-peak hours available in configured charging windows.", soc_start

    soc_start = max(min(soc_start, 1.0), 0.0)
    soc_cutoff = max(min(soc_cutoff, 1.0), 0.0)

    if soc_cutoff <= soc_start + 1e-9:
        return 0.0, 0.0, "No AC charging needed (cutoff already reached).", soc_start

    soc_at_22_kwh = soc_start * BATTERY_KWH
    target_soc_kwh = soc_cutoff * BATTERY_KWH
    required_batt_kwh = max(0.0, target_soc_kwh - soc_at_22_kwh)
    required_grid_kwh = required_batt_kwh / BATTERY_AC_CHARGE_EFF
    recommended_allowed_ac_kw = required_grid_kwh / available_charge_hours
    user_cap = MAX_AC_CHARGE_KW_HARD_LIMIT if user_cap_kw is None else max(user_cap_kw, 0.0)
    effective_cap_kw = min(user_cap, INVERTER_AC_KW_LIMIT, BATTERY_MAX_CHARGE_KW)
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


# ============================================================
# DETAIL SIMULATIE: per uur binnen hoog-tarief uren
# ============================================================

def simulate_expensive_hours_detailed(
    df: "pd.DataFrame",
    total_consumption_kwh: float,
    start_soc: float,
    for_date: dt.date
) -> Tuple["pd.DataFrame", float, float, float, bool]:
    expensive_windows = get_expensive_windows(for_date)
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
        pv_unclipped = _get_float(
            df,
            ts,
            "pv_dc_available_kwh",
            default=_get_float(
                df,
                ts,
                "pv_total_unclipped_kwh",
                default=_get_float(df, ts, "pv_ac_limited_kwh", default=_get_float(df, ts, "pv_total_kwh", default=0.0)),
            ),
        )
        pv_ac_limited = _get_float(df, ts, "pv_ac_limited_kwh", default=_get_float(df, ts, "pv_total_kwh", default=0.0))
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
            grid_export = min(pv_after_batt, export_limit)
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

    detail_df["pv_total_unclipped_kwh"] = pd.to_numeric(detail_df.get("pv_total_unclipped_kwh", 0.0), errors="coerce").fillna(0.0)
    detail_df["pv_total_kwh"] = pd.to_numeric(detail_df.get("pv_total_kwh", 0.0), errors="coerce").fillna(0.0)
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
    tomorrow_date: dt.date,
) -> "pd.DataFrame":
    idx = get_charge_session_index(tomorrow_date - dt.timedelta(days=1))
    tomorrow_start = pd.Timestamp(dt.datetime.combine(tomorrow_date, dt.time(0, 0)), tz=TIMEZONE)
    required_points = pd.DatetimeIndex([tomorrow_start, tomorrow_start + dt.timedelta(hours=7)])
    idx = idx.union(required_points).sort_values()
    energy = max(0.0, min(1.0, soc_at_22)) * BATTERY_KWH
    max_energy = MAX_CUTOFF_SOC * BATTERY_KWH
    charge_cutoff_energy = min(MAX_CUTOFF_SOC, max(MIN_SOC, cutoff_soc)) * BATTERY_KWH

    rows = []
    for ts in idx:
        step_h = 1.0
        soc_start_pct = (energy / BATTERY_KWH * 100.0) if BATTERY_KWH > 0 else 0.0

        batt_charge_kwh = 0.0
        grid_import = 0.0

        if in_any_window(ts.time(), get_charge_windows(ts.date())):
            charge_grid_kwh = min(charge_kw * step_h, BATTERY_MAX_CHARGE_KW * step_h)
            room = max(0.0, min(max_energy, charge_cutoff_energy) - energy)
            charge_to_battery = min(room, charge_grid_kwh * BATTERY_AC_CHARGE_EFF)
            if charge_to_battery > 0:
                grid_import = charge_to_battery / BATTERY_AC_CHARGE_EFF if BATTERY_AC_CHARGE_EFF > 0 else 0.0
                energy = min(max_energy, energy + charge_to_battery)
                batt_charge_kwh = charge_to_battery

        soc_end_pct = (energy / BATTERY_KWH * 100.0) if BATTERY_KWH > 0 else soc_start_pct
        rows.append(
            {
                "time": ts,
                "load_kwh": 0.0,
                "pv_to_load_kwh": 0.0,
                "batt_charge_kwh": batt_charge_kwh,
                "batt_discharge_kwh": 0.0,
                "grid_import_kwh": grid_import,
                "grid_export_kwh": 0.0,
                "curtailed_kwh": 0.0,
                "soc_start_pct": soc_start_pct,
                "soc_end_pct": soc_end_pct,
            }
        )

    night_df = pd.DataFrame(rows).set_index("time")
    if ENABLE_INVARIANT_CHECKS:
        validate_flow_invariants(night_df, "night")
    return night_df


def simulate_full_day_soc(
    df: "pd.DataFrame",
    yesterday_total_kwh: float,
    soc_at_22: float,
    charge_kw: float,
    cutoff_soc: float,
    tomorrow_date: dt.date,
) -> Tuple["pd.Series", "pd.DataFrame"]:
    idx = df.index
    dt_h = timestep_hours(idx)
    loads = build_hourly_load_series(idx, yesterday_total_kwh)

    night_df = simulate_night_charging_series(soc_at_22, charge_kw, cutoff_soc, tomorrow_date)
    tomorrow_start = pd.Timestamp(dt.datetime.combine(tomorrow_date, dt.time(0, 0)), tz=TIMEZONE)
    soc_00 = float(night_df.loc[tomorrow_start, "soc_start_pct"]) / 100.0
    soc_07 = float(night_df.loc[tomorrow_start + dt.timedelta(hours=7), "soc_start_pct"]) / 100.0

    day_idx = idx[idx >= tomorrow_start + dt.timedelta(hours=7)]
    day_loads = loads.loc[day_idx]

    energy = max(MIN_SOC, min(MAX_CUTOFF_SOC, soc_07)) * BATTERY_KWH
    min_energy = MIN_SOC * BATTERY_KWH
    max_energy = MAX_CUTOFF_SOC * BATTERY_KWH

    day_rows = []
    day_soc_vals = []
    for ts in day_idx:
        step_h = float(dt_h.loc[ts])
        pv_ac_limited = float(df.loc[ts, "pv_total_kwh"]) if "pv_total_kwh" in df.columns else 0.0
        pv_unclipped = float(df.loc[ts, "pv_total_unclipped_kwh"]) if "pv_total_unclipped_kwh" in df.columns else pv_ac_limited
        load = float(day_loads.loc[ts])

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
            grid_export = min(pv_after_batt, export_limit)
            curtailed = max(0.0, pv_after_batt - grid_export)

        if remaining_load > 0:
            available = max(0.0, energy - min_energy)
            discharge_power_limited = BATTERY_MAX_DISCHARGE_KW * step_h
            needed_from_batt = remaining_load / BATTERY_DISCHARGE_EFF if BATTERY_DISCHARGE_EFF > 0 else remaining_load
            discharge = min(available, needed_from_batt, discharge_power_limited)
            energy -= discharge
            batt_discharge_kwh = discharge
            delivered = discharge * BATTERY_DISCHARGE_EFF
            grid_import = max(0.0, remaining_load - delivered)

        soc_end_pct = (energy / BATTERY_KWH * 100.0) if BATTERY_KWH > 0 else soc_start_pct
        day_soc_vals.append(soc_end_pct)
        day_rows.append({
            "time": ts,
            "load_kwh": load,
            "pv_to_load_kwh": pv_to_load,
            "batt_charge_kwh": batt_charge_kwh,
            "batt_discharge_kwh": batt_discharge_kwh,
            "grid_import_kwh": grid_import,
            "grid_export_kwh": grid_export,
            "curtailed_kwh": curtailed,
            "soc_start_pct": soc_start_pct,
            "soc_end_pct": soc_end_pct,
        })

    day_flows = pd.DataFrame(day_rows).set_index("time")
    day_soc = pd.Series(day_soc_vals, index=day_idx, name="soc_percent")

    night_tomorrow = night_df[(night_df.index >= tomorrow_start) & (night_df.index < tomorrow_start + dt.timedelta(hours=7))]
    night_soc = night_tomorrow["soc_end_pct"].rename("soc_percent")

    full_flows = pd.concat([night_tomorrow, day_flows]).sort_index()
    full_soc = pd.concat([night_soc, day_soc]).sort_index()

    if BATTERY_KWH > 0 and not full_soc.empty:
        full_soc.iloc[0] = soc_00 * 100.0

    if ENABLE_INVARIANT_CHECKS:
        validate_flow_invariants(full_flows, "full_day")

    return full_soc, full_flows


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
