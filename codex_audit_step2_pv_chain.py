import re
from pathlib import Path

REPO = Path(".").resolve()
FILES = {
  "backend_api.py": REPO/"backend_api.py",
  "planner_core.py": REPO/"planner_core.py",
  "weather_ensemble.py": REPO/"weather_ensemble.py",
  "app.py": REPO/"app.py",
}

def slurp(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8", errors="replace").splitlines()

def show(p: Path, ln: int, before: int=10, after: int=18):
    lines = slurp(p)
    s = max(1, ln-before)
    e = min(len(lines), ln+after)
    for i in range(s, e+1):
        mark = ">>" if i==ln else "  "
        print(f"{mark} {p.name}:{i}: {lines[i-1].rstrip()}")

def find(p: Path, pat: str, limit: int=20):
    rx = re.compile(pat, re.IGNORECASE)
    out = []
    for i, line in enumerate(slurp(p), start=1):
        if rx.search(line):
            out.append((i, line.rstrip()))
            if len(out) >= limit:
                break
    return out

def hdr(t: str):
    print("\n" + "="*110)
    print(t)
    print("="*110)

# 1) Prove where PV forecast is built and what quantile is selected (planner_core)
pc = FILES["planner_core.py"]
hdr("1) planner_core.py — run_forecast_pipeline: where pv is created + which quantile columns are used")
for ln,_ in find(pc, r"def run_forecast_pipeline\(", 1):
    show(pc, ln)

# show any call that creates pv/weather inside pipeline
for pat in [
    r"\bpv\s*=",
    r"build_pv_forecast\(",
    r"weather_ensemble",
    r"pv_ensemble_(p10|p25|p50|p90)",
    r"decision.*p(10|25|50|90)",
]:
    hits = find(pc, pat, 8)
    if hits:
        hdr(f"planner_core.py — hits for pattern: {pat}")
        for ln,_ in hits:
            show(pc, ln, before=8, after=14)

# 2) Prove build_pv_forecast internals (units, index tz, daylight gating, clipping hooks)
hdr("2) planner_core.py — build_pv_forecast definition (units/index/gating/clipping)")
for ln,_ in find(pc, r"def build_pv_forecast\(", 1):
    show(pc, ln)
# key internal helpers often indicate kW→kWh integration + tz alignment
for pat in [
    r"_integrate_hourly_power_trapezoid",
    r"build_local_day_hour_index",
    r"align_timestamp_to_index_tz",
    r"apply_soft_daylight_gating|apply_daylight_clamp|twilight",
    r"pvwatts|ac_limit|inverter",
    r"pv_total_kwh|pv_kwh|pv_kwh_p50",
]:
    hits = find(pc, pat, 6)
    if hits:
        hdr(f"planner_core.py — hits for pattern: {pat}")
        for ln,_ in hits:
            show(pc, ln, before=8, after=14)

# 3) weather_ensemble: where p10/p50/p90 are produced + nowcast hook
we = FILES["weather_ensemble.py"]
hdr("3) weather_ensemble.py — where PV ensemble p10/p25/p50/p90 are defined/produced")
for pat in [
    r"pv_ensemble_p50",
    r"pv_ensemble_p10|pv_ensemble_p25|pv_ensemble_p90",
    r"def fetch_satellite_radiation_nowcast",
    r"def build_weather_ensemble_table",
    r"p10:|p50:|p90:",
]:
    hits = find(we, pat, 6)
    if hits:
        hdr(f"weather_ensemble.py — hits for pattern: {pat}")
        for ln,_ in hits:
            show(we, ln, before=8, after=14)

# 4) backend_api: prove how RunNowPayload PV settings flow into cfg used by planner
ba = FILES["backend_api.py"]
hdr("4) backend_api.py — show RunNowPayload PV fields + where they are applied to cfg/effective_cfg")
for ln,_ in find(ba, r"class RunNowPayload", 1):
    show(ba, ln)
for pat in [
    r"payload\.weather_models|payload\.forecast_mode|payload\.ensemble_method|payload\.pv_uncertainty|payload\.fast_mode|payload\.use_satellite_nowcast_0_6h",
    r"effective_cfg|build_effective_config|apply_config|applied_config",
    r"planner_core\.",
    r"run_forecast_pipeline\(",
]:
    hits = find(ba, pat, 10)
    if hits:
        hdr(f"backend_api.py — hits for pattern: {pat}")
        for ln,_ in hits:
            show(ba, ln, before=8, after=14)

hdr("END — paste this full output here (or upload the output txt).")
