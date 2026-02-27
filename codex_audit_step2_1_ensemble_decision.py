import re
from pathlib import Path

REPO = Path(".").resolve()
BA = REPO/"backend_api.py"
PC = REPO/"planner_core.py"
WE = REPO/"weather_ensemble.py"

def lines(p): return p.read_text(encoding="utf-8", errors="replace").splitlines()

def show(p, ln, before=10, after=25):
    L = lines(p)
    s = max(1, ln-before); e = min(len(L), ln+after)
    for i in range(s, e+1):
        mark = ">>" if i==ln else "  "
        print(f"{mark} {p.name}:{i}: {L[i-1].rstrip()}")

def find(p, pat, limit=20):
    rx = re.compile(pat, re.IGNORECASE)
    out=[]
    for i, line in enumerate(lines(p), start=1):
        if rx.search(line):
            out.append((i, line.rstrip()))
            if len(out)>=limit: break
    return out

def hdr(t):
    print("\n" + "="*110)
    print(t)
    print("="*110)

# 1) backend_api: show _run() signature + body where weather args are used
hdr("1) backend_api.py — def _run(...) signature + usage of weather_models/forecast_mode/ensemble_method/pv_uncertainty/fast_mode/nowcast")
for ln,_ in find(BA, r"def _run\(", 5):
    show(BA, ln)

for pat in [
    r"weather_models|forecast_mode|ensemble_method|pv_uncertainty|fast_mode|use_satellite_nowcast_0_6h",
    r"weather_ensemble",
    r"EnsembleWeatherResult|pv_ensemble_",
    r"pv_total_decision_kwh",
    r"run_forecast_pipeline\(",
]:
    hits = find(BA, pat, 12)
    if hits:
        hdr(f"backend_api.py — hits for: {pat}")
        for ln,_ in hits:
            show(BA, ln, before=8, after=18)

# 2) planner_core: prove fetch_weather_for_date and what it calls
hdr("2) planner_core.py — fetch_weather_for_date + _fetch_weather_payload (do they call weather_ensemble?)")
for pat in [r"def fetch_weather_for_date\(", r"def _fetch_weather_payload\("]:
    for ln,_ in find(PC, pat, 2):
        show(PC, ln)

# 3) Search pv_total_decision_kwh creation anywhere
hdr("3) Search for pv_total_decision_kwh creation/assignment across repo")
for p in REPO.rglob("*.py"):
    if any(x in p.parts for x in [".git","venv",".venv","node_modules","__pycache__"]): 
        continue
    hits = find(p, r"pv_total_decision_kwh\s*=", 5)
    if hits:
        print(f"\nFOUND in {p.relative_to(REPO)}")
        for ln,_ in hits:
            show(p, ln, before=6, after=10)

hdr("END — paste output here.")
