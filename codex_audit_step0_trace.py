import os, re, sys, subprocess
from pathlib import Path
import ast
from typing import List, Tuple, Optional

REPO = Path('.').resolve()

TARGET_FILES = [
    'app.py',
    'backend_api.py',
    'planner_core.py',
    'weather_ensemble.py',
]

# --- helpers ---
def run(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        out = (p.stdout or '') + (p.stderr or '')
        return p.returncode, out.strip()
    except Exception as e:
        return 999, f'ERROR running {cmd}: {e}'


def print_hdr(title: str):
    print('\n' + '=' * 110)
    print(title)
    print('=' * 110)


def slurp(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return path.read_text(encoding='latin-1', errors='replace')
    except Exception:
        return None


def iter_py_files():
    # Limit to repo (skip venv, .git, node_modules, dist, build)
    skip = {'.git', '.venv', 'venv', 'node_modules', 'dist', 'build', '__pycache__'}
    for p in REPO.rglob('*.py'):
        parts = set(p.parts)
        if parts & skip:
            continue
        yield p


def find_matches(patterns: List[str], files: List[Path], context: int = 2) -> List[Tuple[str, int, str]]:
    out = []
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]
    for f in files:
        txt = slurp(f)
        if txt is None:
            continue
        lines = txt.splitlines()
        for i, line in enumerate(lines, start=1):
            if any(r.search(line) for r in regs):
                # add a small context block: current line only (keep output readable)
                out.append((str(f.relative_to(REPO)), i, line.rstrip()))
    return out


def print_matches(label: str, matches: List[Tuple[str, int, str]], limit: int = 200):
    print_hdr(label + f'  (matches: {len(matches)})')
    if not matches:
        print('NO MATCHES FOUND.')
        return
    for idx, (fp, ln, line) in enumerate(matches[:limit], start=1):
        print(f'{idx:03d}  {fp}:{ln}: {line}')
    if len(matches) > limit:
        print(f'... truncated ({len(matches) - limit} more)')


def load_ast(path: Path):
    txt = slurp(path)
    if txt is None:
        return None
    try:
        return ast.parse(txt)
    except SyntaxError as e:
        print(f'AST parse failed for {path}: {e}')
        return None


def ast_find_functions(tree: ast.AST) -> List[str]:
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            names.append(n.name)
    return sorted(set(names))


def ast_find_class_methods(tree: ast.AST, class_name: str) -> List[str]:
    methods = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            for b in n.body:
                if isinstance(b, ast.FunctionDef):
                    methods.append(b.name)
    return sorted(set(methods))


def ast_find_calls_in_func(tree: ast.AST, func_name: str) -> List[str]:
    calls = set()
    # find function defs at module level
    target_funcs = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == func_name:
            target_funcs.append(n)
    for fn in target_funcs:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    calls.add(f.id)
                elif isinstance(f, ast.Attribute):
                    calls.add(f.attr)
    return sorted(calls)


def ast_find_calls_in_method(tree: ast.AST, class_name: str, method_name: str) -> List[str]:
    calls = set()
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            for b in n.body:
                if isinstance(b, ast.FunctionDef) and b.name == method_name:
                    for node in ast.walk(b):
                        if isinstance(node, ast.Call):
                            f = node.func
                            if isinstance(f, ast.Name):
                                calls.add(f.id)
                            elif isinstance(f, ast.Attribute):
                                calls.add(f.attr)
    return sorted(calls)


def must_exist(file_name: str) -> Path:
    p = REPO / file_name
    if not p.exists():
        print(f'FATAL: expected file not found: {file_name} (run from repo root)')
        sys.exit(2)
    return p


# --- 0) repo state proof ---
print_hdr('0) REPO STATE (branch + commit + clean status)')
rc, branch = run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
rc2, commit = run(['git', 'rev-parse', 'HEAD'])
rc3, status = run(['git', 'status', '--porcelain'])
print(f'Repo:   {REPO}')
print(f'Branch: {branch if rc == 0 else "ERROR"}')
print(f'Commit: {commit if rc2 == 0 else "ERROR"}')
print('Clean:  ' + ('YES' if (rc3 == 0 and status.strip() == '') else 'NO'))
if rc3 == 0 and status.strip():
    print('\nDirty files:')
    print(status)

# --- file anchors ---
app_py = must_exist('app.py')
backend_py = must_exist('backend_api.py')
planner_py = must_exist('planner_core.py')
weather_py = must_exist('weather_ensemble.py')

# --- 1) Endpoints + run-now chain anchors ---
print_hdr('1) BACKEND ENTRYPOINT ANCHORS (run/now, API endpoints, PlannerService entry)')
anchors = find_matches(
    patterns=[
        r'/v1/run/now',
        r'run_now',
        r'@app\.post',
        r'FastAPI\(',
        r'PlannerService',
    ],
    files=[backend_py, app_py],
)
for fp, ln, line in anchors[:250]:
    print(f'{fp}:{ln}: {line}')

# --- 2) INPUT 1: SOC NOW ingress + planner usage ---
soc_patterns = [
    r'\bsoc_now\b',
    r'\bsoc_now_pct\b',
    r'battery.*soc.*now',
    r'\bsoc\b.*now',
    r'state_of_charge',
]
soc_matches = find_matches(soc_patterns, [app_py, backend_py, planner_py])
print_matches('2) INPUT 1 — SOC NOW (%) — ingress (app/backend) + usage (planner_core)', soc_matches)

# --- 3) INPUT 2: Consumption ingress + processing ---
cons_patterns = [
    r'\b(yesterday|verbruik|consumption|daily_kwh|kwh_daily|kwh_yesterday)\b',
    r'\b(load|usage)\b',
    r'\bprofile\b',
    r'hourly.*load',
]
cons_matches = find_matches(cons_patterns, [app_py, backend_py, planner_py])
print_matches('3) INPUT 2 — CONSUMPTION (kWh) — ingress + processing (daily vs hourly vs fallback)', cons_matches)

# --- 4) INPUT 3: PV forecast tomorrow — weather models/ensemble output fields passed into planner ---
pv_patterns = [
    r'\bpv\b.*forecast',
    r'PV Outlook',
    r'\bensemble\b',
    r'p10|p50|p90',
    r'pv_ensemble',
    r'weather_ensemble',
    r'ICON|ECMWF|AIFS|GFS|meteofrance|dwd',
    r'irradiance|radiation',
    r'daylight|sunrise|sunset|elevation',
    r'clipping|ac_limit|pvwatts|inverter',
]
pv_matches = find_matches(pv_patterns, [weather_py, planner_py, backend_py, app_py])
print_matches('4) INPUT 3 — PV FORECAST TOMORROW — models/ensemble + fields forwarded to planner', pv_matches)

# --- 5) OUTPUT 1: Savings € — baseline vs plan, cycle/tomorrow scope, terminal SOC value ---
save_patterns = [
    r'savings|besparing',
    r'baseline',
    r'plan_cost|baseline_cost',
    r'cycle|horizon|22:00|off-peak',
    r'terminal.*soc|soc.*value',
    r'eur|€',
]
save_matches = find_matches(save_patterns, [planner_py, backend_py, app_py])
print_matches('5) OUTPUT 1 — SAVINGS (€) — baseline vs plan + scope (cycle/tomorrow) + terminal SOC value', save_matches)

# --- 6) OUTPUT 2: Action plan — charge kW, cutoff SOC, warnings/limits ---
plan_patterns = [
    r'charge_kw|grid_charge|planned.*charge|charge power',
    r'cutoff|minimum morning|target',
    r'allow_injection|inject',
    r'max_grid_import|grid import cap|import cap',
    r'unreachable|limit|binding|warning|warnings',
    r'reserve|min_soc|backup',
]
plan_matches = find_matches(plan_patterns, [planner_py, backend_py, app_py])
print_matches('6) OUTPUT 2 — ACTION PLAN — charge kW + cutoff/target + warnings/limits end-to-end', plan_matches)

# --- 7) FUNCTION MAP (AST): backend_api → weather_ensemble → planner_core → savings → payload → app ---
print_hdr('7) FUNCTION MAP (AST) — direct calls from key entrypoints')
backend_tree = load_ast(backend_py)
planner_tree = load_ast(planner_py)
weather_tree = load_ast(weather_py)
app_tree = load_ast(app_py)


def show_ast_summary(name: str, tree: Optional[ast.AST]):
    if tree is None:
        print(f'{name}: AST unavailable')
        return
    fns = ast_find_functions(tree)
    print(f"{name}: functions={len(fns)} (showing up to 40): {', '.join(fns[:40])}{' ...' if len(fns) > 40 else ''}")


show_ast_summary('backend_api.py', backend_tree)
show_ast_summary('planner_core.py', planner_tree)
show_ast_summary('weather_ensemble.py', weather_tree)
show_ast_summary('app.py', app_tree)

# Try common entrypoints: PlannerService.run_now, run_now, /v1/run/now handler
if backend_tree is not None:
    # If PlannerService class exists, list methods and calls
    classes = [n.name for n in backend_tree.body if isinstance(n, ast.ClassDef)]
    print('\nbackend_api.py classes (top-level):', ', '.join(classes) if classes else '(none)')

    if 'PlannerService' in classes:
        methods = ast_find_class_methods(backend_tree, 'PlannerService')
        print('PlannerService methods:', ', '.join(methods) if methods else '(none)')
        if 'run_now' in methods:
            calls = ast_find_calls_in_method(backend_tree, 'PlannerService', 'run_now')
            print('\nDirect calls inside PlannerService.run_now():')
            for c in calls:
                print(' -', c)

    # Also check module-level run_now()
    module_calls = ast_find_calls_in_func(backend_tree, 'run_now')
    if module_calls:
        print('\nDirect calls inside module-level run_now():')
        for c in module_calls:
            print(' -', c)

print_hdr('END — Paste this full output back here. Then we start Stap 1 (max 8 regels end-to-end + mini rekenvoorbeeld).')
