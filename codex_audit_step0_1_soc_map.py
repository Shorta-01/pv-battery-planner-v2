import ast, re
from pathlib import Path

REPO = Path('.').resolve()
TARGETS = ['app.py','backend_api.py','planner_core.py','weather_ensemble.py']
FILES = [REPO/p for p in TARGETS if (REPO/p).exists()]

def slurp(p: Path) -> str:
    return p.read_text(encoding='utf-8', errors='replace')

def print_hdr(t: str):
    print('\n' + '='*110)
    print(t)
    print('='*110)

def show_context(p: Path, lineno: int, before: int=6, after: int=6):
    lines = slurp(p).splitlines()
    s = max(1, lineno-before)
    e = min(len(lines), lineno+after)
    for ln in range(s, e+1):
        mark = '>>' if ln==lineno else '  '
        print(f"{mark} {p.name}:{ln}: {lines[ln-1].rstrip()}")

def find_lines(p: Path, pattern: str, limit: int=30):
    rx = re.compile(pattern, re.IGNORECASE)
    hits = []
    for i, line in enumerate(slurp(p).splitlines(), start=1):
        if rx.search(line):
            hits.append((i,line.rstrip()))
            if len(hits)>=limit:
                break
    return hits

# 1) Locate /v1/run/now payload build in app.py and dump nearby dict keys
app = REPO/'app.py'
print_hdr('A) app.py — where /v1/run/now payload is built (show context)')
for ln, line in find_lines(app, r'"/v1/run/now"|/v1/run/now', limit=10):
    show_context(app, ln)

# 2) RunNowPayload fields (exact) + line numbers
backend = REPO/'backend_api.py'
print_hdr('B) backend_api.py — RunNowPayload definition (fields + line numbers)')
tree = ast.parse(slurp(backend))
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name=='RunNowPayload':
        print(f'Found RunNowPayload at line {node.lineno}')
        for b in node.body:
            if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                field = b.target.id
                ln = b.lineno
                print(f'- {field}  (line {ln})')
        show_context(backend, node.lineno)
        break
else:
    print('RunNowPayload NOT FOUND')

# 3) /v1/run/now handler → state.run_now(payload) chain
print_hdr('C) backend_api.py — /v1/run/now handler and its immediate calls (context)')
for ln, line in find_lines(backend, r'@app\.post\("/v1/run/now"\)|def run_now\(', limit=10):
    show_context(backend, ln)

# 4) BackendState.run_now + PlannerService.run_now: show where SOC + consumption + PV are resolved
print_hdr('D) backend_api.py — BackendState.run_now / PlannerService.run_now / _resolve_soc_percent')
for pat in [r'class BackendState', r'def run_now\(self, payload: RunNowPayload\)', r'def run_now\(payload: RunNowPayload', r'def _run\(', r'def _resolve_soc_percent\(']:
    for ln, line in find_lines(backend, pat, limit=5):
        show_context(backend, ln)

# 5) planner_core: find any function params/usage that look like SOC input from backend
planner = REPO/'planner_core.py'
print_hdr('E) planner_core.py — find SOC input usage (context around key functions)')
for pat in [
    r'Estimate SOC at off-peak start',
    r'def .*soc',
    r'soc_now',
    r'soc_at_22',
    r'soc_start',
]:
    hits = find_lines(planner, pat, limit=8)
    for ln, line in hits:
        show_context(planner, ln)

print_hdr('END — Paste this output back here. Then Step 0 is fully proven and we can start Step 1.')
