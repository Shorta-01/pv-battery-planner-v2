from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

API_BASE = os.getenv("PVBP_BACKEND_URL", "http://127.0.0.1:8787")
TOKEN = os.getenv("PVBP_API_TOKEN", "")
if not TOKEN:
    token_path = Path("local_state/api_token.txt")
    if token_path.exists():
        TOKEN = token_path.read_text(encoding="utf-8").strip()

if not TOKEN:
    print("Missing API token. Set PVBP_API_TOKEN or local_state/api_token.txt", file=sys.stderr)
    sys.exit(2)


def call_tick() -> requests.Response:
    return requests.post(
        f"{API_BASE}/v1/run/nightly",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"force": False},
        timeout=60,
    )


def try_start_backend() -> None:
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend_api:app", "--host", "127.0.0.1", "--port", "8787"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


try:
    response = call_tick()
except requests.RequestException:
    try_start_backend()
    time.sleep(5)
    try:
        response = call_tick()
    except requests.RequestException as exc:
        print(f"Nightly tick failed after backend start attempt: {exc}", file=sys.stderr)
        sys.exit(1)

if response.status_code >= 400:
    print(f"Nightly tick failed ({response.status_code}): {response.text}", file=sys.stderr)
    sys.exit(1)

print(response.text)
sys.exit(0)
