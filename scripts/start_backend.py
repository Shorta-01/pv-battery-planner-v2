from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    host = os.getenv("PVBP_BACKEND_HOST", "127.0.0.1")
    port = os.getenv("PVBP_BACKEND_PORT", "8787")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend_api:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
