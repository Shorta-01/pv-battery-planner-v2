from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    backend_api = importlib.import_module("backend_api")

    temp_root = Path(tempfile.mkdtemp(prefix="pvbp-smoke-"))
    try:
        local_state_dir = temp_root / "local_state"
        backend_api.LOCAL_STATE_DIR = local_state_dir
        backend_api.SETTINGS_PATH = local_state_dir / "settings.json"
        backend_api.INPUTS_PATH = local_state_dir / "last_inputs.json"
        backend_api.LATEST_RESULT_PATH = local_state_dir / "latest_result.json"
        backend_api.HISTORY_PATH = local_state_dir / "results_history.json"
        backend_api.SQLITE_PATH = local_state_dir / "planner_history.sqlite"
        backend_api.TOKEN_PATH = local_state_dir / "api_token.txt"
        backend_api.RUN_HISTORY_PATH = temp_root / "run_history_log.json"

        state = backend_api.BackendState()
        assert state.api_token
        assert backend_api.SQLITE_PATH.exists()
        assert backend_api.TOKEN_PATH.exists()

        streamlit_loaded = any(name == "streamlit" or name.startswith("streamlit.") for name in sys.modules)
        assert not streamlit_loaded, "backend startup should not require streamlit"

        print("PASS: import backend_api")
        print(f"PASS: sqlite initialized at {backend_api.SQLITE_PATH}")
        print(f"PASS: token created at {backend_api.TOKEN_PATH}")
        print("PASS: streamlit not imported during backend startup")
        return 0
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
