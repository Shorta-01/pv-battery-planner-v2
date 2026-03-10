from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
import importlib.abc
import sys
from contextlib import contextmanager


class _BlockFrontendFinder(importlib.abc.MetaPathFinder):
    """Blocks frontend-only imports to simulate backend-only runtime."""

    BLOCKED_PREFIXES = ("streamlit", "plotly")

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(self.BLOCKED_PREFIXES):
            raise ImportError(f"blocked frontend dependency: {fullname}")
        return None


@contextmanager
def _blocked_frontend_imports():
    finder = _BlockFrontendFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


@contextmanager
def _fresh_module_state(*module_names: str):
    saved = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


def test_backend_modules_import_without_frontend_stack():
    with _blocked_frontend_imports(), _fresh_module_state(
        "planner_core", "weather_ensemble", "db_sqlite", "backend_api"
    ):
        importlib.import_module("planner_core")
        importlib.import_module("weather_ensemble")
        importlib.import_module("db_sqlite")
        importlib.import_module("backend_api")


def test_frontend_dependencies_import_in_full_environment():
    with _fresh_module_state("streamlit", "plotly", "plotly.graph_objects"):
        importlib.import_module("streamlit")
        importlib.import_module("plotly.graph_objects")
