import ast
from pathlib import Path


def test_settings_payload_marks_location_metadata_auto_resolved():
    src = Path("app.py").read_text(encoding="utf-8")
    assert '"auto_resolve_metadata": True' in src


def test_location_validation_no_longer_requires_manual_timezone_entry():
    src = Path("app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Regression guard: settings validation should not show a manual-timezone error.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "Timezone must be a valid IANA name" not in node.value
