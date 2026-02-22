import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api


def test_resolve_soc_prefers_soc_now_payload():
    warnings: list[str] = []
    soc = backend_api._resolve_soc_percent(
        payload_soc_now=55.0,
        payload_soc_legacy=20.0,
        source={"soc_now_percent": 40.0, "soc_at_22_percent": 35.0},
        warnings=warnings,
    )
    assert soc == 55.0
    assert warnings == []


def test_resolve_soc_falls_back_to_last_inputs():
    warnings: list[str] = []
    soc = backend_api._resolve_soc_percent(
        payload_soc_now=None,
        payload_soc_legacy=None,
        source={"soc_now_percent": 47.5, "soc_at_22_percent": 30.0},
        warnings=warnings,
    )
    assert soc == 47.5
    assert warnings == []


def test_resolve_soc_clamps_and_warns():
    warnings: list[str] = []
    soc = backend_api._resolve_soc_percent(
        payload_soc_now=None,
        payload_soc_legacy=-12.0,
        source={},
        warnings=warnings,
    )
    assert soc == 0.0
    assert any("soc_at_22_percent: clamped to 0.0" in msg for msg in warnings)
