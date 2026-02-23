from __future__ import annotations

from typing import Any

import planner_core as core


def get_inverter_ac_kw_limit(effective_cfg: Any) -> float:
    """
    Return inverter AC limit (kW) from effective_cfg, supporting both config schemas:
    - New: effective_cfg["inverter"]["ac_limit_kw"]
    - Legacy: effective_cfg["pv"]["inverter_ac_kw_limit"]
    Falls back to repo defaults if missing/unparseable.
    """
    default = float(core.DEFAULT_CONFIG.get("pv", {}).get("inverter_ac_kw_limit", core.INVERTER_AC_KW_LIMIT))

    if not isinstance(effective_cfg, dict):
        return default

    inv = effective_cfg.get("inverter")
    if isinstance(inv, dict):
        v = inv.get("ac_limit_kw")
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass

    pv = effective_cfg.get("pv")
    if isinstance(pv, dict):
        v = pv.get("inverter_ac_kw_limit")
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass

    return default
