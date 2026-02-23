import planner_core as core
from config_accessors import get_inverter_ac_kw_limit


def test_inverter_limit_new_schema():
    cfg = {"inverter": {"ac_limit_kw": 7.2}}
    assert get_inverter_ac_kw_limit(cfg) == 7.2


def test_inverter_limit_legacy_schema():
    cfg = {"pv": {"inverter_ac_kw_limit": 5.5}}
    assert get_inverter_ac_kw_limit(cfg) == 5.5


def test_inverter_limit_missing_uses_default():
    cfg = {"pv": {}}
    expected = float(core.DEFAULT_CONFIG.get("pv", {}).get("inverter_ac_kw_limit", core.INVERTER_AC_KW_LIMIT))
    assert get_inverter_ac_kw_limit(cfg) == expected


def test_inverter_limit_unparseable_uses_default():
    cfg = {"inverter": {"ac_limit_kw": "nope"}, "pv": {"inverter_ac_kw_limit": ""}}
    expected = float(core.DEFAULT_CONFIG.get("pv", {}).get("inverter_ac_kw_limit", core.INVERTER_AC_KW_LIMIT))
    assert get_inverter_ac_kw_limit(cfg) == expected
