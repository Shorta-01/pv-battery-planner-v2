import ast
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner_core as core


def test_apply_config_enforces_pv_modelling_defaults() -> None:
    cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    cfg["pv"]["inverter_eff"] = 0.50
    cfg["pv"]["pv_loss_model"] = "combined"
    cfg["pv"]["iam_model"] = "none"
    cfg["pv"]["iam_ashrae_b"] = 0.11
    cfg["pv"]["albedo"] = None
    cfg["pv"]["inverter_ac_model"] = "linear"

    core.apply_config(cfg)

    assert core.INVERTER_EFF == 0.97
    assert core.PV_LOSS_MODEL == "split"
    assert core.PV_IAM_MODEL == "ashrae"
    assert core.PV_IAM_ASHRAE_B == 0.05
    assert core.PV_ALBEDO == 0.20
    assert core.INVERTER_AC_MODEL == "pvwatts"


def test_build_settings_payload_uses_enforced_pv_defaults_constant() -> None:
    app_source = Path("app.py").read_text()
    tree = ast.parse(app_source)

    defaults_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ENFORCED_PV_DEFAULTS":
                    defaults_node = node.value
                    break

    assert isinstance(defaults_node, ast.Dict)
    defaults = {
        key.value: ast.literal_eval(value)
        for key, value in zip(defaults_node.keys, defaults_node.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    assert defaults["inverter_eff"] == 0.97
    assert defaults["pv_loss_model"] == "split"
    assert defaults["iam_model"] == "ashrae"
    assert defaults["iam_ashrae_b"] == 0.05
    assert defaults["albedo"] == 0.20
    assert defaults["inverter_ac_model"] == "pvwatts"
