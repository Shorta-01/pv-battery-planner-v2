import ast
import datetime as dt
from pathlib import Path

import pandas as pd


class _DummyCore:
    @staticmethod
    def get_offpeak_windows_for_date(_date, _cfg):
        return []

    @staticmethod
    def get_expensive_windows(_date, _cfg):
        return []


class _Container:
    def __init__(self):
        self.calls = []

    def markdown(self, html: str, unsafe_allow_html: bool = False):
        self.calls.append((html, unsafe_allow_html))


def _load_symbols():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8", errors="replace")
    module = ast.parse(source)
    wanted = {
        "_pv_outlook_badge_html",
        "_resolve_est_injection_kwh",
        "render_pv_quality_widget",
    }
    nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    isolated = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(isolated)

    ns = {
        "pd": pd,
        "dt": dt,
        "core": _DummyCore,
        "windows_to_segments": lambda _: [],
        "clamp_pct": lambda x: x,
        "make_summary_lines": lambda _a, _b: ("Off-peak", ""),
        "weather_code_to_icon": lambda _c: "☀️",
        "weather_code_to_label": lambda _c: "Clear",
        "_pv_quality_signal_html": lambda _l, _c, _t: "<span>Q</span>",
        "_esc_attr": lambda s: str(s),
        "_esc": lambda s: str(s),
        "_safe_float": lambda v, d=0.0: float(v) if v is not None else float(d),
        "UI_PROGRESS_BAR_HEIGHT_PX": 8,
        "APP_DEBUG": False,
    }
    exec(compile(isolated, filename="app.py", mode="exec"), ns)
    return ns, source


def _render_html(*, load_kwh: float | None, injection_kwh: float | None) -> str:
    ns, _ = _load_symbols()
    container = _Container()
    ns["render_pv_quality_widget"](
        container=container,
        pv_df=pd.DataFrame(),
        pv_quality_dict={"score": 75, "ratio": 0.6, "label": "Mixed", "color": "#94a3b8", "pv_total_kwh": 12.0},
        tomorrow_date=dt.date(2026, 1, 10),
        forecast_total_load_kwh=load_kwh,
        est_injection_kwh=injection_kwh,
    )
    assert container.calls
    return container.calls[0][0]


def test_badge_labels_are_explicit_english() -> None:
    html = _render_html(load_kwh=18.0, injection_kwh=10.6)
    assert "🏠" in html
    assert "📤" in html
    assert "Load 18.0 kWh" in html
    assert "Injection 10.6 kWh" in html


def test_old_ambiguous_strings_are_not_in_widget_markup() -> None:
    html = _render_html(load_kwh=18.0, injection_kwh=10.6)
    assert "Estimated export/curtailment" not in html
    assert "🧾" not in html


def test_injection_excludes_curtailment() -> None:
    ns, _ = _load_symbols()
    flows_df = pd.DataFrame({"grid_export_kwh": [7.0], "curtailed_kwh": [3.0]})
    injection = ns["_resolve_est_injection_kwh"](flows_df, metrics_grid_export=99.0)
    assert injection == 7.0

    html = _render_html(load_kwh=18.0, injection_kwh=injection)
    assert "Injection 7.0 kWh" in html
    assert "Injection 10.0 kWh" not in html


def test_injection_badge_hidden_when_zero() -> None:
    html = _render_html(load_kwh=18.0, injection_kwh=0.0)
    assert "Injection" not in html
    assert "Load 18.0 kWh" in html
