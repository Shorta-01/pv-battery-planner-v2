import ast
from pathlib import Path


def _load_weather_code_to_icon():
    source = Path("app.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="app.py")
    target = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "weather_code_to_icon"
    )
    target.decorator_list = []
    isolated_module = ast.Module(body=[target], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(isolated_module, filename="app.py", mode="exec"), namespace)
    return namespace["weather_code_to_icon"]


def test_weather_code_to_icon_common_codes_are_non_empty() -> None:
    weather_code_to_icon = _load_weather_code_to_icon()
    expected_non_empty = [0, 1, 2, 3, 45, 51, 61, 71, 80, 95]

    for code in expected_non_empty:
        icon = weather_code_to_icon(code)
        assert isinstance(icon, str)
        assert icon.strip()
        assert icon != "❓"



def test_weather_code_to_icon_none_and_unknown() -> None:
    weather_code_to_icon = _load_weather_code_to_icon()

    assert weather_code_to_icon(None) == "🌥️"
    assert weather_code_to_icon(12345) == "🌥️"
    assert weather_code_to_icon("0") == "☀️"
    assert weather_code_to_icon(1.0) == "🌤️"


def test_weather_code_to_icon_symbolic_strings() -> None:
    weather_code_to_icon = _load_weather_code_to_icon()

    assert weather_code_to_icon("rain") == "🌧️"
    assert weather_code_to_icon("partly_cloudy") == "⛅"
    assert weather_code_to_icon("unknown") == "🌥️"
