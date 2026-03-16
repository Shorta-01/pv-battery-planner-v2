import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend_api


class _NoDiskPath:
    def exists(self):
        raise AssertionError("disk fallback should not be touched when in-memory settings are valid")


def test_car_charger_runtime_settings_prefers_in_memory(monkeypatch):
    monkeypatch.setattr(
        backend_api,
        "state",
        SimpleNamespace(
            settings={
                "config": {
                    "car_charger": {
                        "enabled": True,
                        "basic_user": "alice",
                        "basic_pass": "secret",
                    }
                }
            }
        ),
    )
    monkeypatch.setattr(backend_api, "SETTINGS_PATH", _NoDiskPath())

    enabled, user, pw = backend_api._car_charger_runtime_settings(
        warning_label="evse_status_settings_read_failed"
    )

    assert enabled is True
    assert user == "alice"
    assert pw == "secret"


def test_car_charger_runtime_settings_falls_back_to_disk(monkeypatch, tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"config": {"car_charger": {"enabled": True, "basic_user": "bob", "basic_pass": "pw"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(backend_api, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(backend_api, "state", SimpleNamespace(settings={"config": []}))

    enabled, user, pw = backend_api._car_charger_runtime_settings(warning_label="ocpp_ws_settings_read_failed")

    assert enabled is True
    assert user == "bob"
    assert pw == "pw"


def test_car_charger_runtime_settings_defaults_when_unavailable(monkeypatch, tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(backend_api, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(backend_api, "state", SimpleNamespace(settings={"config": []}))

    enabled, user, pw = backend_api._car_charger_runtime_settings(warning_label="ocpp_ws_settings_read_failed")

    assert enabled is False
    assert user == ""
    assert pw == ""
