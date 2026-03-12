import base64
import datetime as dt
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bmw_auth import BmwAuthClient
from bmw_cardata_provider import BmwCarDataProvider
from bmw_mapping import apply_planner_derivations, freshness_bucket, map_bmw_payload_to_vehicle_states
from bmw_models import BmwTokenData, NormalizedVehicleState
from bmw_storage import BmwStorage


def test_freshness_bucket_ranges():
    now = dt.datetime.now(dt.timezone.utc)
    assert freshness_bucket(now - dt.timedelta(seconds=30), now=now)[0] == "fresh"
    assert freshness_bucket(now - dt.timedelta(seconds=300), now=now)[0] == "aging"
    assert freshness_bucket(now - dt.timedelta(seconds=1200), now=now)[0] == "stale"
    assert freshness_bucket(now - dt.timedelta(seconds=2000), now=now)[0] == "error"


def test_mapping_bmw_fixture_realistic_shape():
    fixture_path = pathlib.Path(__file__).parent / "fixtures" / "bmw_cardata_phase1_realistic.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    states = map_bmw_payload_to_vehicle_states(payload)
    assert len(states) == 1
    st = states[0]
    assert st.vehicle_id == "WBY12345678900001"
    assert st.display_name == "i4 eDrive40"
    assert st.soc_pct == 62.5
    assert st.is_plugged is True
    assert st.is_charging is True
    assert st.range_km == 310
    assert st.time_to_full_min == 115
    assert st.charge_power_kw == 9.8
    assert st.ac_current_limit_a == 16
    assert st.charging_mode == "IMMEDIATE_CHARGING"
    assert st.optimized_charging_preference == "OFF"
    assert st.charge_window_start == "22:00"
    assert st.charge_window_end == "06:00"
    assert st.odometer_km == 12345.6
    assert st.travelled_distance_km == 44.1
    assert st.plug_status_raw == "CONNECTED"
    assert st.flap_lock_status_raw == "UNLOCKED"
    assert st.charge_error_raw is None


def test_derivations_energy_and_economics():
    st = NormalizedVehicleState(vehicle_id="V1", soc_pct=40, battery_capacity_kwh=80, is_plugged=True, data_status="fresh", range_km=250)
    out = apply_planner_derivations(
        st,
        petrol_price_eur_per_l=1.8,
        petrol_consumption_l_per_100km=6.0,
        charger_max_power_kw=11.0,
    )
    assert round(out.energy_needed_kwh, 2) == 48.0
    assert out.planner_demand_active is True
    assert out.planned_charge_cost_eur is not None
    assert out.avoided_petrol_cost_eur is not None
    assert out.net_economic_benefit_eur is not None


def test_token_freshness_eval():
    tok = BmwTokenData(expires_at=dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5))
    assert tok.is_fresh()


def test_multi_vehicle_storage_roundtrip(tmp_path):
    st = BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json"))
    st.save_vehicle_states({
        "a": NormalizedVehicleState(vehicle_id="a", soc_pct=10),
        "b": NormalizedVehicleState(vehicle_id="b", soc_pct=20),
    })
    loaded = st.load_vehicle_states()
    assert set(loaded.keys()) == {"a", "b"}


class _DummyResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_pkce_challenge_generation_matches_s256():
    verifier = "abc123verifier"
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest()).decode("ascii").rstrip("=")
    assert BmwAuthClient._pkce_code_challenge(verifier) == expected


def test_device_flow_start_builds_form_encoded_payload_and_session(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, data=None, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["json"] = json
        return _DummyResponse(
            payload={
                "device_code": "abc",
                "user_code": "uc",
                "verification_uri": "https://verify",
                "interval": 5,
                "expires_in": 1800,
            }
        )

    monkeypatch.setattr("bmw_auth.requests.post", fake_post)
    client = BmwAuthClient(client_id="cid", token_cache_path=str(tmp_path / "token.json"))

    client.start_device_flow()

    assert captured["url"] == "https://customer.bmwgroup.com/gcdm/oauth/device/code"
    assert captured["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    assert captured["json"] is None
    assert set(captured["data"].keys()) == {"client_id", "code_challenge", "code_challenge_method", "response_type", "scope"}
    assert captured["data"]["client_id"] == "cid"
    assert captured["data"]["code_challenge_method"] == "S256"
    assert captured["data"]["response_type"] == "device_code"
    assert captured["data"]["scope"] == BmwAuthClient.DEVICE_FLOW_SCOPE

    session_data = json.loads((tmp_path / "token_device_flow_session.json").read_text(encoding="utf-8"))
    assert session_data["client_id"] == "cid"
    assert session_data["device_code"] == "abc"
    assert session_data["user_code"] == "uc"
    assert session_data["verification_uri"] == "https://verify"
    assert session_data["interval"] == 5
    assert session_data["expires_in"] == 1800
    assert session_data["code_verifier"]


def test_poll_device_token_builds_form_encoded_payload(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, data=None, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["json"] = json
        return _DummyResponse(payload={"access_token": "a", "id_token": "i", "expires_in": 3600})

    session_path = tmp_path / "token_device_flow_session.json"
    session_path.write_text(
        json.dumps(
            {
                "client_id": "cid",
                "code_verifier": "verifier-1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "device_code": "device-code-1",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("bmw_auth.requests.post", fake_post)
    client = BmwAuthClient(client_id="cid", token_cache_path=str(tmp_path / "token.json"))

    client.poll_device_token("device-code-1")

    assert captured["url"].endswith("/gcdm/oauth/token")
    assert captured["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    assert captured["json"] is None
    assert set(captured["data"].keys()) == {"client_id", "code_verifier", "device_code", "grant_type"}
    assert captured["data"]["client_id"] == "cid"
    assert captured["data"]["code_verifier"] == "verifier-1"
    assert captured["data"]["device_code"] == "device-code-1"
    assert captured["data"]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"


def test_provider_uses_access_token_and_cardata_base_url(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", id_token="id-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    captured = []

    def fake_get(url, headers=None, timeout=None):
        captured.append({"url": url, "headers": headers})
        if url.endswith("/v1/vehicle-mappings"):
            return _DummyResponse(payload={"vehicleMappings": [{"vin": "VIN1", "displayName": "BMW"}]})
        if url.endswith("/v1/vehicles"):
            return _DummyResponse(payload={"vehicles": [{"vin": "VIN1", "lastUpdatedAt": "2026-03-11T10:00:00Z", "battery": {"socPercent": 80}}]})
        return _DummyResponse(status_code=404, text="not found")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    provider = BmwCarDataProvider(
        config={"bmw_enabled": True},
        storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")),
        auth=_Auth(),
    )

    out = provider.refresh_once()
    assert out["ok"] is True
    assert provider.rest_base_url() == "https://api-cardata.bmwgroup.com"
    assert captured[0]["headers"]["Authorization"] == "Bearer access-1"
    assert all(item["url"].startswith("https://api-cardata.bmwgroup.com") for item in captured)


def test_device_flow_scope_regression():
    assert BmwAuthClient.DEVICE_FLOW_SCOPE == "authenticate_user openid cardata:streaming:read cardata:api:read"


def test_no_stale_protocol_strings_in_bmw_modules():
    files = [
        pathlib.Path("bmw_auth.py"),
        pathlib.Path("bmw_cardata_provider.py"),
    ]
    for fp in files:
        text = fp.read_text(encoding="utf-8")
        assert "device_authorization" not in text
        assert "customer.bmwgroup.com/gcdm/cardata" not in text
