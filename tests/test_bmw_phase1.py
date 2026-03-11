import base64
import datetime as dt
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bmw_auth import BmwAuthClient
from bmw_mapping import apply_planner_derivations, freshness_bucket, map_bmw_payload_to_vehicle_states
from bmw_models import BmwTokenData, NormalizedVehicleState
from bmw_storage import BmwStorage


def test_freshness_bucket_ranges():
    now = dt.datetime.now(dt.timezone.utc)
    assert freshness_bucket(now - dt.timedelta(seconds=30), now=now)[0] == "fresh"
    assert freshness_bucket(now - dt.timedelta(seconds=300), now=now)[0] == "aging"
    assert freshness_bucket(now - dt.timedelta(seconds=1200), now=now)[0] == "stale"
    assert freshness_bucket(now - dt.timedelta(seconds=2000), now=now)[0] == "error"


def test_mapping_partial_payload_resilient():
    payload = {"vehicles": [{"vin": "VIN123", "soc": 55, "plug_status": "plugged", "timestamp": "2026-03-11T10:00:00Z"}]}
    states = map_bmw_payload_to_vehicle_states(payload)
    assert len(states) == 1
    st = states[0]
    assert st.vehicle_id == "VIN123"
    assert st.soc_pct == 55
    assert st.is_plugged is True
    assert st.range_km is None


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
        return _DummyResponse(payload={"device_code": "abc", "user_code": "uc", "verification_uri": "https://verify"})

    monkeypatch.setattr("bmw_auth.requests.post", fake_post)
    client = BmwAuthClient(client_id="cid", token_cache_path=str(tmp_path / "token.json"))

    client.start_device_flow()

    assert captured["url"] == "https://customer.bmwgroup.com/gcdm/oauth/device/code"
    assert captured["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    assert captured["json"] is None
    assert set(captured["data"].keys()) == {"client_id", "code_challenge", "code_challenge_method", "scope"}
    assert captured["data"]["client_id"] == "cid"
    assert captured["data"]["code_challenge_method"] == "S256"
    assert captured["data"]["scope"] == BmwAuthClient.DEVICE_FLOW_SCOPE

    session_data = json.loads((tmp_path / "token_device_flow_session.json").read_text(encoding="utf-8"))
    assert session_data["client_id"] == "cid"
    assert session_data["device_code"] == "abc"
    assert session_data["code_verifier"]


def test_poll_device_token_builds_form_encoded_payload(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, data=None, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["json"] = json
        return _DummyResponse(payload={"access_token": "a", "expires_in": 3600})

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
    assert set(captured["data"].keys()) == {"client_id", "code_verifier", "device_code", "grant_type", "scope"}
    assert captured["data"]["client_id"] == "cid"
    assert captured["data"]["code_verifier"] == "verifier-1"
    assert captured["data"]["device_code"] == "device-code-1"
    assert captured["data"]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
    assert captured["data"]["scope"] == BmwAuthClient.DEVICE_FLOW_SCOPE


def test_device_flow_scope_regression():
    assert BmwAuthClient.DEVICE_FLOW_SCOPE == "authenticate_user openid cardata:streaming:read cardata:api:read"
