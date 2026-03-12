import base64
import datetime as dt
import hashlib
import json
import pathlib
from pathlib import Path
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bmw_auth import BmwAuthClient
from bmw_cardata_provider import BmwCarDataProvider
from bmw_mapping import apply_planner_derivations, freshness_bucket, map_bmw_payload_to_vehicle_states
from bmw_models import BmwTokenData, NormalizedVehicleState
from bmw_service import BmwService
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
    assert session_data["expires_at"]
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
    assert "scope" not in captured["data"]


def test_provider_uses_access_token_and_cardata_base_url(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", id_token="id-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    captured = []

    def fake_get(url, headers=None, timeout=None):
        captured.append({"url": url, "headers": headers})
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(payload={"vehicleMappings": [{"vehicleId": "VIN1", "vin": "VIN1", "displayName": "BMW"}]})
        if url.endswith("/customers/vehicles/VIN1/basicData"):
            return _DummyResponse(payload={"vehicleId": "VIN1", "vin": "VIN1", "lastUpdatedAt": "2026-03-11T10:00:00Z", "battery": {"socPercent": 80}})
        if url.endswith("/customers/containers"):
            return _DummyResponse(payload={"containers": [{"containerId": "CONT1", "state": "ACTIVE", "name": "prod"}]})
        if url.endswith("/customers/vehicles/VIN1/telematicData?containerId=CONT1"):
            return _DummyResponse(payload={"charging": {"plugConnectionState": "CONNECTED", "chargingState": "CHARGING"}})
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
        assert "/v1/vehicle-mappings" not in text
        assert "\"/v1/vehicles\"" not in text


def test_start_device_flow_fails_fast_without_client_id(tmp_path):
    client = BmwAuthClient(client_id="", token_cache_path=str(tmp_path / "token.json"))
    try:
        client.start_device_flow()
        assert False, "Expected runtime error for missing client_id"
    except RuntimeError as exc:
        assert "BMW client ID not configured" in str(exc)


def test_update_config_rebuilds_runtime_and_updates_client_id(tmp_path):
    service = BmwService({
        "bmw_enabled": True,
        "bmw_client_id": "old-client-id-1234",
        "bmw_token_cache_path": str(tmp_path / "a" / "token.json"),
        "bmw_raw_event_store_path": str(tmp_path / "a" / "raw.jsonl"),
        "bmw_vehicle_state_store_path": str(tmp_path / "a" / "state.json"),
    })
    first_auth = service.auth
    first_provider = service.provider

    service.update_config(
        {
            "bmw_enabled": True,
            "bmw_client_id": "new-client-id-9999",
            "bmw_token_cache_path": str(tmp_path / "b" / "token.json"),
            "bmw_raw_event_store_path": str(tmp_path / "b" / "raw.jsonl"),
            "bmw_vehicle_state_store_path": str(tmp_path / "b" / "state.json"),
            "bmw_auth_base_url": "https://customer.bmwgroup.com/gcdm/oauth",
            "bmw_api_base_url": "https://api-cardata.bmwgroup.com",
            "bmw_stream_enabled": False,
        }
    )

    assert service.auth is not first_auth
    assert service.provider is not first_provider
    assert service.auth.client_id == "new-client-id-9999"
    debug = service.device_flow_debug_info()
    assert debug["provider_rebuilt_after_config_update"] is True
    assert debug["active_client_id_masked"] == "new-cl...9999"
    assert debug["device_flow_start_url"] == "https://customer.bmwgroup.com/gcdm/oauth/device/code"
    assert debug["device_flow_poll_url"] == "https://customer.bmwgroup.com/gcdm/oauth/token"
    assert debug["rest_api_base_url"] == "https://api-cardata.bmwgroup.com"
    assert debug["rest_token_mode"] == "access_token"
    assert debug["request_versioning_mode"] == "header:X-Version=v1"
    assert isinstance(debug["refresh_sequence_endpoints"], list)
    assert "vehicle_data_mode" in debug
    assert "has_live_telematics" in debug
    assert "last_rest_endpoint_attempted" in debug
    assert "last_rest_status_code" in debug
    assert "last_rest_safe_error_excerpt" in debug
    assert "capture_files_written" in debug
    assert "mapping_diagnostics" in debug
    assert "discovered_container_ids" in debug
    assert "active_container_id" in debug
    assert "container_diagnostics" in debug
    assert "last_telematic_url" in debug
    assert "last_telematic_status_code" in debug


def test_mapping_supports_live_capture_wrapper_shape():
    payload = {
        "endpoint": "/customers/vehicles/VINWRAP1/basicData",
        "payload": {
            "vehicles": [
                {
                    "vin": "VINWRAP1",
                    "lastUpdatedAt": "2026-03-11T10:00:00Z",
                    "battery": {"socPercent": 77},
                }
            ]
        },
    }
    states = map_bmw_payload_to_vehicle_states(payload)
    assert len(states) == 1
    assert states[0].vehicle_id == "VINWRAP1"
    assert states[0].soc_pct == 77


def test_provider_refresh_captures_live_payloads(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(payload={"vehicleMappings": [{"vehicleId": "VINCAP1", "vin": "VINCAP1", "displayName": "BMW"}]})
        if url.endswith("/customers/vehicles/VINCAP1/basicData"):
            return _DummyResponse(payload={"vehicleId": "VINCAP1", "vin": "VINCAP1", "lastUpdatedAt": "2026-03-11T10:00:00Z", "battery": {"socPercent": 55}})
        if url.endswith("/customers/containers"):
            return _DummyResponse(payload={"containers": [{"containerId": "CONTCAP1", "state": "ACTIVE"}]})
        if url.endswith("/customers/vehicles/VINCAP1/telematicData?containerId=CONTCAP1"):
            return _DummyResponse(payload={"charging": {"plugConnectionState": "CONNECTED", "chargingState": "NOT_CHARGING"}})
        return _DummyResponse(status_code=404, text="not found")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    storage = BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json"))
    provider = BmwCarDataProvider(config={"bmw_enabled": True}, storage=storage, auth=_Auth())

    out = provider.refresh_once()
    assert out["ok"] is True
    assert len(out["capture_files"]) >= 2
    assert any("telematicData" in str(p) for p in out["capture_files"])
    assert all(Path(p).exists() for p in out["capture_files"])


def test_mapping_live_fixture_file_support():
    fixture_path = pathlib.Path(__file__).parent / "fixtures" / "bmw_cardata_live_v1_vehicles_sample_20260311.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    states = map_bmw_payload_to_vehicle_states(payload)
    assert len(states) == 1
    assert states[0].vehicle_id == "WBY98765432100002"
    assert states[0].soc_pct == 71


def test_provider_refresh_sequence_and_version_headers(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append((url, dict(headers or {})))
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(payload={"vehicleMappings": [{"vehicleId": "VINSEQ1", "vin": "VINSEQ1"}]})
        if url.endswith("/customers/vehicles/VINSEQ1/basicData"):
            return _DummyResponse(payload={"vehicleId": "VINSEQ1", "vin": "VINSEQ1", "lastUpdatedAt": "2026-03-11T10:00:00Z", "battery": {"socPercent": 66}})
        if url.endswith("/customers/containers"):
            return _DummyResponse(payload={"containers": [{"containerId": "CONTSEQ1", "state": "ACTIVE"}]})
        if url.endswith("/customers/vehicles/VINSEQ1/telematicData?containerId=CONTSEQ1"):
            return _DummyResponse(payload={"charging": {"plugConnectionState": "CONNECTED", "chargingState": "CHARGING"}})
        return _DummyResponse(status_code=404, text="not found")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    provider = BmwCarDataProvider(
        config={"bmw_enabled": True},
        storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")),
        auth=_Auth(),
    )

    out = provider.refresh_once()
    assert out["ok"] is True
    assert [u for u, _ in seen][:4] == [
        "https://api-cardata.bmwgroup.com/customers/vehicles/mappings",
        "https://api-cardata.bmwgroup.com/customers/vehicles/VINSEQ1/basicData",
        "https://api-cardata.bmwgroup.com/customers/containers",
        "https://api-cardata.bmwgroup.com/customers/vehicles/VINSEQ1/telematicData?containerId=CONTSEQ1",
    ]
    assert all(h["Authorization"] == "Bearer access-1" for _, h in seen)
    assert all(h["X-Version"] == "v1" for _, h in seen)
    assert all(h["Accept"] == "application/json" for _, h in seen)
    assert out["request_versioning_mode"] == "header:X-Version=v1"
    assert out["rest_token_mode"] == "access_token"


def test_refresh_supports_top_level_array_mappings_payload(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    seen_urls = []

    def fake_get(url, headers=None, timeout=None):
        seen_urls.append(url)
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(
                payload=[
                    {
                        "mappedSince": "2024-03-05T09:12:47.671Z",
                        "mappingType": "PRIMARY",
                        "vin": "WBA51EH0X0CR89778",
                    }
                ]
            )
        if url.endswith("/customers/vehicles/WBA51EH0X0CR89778/basicData"):
            return _DummyResponse(
                payload={
                    "vin": "WBA51EH0X0CR89778",
                    "lastUpdatedAt": "2026-03-11T10:00:00Z",
                    "battery": {"socPercent": 64},
                }
            )
        if url.endswith("/customers/containers"):
            return _DummyResponse(payload={"containers": []})
        return _DummyResponse(status_code=500, text="unexpected")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    provider = BmwCarDataProvider(
        config={"bmw_enabled": True},
        storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")),
        auth=_Auth(),
    )

    out = provider.refresh_once()
    assert out["ok"] is True
    assert out["discovered_vehicle_ids"] == ["WBA51EH0X0CR89778"]
    assert out["active_vehicle_id"] == "WBA51EH0X0CR89778"
    assert seen_urls[:3] == [
        "https://api-cardata.bmwgroup.com/customers/vehicles/mappings",
        "https://api-cardata.bmwgroup.com/customers/vehicles/WBA51EH0X0CR89778/basicData",
        "https://api-cardata.bmwgroup.com/customers/containers",
    ]


def test_refresh_graceful_no_discovered_vehicles(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(payload={"vehicleMappings": []})
        return _DummyResponse(status_code=500, text="should not be called")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    provider = BmwCarDataProvider(
        config={"bmw_enabled": True},
        storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")),
        auth=_Auth(),
    )

    out = provider.refresh_once()
    assert out["ok"] is False
    assert out["reason"] == "no_vehicles"
    assert "no accessible BMW vehicle mappings" in out["message"]


def test_refresh_graceful_vehicle_endpoint_403_and_404(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(payload={"vehicleMappings": [{"vehicleId": "VINERR1", "vin": "VINERR1"}]})
        if url.endswith("/customers/vehicles/VINERR1/basicData"):
            return _DummyResponse(status_code=403, text="forbidden by API")
        if url.endswith("/customers/vehicles/VINERR1/telematicData"):
            return _DummyResponse(status_code=404, text="not found")
        return _DummyResponse(status_code=500, text="unexpected")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    provider = BmwCarDataProvider(
        config={"bmw_enabled": True},
        storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")),
        auth=_Auth(),
    )

    out = provider.refresh_once()
    assert out["ok"] is False
    assert out.get("reason") in {None, "poll_failed"}
    assert provider.status.last_rest_status_code == 403
    assert provider.status.last_rest_error_excerpt in {None, "forbidden by API"}


def test_storage_capture_naming_and_listing(tmp_path):
    storage = BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json"))
    capture_path = storage.store_raw_capture("/customers/vehicles/mappings", {"vehicleMappings": []}, status_code=200)
    assert capture_path.name.startswith("bmw_capture_customers_vehicles_mappings_")
    listed = storage.list_raw_captures(limit=5)
    assert str(capture_path) in listed


def test_primary_mapping_is_selected_when_available(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(payload={"vehicleMappings": [{"vin": "VINSECOND", "mappingType": "SECONDARY"}, {"vin": "VINPRIME", "mappingType": "PRIMARY"}]})
        if url.endswith("/customers/vehicles/VINPRIME/basicData"):
            return _DummyResponse(payload={"vin": "VINPRIME", "lastUpdatedAt": "2026-03-11T10:00:00Z", "battery": {"socPercent": 50}})
        if url.endswith("/customers/vehicles/VINPRIME/telematicData"):
            return _DummyResponse(status_code=404, text="not found")
        return _DummyResponse(status_code=500, text="unexpected")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    provider = BmwCarDataProvider(
        config={"bmw_enabled": True},
        storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")),
        auth=_Auth(),
    )
    out = provider.refresh_once()
    assert out["ok"] is True
    assert out["active_vehicle_id"] == "VINPRIME"
    assert any((row.get("mapping_role") == "PRIMARY") for row in out["mapping_diagnostics"])


def test_refresh_partial_data_basic_data_success_live_data_forbidden(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    now_iso = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(payload={"vehicleMappings": [{"vin": "VINPART1"}]})
        if url.endswith("/customers/vehicles/VINPART1/basicData"):
            return _DummyResponse(payload={"vin": "VINPART1", "lastUpdatedAt": now_iso, "battery": {"socPercent": 70}})
        if url.endswith("/customers/containers"):
            return _DummyResponse(payload={"containers": [{"containerId": "CONTPART1", "state": "ACTIVE"}]})
        if url.endswith("/customers/vehicles/VINPART1/telematicData?containerId=CONTPART1"):
            return _DummyResponse(status_code=403, text="forbidden")
        return _DummyResponse(status_code=500, text="unexpected")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    provider = BmwCarDataProvider(
        config={"bmw_enabled": True},
        storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")),
        auth=_Auth(),
    )
    out = provider.refresh_once()
    assert out["ok"] is True
    assert provider.status.last_rest_status_code == 403
    assert provider.status.vehicle_data_mode == "static_only"
    assert provider.vehicles["VINPART1"].soc_pct == 70
    assert provider.vehicles["VINPART1"].data_status != "error"


def test_no_legacy_v1_vehicle_mappings_endpoint_reference():
    text = pathlib.Path("bmw_cardata_provider.py").read_text(encoding="utf-8")
    assert "/v1/vehicles/mappings" not in text



def test_select_active_container_prefers_active_newest(tmp_path):
    provider = BmwCarDataProvider(config={"bmw_enabled": True}, storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")), auth=None)
    diags = [
        {"container_id": "A", "state": "ACTIVE", "created_at": "2026-03-11T08:00:00Z"},
        {"container_id": "B", "state": "INACTIVE", "created_at": "2026-03-11T09:00:00Z"},
        {"container_id": "C", "state": "ACTIVE", "created_at": "2026-03-11T10:00:00Z"},
    ]
    assert provider._select_active_container(diags) == "C"


def test_refresh_graceful_when_no_containers(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    seen_urls = []

    def fake_get(url, headers=None, timeout=None):
        seen_urls.append(url)
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(payload={"vehicleMappings": [{"vin": "VINNC1"}]})
        if url.endswith("/customers/vehicles/VINNC1/basicData"):
            return _DummyResponse(payload={"vin": "VINNC1", "lastUpdatedAt": "2026-03-11T10:00:00Z", "battery": {"socPercent": 81}})
        if url.endswith("/customers/containers"):
            return _DummyResponse(payload={"containers": []})
        return _DummyResponse(status_code=500, text="unexpected")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    provider = BmwCarDataProvider(config={"bmw_enabled": True}, storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")), auth=_Auth())

    out = provider.refresh_once()
    assert out["ok"] is True
    assert provider.status.active_container_id is None
    assert provider.status.vehicle_data_mode == "static_only"
    assert provider.vehicles["VINNC1"].soc_pct == 81
    assert all("telematicData?containerId=" not in u for u in seen_urls)


def test_capture_files_include_containers_and_telematics(monkeypatch, tmp_path):
    class _Auth:
        def load_token(self):
            return BmwTokenData(access_token="access-1", obtained_at=dt.datetime.now(dt.timezone.utc))

        def refresh_if_possible(self, tok):
            return tok

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/customers/vehicles/mappings"):
            return _DummyResponse(payload={"vehicleMappings": [{"vin": "VINCAP2"}]})
        if url.endswith("/customers/vehicles/VINCAP2/basicData"):
            return _DummyResponse(payload={"vin": "VINCAP2", "lastUpdatedAt": "2026-03-11T10:00:00Z", "battery": {"socPercent": 88}})
        if url.endswith("/customers/containers"):
            return _DummyResponse(payload={"containers": [{"containerId": "CONTCAP2", "state": "ACTIVE"}]})
        if url.endswith("/customers/vehicles/VINCAP2/telematicData?containerId=CONTCAP2"):
            return _DummyResponse(payload={"charging": {"plugConnectionState": "CONNECTED", "chargingState": "CHARGING"}})
        return _DummyResponse(status_code=500, text="unexpected")

    monkeypatch.setattr("bmw_cardata_provider.requests.get", fake_get)
    provider = BmwCarDataProvider(config={"bmw_enabled": True}, storage=BmwStorage(str(tmp_path / "raw.jsonl"), str(tmp_path / "state.json")), auth=_Auth())
    out = provider.refresh_once()

    assert out["ok"] is True
    assert any("customers_containers" in Path(p).name for p in out["capture_files"])
    assert any("customers_vehicles_VINCAP2_telematicData" in Path(p).name for p in out["capture_files"])
