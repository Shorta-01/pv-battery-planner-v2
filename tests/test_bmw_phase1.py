import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import datetime as dt

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
