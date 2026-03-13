from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime as dt
from typing import Any


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_or_none(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_dt(value: Any) -> dt.datetime | None:
    if not value:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, str):
        try:
            v = value.replace("Z", "+00:00")
            parsed = dt.datetime.fromisoformat(v)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return None
    return None


@dataclass
class BmwTokenData:
    access_token: str | None = None
    refresh_token: str | None = None
    id_token: str | None = None
    token_type: str | None = None
    expires_at: dt.datetime | None = None
    obtained_at: dt.datetime | None = None

    def is_fresh(self, min_valid_seconds: int = 60) -> bool:
        if self.expires_at is None:
            return False
        return (self.expires_at - utcnow()).total_seconds() > min_valid_seconds

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["expires_at"] = iso_or_none(self.expires_at)
        out["obtained_at"] = iso_or_none(self.obtained_at)
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BmwTokenData":
        return cls(
            access_token=payload.get("access_token"),
            refresh_token=payload.get("refresh_token"),
            id_token=payload.get("id_token"),
            token_type=payload.get("token_type"),
            expires_at=parse_dt(payload.get("expires_at")),
            obtained_at=parse_dt(payload.get("obtained_at")),
        )


@dataclass
class BmwProviderStatus:
    provider: str = "bmw_cardata"
    provider_status: str = "disabled"
    data_status: str = "stale"
    stream_connected: bool = False
    stream_status: str = "not_implemented"
    last_error: str | None = None
    last_auth_refresh: dt.datetime | None = None
    last_raw_event_received: dt.datetime | None = None
    last_vehicle_update: dt.datetime | None = None
    last_rest_endpoint: str | None = None
    last_rest_status_code: int | None = None
    last_rest_error_excerpt: str | None = None
    request_versioning_mode: str | None = None
    active_vehicle_id: str | None = None
    discovered_vehicle_ids: list[str] = field(default_factory=list)
    refresh_sequence_endpoints: list[str] = field(default_factory=list)
    mapping_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    discovered_container_ids: list[str] = field(default_factory=list)
    active_container_id: str | None = None
    container_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    last_telematic_url: str | None = None
    last_telematic_status_code: int | None = None
    capture_files_written: list[str] = field(default_factory=list)
    vehicle_data_mode: str = "unknown"
    container_auto_create_attempted: bool = False
    container_auto_create_succeeded: bool = False
    force_reprobe_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_status": self.provider_status,
            "data_status": self.data_status,
            "stream_connected": self.stream_connected,
            "stream_status": self.stream_status,
            "last_error": self.last_error,
            "last_auth_refresh": iso_or_none(self.last_auth_refresh),
            "last_raw_event_received": iso_or_none(self.last_raw_event_received),
            "last_vehicle_update": iso_or_none(self.last_vehicle_update),
            "last_rest_endpoint": self.last_rest_endpoint,
            "last_rest_status_code": self.last_rest_status_code,
            "last_rest_error_excerpt": self.last_rest_error_excerpt,
            "request_versioning_mode": self.request_versioning_mode,
            "active_vehicle_id": self.active_vehicle_id,
            "discovered_vehicle_ids": list(self.discovered_vehicle_ids),
            "refresh_sequence_endpoints": list(self.refresh_sequence_endpoints),
            "mapping_diagnostics": list(self.mapping_diagnostics),
            "discovered_container_ids": list(self.discovered_container_ids),
            "active_container_id": self.active_container_id,
            "container_diagnostics": list(self.container_diagnostics),
            "last_telematic_url": self.last_telematic_url,
            "last_telematic_status_code": self.last_telematic_status_code,
            "capture_files_written": list(self.capture_files_written),
            "vehicle_data_mode": self.vehicle_data_mode,
            "container_auto_create_attempted": self.container_auto_create_attempted,
            "container_auto_create_succeeded": self.container_auto_create_succeeded,
            "force_reprobe_diagnostics": dict(self.force_reprobe_diagnostics),
        }


@dataclass
class NormalizedVehicleState:
    vehicle_id: str
    display_name: str | None = None
    data_provider: str = "bmw_cardata"
    data_status: str = "stale"
    last_update_ts: dt.datetime | None = None
    freshness_seconds: int | None = None
    soc_pct: float | None = None
    is_plugged: bool | None = None
    is_charging: bool | None = None
    range_km: float | None = None
    time_to_full_min: int | None = None
    charge_power_kw: float | None = None
    ac_current_limit_a: float | None = None
    battery_capacity_kwh: float | None = None
    charging_mode: str | None = None
    optimized_charging_preference: str | None = None
    charge_window_start: str | None = None
    charge_window_end: str | None = None
    odometer_km: float | None = None
    travelled_distance_km: float | None = None
    plug_status_raw: str | None = None
    flap_lock_status_raw: str | None = None
    charge_error_raw: str | None = None
    charge_session_active: bool | None = None
    energy_needed_kwh: float | None = None
    effective_charge_power_limit_kw: float | None = None
    max_deliverable_kwh: float | None = None
    max_reachable_soc_pct: float | None = None
    planner_demand_active: bool | None = None
    planned_energy_kwh: float | None = None
    planned_charge_cost_eur: float | None = None
    avoided_petrol_cost_eur: float | None = None
    net_economic_benefit_eur: float | None = None
    expected_full_charge_ts: dt.datetime | None = None
    planner_status_text: str | None = None
    planner_priority: str | None = None
    field_availability: dict[str, bool] = field(default_factory=dict)
    raw_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["last_update_ts"] = iso_or_none(self.last_update_ts)
        out["expected_full_charge_ts"] = iso_or_none(self.expected_full_charge_ts)
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizedVehicleState":
        data = dict(payload)
        data["last_update_ts"] = parse_dt(payload.get("last_update_ts"))
        data["expected_full_charge_ts"] = parse_dt(payload.get("expected_full_charge_ts"))
        return cls(**data)


@dataclass
class RawEventRecord:
    provider: str
    received_at: dt.datetime
    payload: dict[str, Any]
    vehicle_id: str | None = None
    event_type: str | None = None
    parse_ok: bool = True
    parse_error: str | None = None
    mapping_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "received_at": iso_or_none(self.received_at),
            "vehicle_id": self.vehicle_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "mapping_version": self.mapping_version,
        }
