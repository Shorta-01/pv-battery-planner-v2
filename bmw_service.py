from __future__ import annotations

import inspect
from typing import Any
import logging

from bmw_auth import BmwAuthClient
from bmw_cardata_provider import BmwCarDataProvider
from bmw_logging import bmw_log_context
from bmw_storage import BmwStorage


logger = logging.getLogger(__name__)


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"


class BmwService:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = {}
        self.storage: BmwStorage | None = None
        self.auth: BmwAuthClient | None = None
        self.provider: BmwCarDataProvider | None = None
        self.runtime_generation = 0
        self.last_runtime_rebuild_reason = "init"
        self.update_config(config)

    def _build_storage(self, config: dict[str, Any]) -> BmwStorage:
        return BmwStorage(
            raw_event_store_path=str(config.get("bmw_raw_event_store_path", "local_state/bmw_raw_events.jsonl")),
            vehicle_state_store_path=str(config.get("bmw_vehicle_state_store_path", "local_state/bmw_vehicle_state.json")),
        )

    def _build_auth(self, config: dict[str, Any]) -> BmwAuthClient:
        return BmwAuthClient(
            client_id=str(config.get("bmw_client_id", "")),
            token_cache_path=str(config.get("bmw_token_cache_path", "local_state/bmw_token.json")),
            auth_base_url=str(config.get("bmw_auth_base_url", "https://customer.bmwgroup.com/gcdm/oauth")),
        )

    def _rebuild_runtime(self, reason: str) -> None:
        logger.info(
            "BMW service runtime rebuild start %s",
            bmw_log_context(operation="bmw_runtime_rebuild", bmw_operation="service_runtime", phase="start", reason=reason),
        )
        self.storage = self._build_storage(self.config)
        self.auth = self._build_auth(self.config)
        self.provider = BmwCarDataProvider(config=self.config, storage=self.storage, auth=self.auth)
        self.runtime_generation += 1
        self.last_runtime_rebuild_reason = reason
        logger.info(
            "BMW service runtime rebuild complete %s",
            bmw_log_context(
                operation="bmw_runtime_rebuild",
                bmw_operation="service_runtime",
                phase="complete",
                reason=reason,
                runtime_generation=self.runtime_generation,
            ),
        )

    def update_config(self, config: dict[str, Any]) -> None:
        runtime_cfg = dict(config or {})
        ev_cfg = runtime_cfg.get("ev_vehicle_data") if isinstance(runtime_cfg.get("ev_vehicle_data"), dict) else None
        if ev_cfg is not None:
            merged = dict(ev_cfg)
            for key in (
                "charger_max_power_kw",
                "petrol_price_eur_per_l",
                "petrol_consumption_l_per_100km",
                "bmw_enabled",
                "bmw_client_id",
                "bmw_token_cache_path",
                "bmw_raw_event_store_path",
                "bmw_vehicle_state_store_path",
                "bmw_healthcheck_seconds",
            ):
                if key in runtime_cfg and key not in merged:
                    merged[key] = runtime_cfg.get(key)
            self.config = merged
        else:
            self.config = runtime_cfg
        self._rebuild_runtime("config_update")

    def vehicles(self) -> dict[str, dict]:
        assert self.provider is not None
        return {vid: st.to_dict() for vid, st in self.provider.vehicles.items()}

    def provider_status(self) -> dict[str, Any]:
        assert self.provider is not None
        return self.provider.status.to_dict()

    def manual_refresh(self, *, force_reprobe: bool = False) -> dict[str, Any]:
        assert self.provider is not None
        logger.info(
            "BMW service manual refresh start %s",
            bmw_log_context(operation="ev_manual_refresh", bmw_operation="manual_refresh", phase="start", force_reprobe=bool(force_reprobe)),
        )
        return self.provider.manual_refresh(force_reprobe=force_reprobe)

    def start_device_flow(self) -> dict[str, Any]:
        assert self.auth is not None
        logger.info(
            "BMW service device flow start %s",
            bmw_log_context(operation="ev_device_flow_start", bmw_operation="device_flow", phase="start"),
        )
        return self.auth.start_device_flow()

    def poll_device_token(self, device_code: str) -> dict[str, Any]:
        assert self.auth is not None
        logger.info(
            "BMW service device flow poll %s",
            bmw_log_context(
                operation="ev_device_flow_poll",
                bmw_operation="device_flow",
                phase="poll_token",
                has_device_code=bool(str(device_code or "").strip()),
            ),
        )
        token = self.auth.poll_device_token(device_code)
        return {
            "ok": bool(token.access_token),
            "expires_at": token.to_dict().get("expires_at"),
            "token_status": "valid" if token.is_fresh() else "expiring",
        }

    def device_flow_debug_info(self) -> dict[str, Any]:
        assert self.auth is not None
        assert self.provider is not None
        session = self.auth.get_device_flow_session()
        safe_session = {
            "client_id_masked": _mask_secret(str(session.get("client_id") or "")),
            "device_code": session.get("device_code"),
            "user_code": session.get("user_code"),
            "verification_uri": session.get("verification_uri"),
            "interval": session.get("interval"),
            "created_at": session.get("created_at"),
            "expires_in": session.get("expires_in"),
            "expires_at": session.get("expires_at"),
        }
        return {
            "bmw_auth_module_path": inspect.getfile(self.auth.__class__),
            "runtime_generation": self.runtime_generation,
            "provider_rebuilt_after_config_update": self.last_runtime_rebuild_reason == "config_update",
            "active_client_id_masked": _mask_secret(str(self.auth.client_id or "")),
            "active_auth_base_url": self.auth.auth_base_url,
            "device_flow_start_url": self.auth.device_flow_start_url(),
            "device_flow_poll_url": self.auth.device_flow_poll_url(),
            "rest_api_base_url": self.provider.rest_base_url(),
            "rest_endpoints": self.provider.rest_endpoints(),
            "refresh_sequence_endpoints": list(self.provider.status.refresh_sequence_endpoints),
            "request_versioning_mode": self.provider.status.request_versioning_mode or self.provider.request_versioning_mode(),
            "rest_token_mode": "access_token",
            "last_rest_endpoint_attempted": self.provider.status.last_rest_endpoint,
            "last_rest_status_code": self.provider.status.last_rest_status_code,
            "last_rest_safe_error_excerpt": self.provider.status.last_rest_error_excerpt,
            "discovered_vehicle_ids": list(self.provider.status.discovered_vehicle_ids),
            "active_vehicle_id": self.provider.status.active_vehicle_id,
            "mapping_diagnostics": list(self.provider.status.mapping_diagnostics),
            "discovered_container_ids": list(self.provider.status.discovered_container_ids),
            "active_container_id": self.provider.status.active_container_id,
            "container_diagnostics": list(self.provider.status.container_diagnostics),
            "last_telematic_url": self.provider.status.last_telematic_url,
            "last_telematic_status_code": self.provider.status.last_telematic_status_code,
            "stream_status": self.provider.status.stream_status,
            "stream_enabled": bool(self.config.get("bmw_stream_enabled", False)),
            "token_cache_path": str(self.auth.token_cache_path),
            "raw_event_store_path": str(self.storage.raw_path) if self.storage else None,
            "vehicle_state_store_path": str(self.storage.state_path) if self.storage else None,
            "recent_capture_files": self.storage.list_raw_captures(limit=5) if self.storage else [],
            "capture_files_written": list(self.provider.status.capture_files_written),
            "vehicle_data_mode": self.provider.status.vehicle_data_mode,
            "has_live_telematics": self.provider.status.vehicle_data_mode == "live_telematics",
            "container_auto_create_attempted": self.provider.status.container_auto_create_attempted,
            "container_auto_create_succeeded": self.provider.status.container_auto_create_succeeded,
            "force_reprobe_diagnostics": dict(self.provider.status.force_reprobe_diagnostics),
            "bmw_ev_diagnostics": dict(self.provider.status.bmw_ev_diagnostics),
            "device_flow_session": safe_session,
        }
