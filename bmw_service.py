from __future__ import annotations

from typing import Any

from bmw_auth import BmwAuthClient
from bmw_cardata_provider import BmwCarDataProvider
from bmw_storage import BmwStorage


class BmwService:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.storage = BmwStorage(
            raw_event_store_path=str(config.get("bmw_raw_event_store_path", "local_state/bmw_raw_events.jsonl")),
            vehicle_state_store_path=str(config.get("bmw_vehicle_state_store_path", "local_state/bmw_vehicle_state.json")),
        )
        self.auth = BmwAuthClient(
            client_id=str(config.get("bmw_client_id", "")),
            token_cache_path=str(config.get("bmw_token_cache_path", "local_state/bmw_token.json")),
            auth_base_url=str(config.get("bmw_auth_base_url", "https://customer.bmwgroup.com/gcdm/oauth")),
        )
        self.provider = BmwCarDataProvider(config=config, storage=self.storage, auth=self.auth)

    def update_config(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider.config = config

    def vehicles(self) -> dict[str, dict]:
        return {vid: st.to_dict() for vid, st in self.provider.vehicles.items()}

    def provider_status(self) -> dict[str, Any]:
        return self.provider.status.to_dict()

    def manual_refresh(self) -> dict[str, Any]:
        return self.provider.manual_refresh()

    def start_device_flow(self) -> dict[str, Any]:
        return self.auth.start_device_flow()

    def poll_device_token(self, device_code: str) -> dict[str, Any]:
        token = self.auth.poll_device_token(device_code)
        return {
            "ok": bool(token.id_token),
            "expires_at": token.to_dict().get("expires_at"),
            "token_status": "valid" if token.is_fresh() else "expiring",
        }


    def device_flow_debug_info(self) -> dict[str, str]:
        return {
            "device_flow_start_url": self.auth.device_flow_start_url(),
            "device_flow_poll_url": self.auth.device_flow_poll_url(),
        }
