from __future__ import annotations

import logging
from typing import Any

import requests

from bmw_auth import BmwAuthClient
from bmw_mapping import apply_planner_derivations, map_bmw_payload_to_vehicle_states
from bmw_models import BmwProviderStatus, NormalizedVehicleState, RawEventRecord, utcnow
from bmw_storage import BmwStorage

logger = logging.getLogger(__name__)


class BmwCarDataProvider:
    REST_BASE_URL = "https://api-cardata.bmwgroup.com"

    def __init__(self, config: dict[str, Any], storage: BmwStorage, auth: BmwAuthClient) -> None:
        self.config = config
        self.storage = storage
        self.auth = auth
        self.status = BmwProviderStatus(provider_status="initializing")
        self.vehicles: dict[str, NormalizedVehicleState] = storage.load_vehicle_states()
        self.status.provider_status = "ready"

    def update_runtime(self, *, config: dict[str, Any], storage: BmwStorage, auth: BmwAuthClient) -> None:
        self.config = config
        self.storage = storage
        self.auth = auth

    def rest_base_url(self) -> str:
        return str(self.config.get("bmw_api_base_url", self.REST_BASE_URL)).rstrip("/")

    def rest_endpoints(self) -> list[str]:
        return [
            "/v1/vehicle-mappings",
            "/v1/vehicles",
        ]

    def _raise_http_error(self, endpoint: str, resp: requests.Response) -> None:
        body_excerpt = resp.text[:300]
        logger.error(
            "BMW provider REST error endpoint=%s status=%s body_excerpt=%s",
            endpoint,
            resp.status_code,
            body_excerpt,
        )
        raise RuntimeError(
            f"BMW REST call failed: status={resp.status_code} endpoint={endpoint} body={body_excerpt} auth_mode=Bearer access_token"
        )

    def refresh_once(self) -> dict[str, Any]:
        if not bool(self.config.get("bmw_enabled", False)):
            self.status.provider_status = "disabled"
            return {"ok": False, "reason": "disabled"}
        token = self.auth.refresh_if_possible(self.auth.load_token())
        if not token.access_token:
            self.status.provider_status = "auth_required"
            self.status.last_error = "Missing access_token"
            return {"ok": False, "reason": "auth_required"}

        headers = {"Authorization": f"Bearer {token.access_token}"}
        self.status.stream_connected = False
        self.status.provider_status = "polling"
        self.status.stream_status = "disabled" if not bool(self.config.get("bmw_stream_enabled", False)) else "not_implemented"
        try:
            aggregate_payload: dict[str, Any] = {}
            base = self.rest_base_url()
            capture_paths: list[str] = []
            for path in self.rest_endpoints():
                endpoint = f"{base}{path}"
                resp = requests.get(endpoint, headers=headers, timeout=20)
                if resp.status_code >= 400:
                    self._raise_http_error(endpoint, resp)
                endpoint_payload = resp.json()
                aggregate_payload[path] = endpoint_payload
                capture_paths.append(str(self.storage.store_raw_capture(path, endpoint_payload)))
            self.status.last_auth_refresh = token.obtained_at
            self._ingest_payload(aggregate_payload)
            self.status.provider_status = "healthy"
            return {
                "ok": True,
                "vehicles": len(self.vehicles),
                "endpoints": self.rest_endpoints(),
                "capture_files": capture_paths,
            }
        except Exception as exc:
            self.status.last_error = str(exc)
            self.status.provider_status = "degraded"
            logger.warning("BMW provider poll failed: %s", exc)
            return {"ok": False, "reason": "poll_failed", "error": str(exc)}

    def _ingest_payload(self, payload: dict[str, Any]) -> None:
        records = map_bmw_payload_to_vehicle_states(payload)
        self.status.last_raw_event_received = utcnow()
        if not records:
            self.storage.append_raw_event(
                RawEventRecord(provider="bmw_cardata", received_at=utcnow(), payload=payload, parse_ok=False, parse_error="No vehicles mapped")
            )
            return

        petrol = self.config.get("petrol_price_eur_per_l")
        consumption = self.config.get("petrol_consumption_l_per_100km")
        charger_cap = self.config.get("charger_max_power_kw")

        for st in records:
            st = apply_planner_derivations(
                st,
                petrol_price_eur_per_l=float(petrol) if petrol not in (None, "") else None,
                petrol_consumption_l_per_100km=float(consumption) if consumption not in (None, "") else None,
                charger_max_power_kw=float(charger_cap) if charger_cap not in (None, "") else None,
            )
            self.vehicles[st.vehicle_id] = st
            self.status.last_vehicle_update = st.last_update_ts or utcnow()
            self.storage.append_raw_event(
                RawEventRecord(
                    provider="bmw_cardata",
                    received_at=utcnow(),
                    payload=payload,
                    vehicle_id=st.vehicle_id,
                    event_type="vehicle_update",
                    parse_ok=True,
                )
            )
        self.storage.save_vehicle_states(self.vehicles)
        newest = max((v.last_update_ts for v in self.vehicles.values() if v.last_update_ts), default=None)
        age = int((utcnow() - newest).total_seconds()) if newest else 3600
        self.status.data_status = "fresh" if age < 120 else "aging" if age < 600 else "stale" if age < 1800 else "error"

    def manual_refresh(self) -> dict[str, Any]:
        return self.refresh_once()
