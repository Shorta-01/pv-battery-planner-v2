from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from bmw_auth import BmwAuthClient
from bmw_mapping import apply_planner_derivations, map_bmw_payload_to_vehicle_states
from bmw_models import BmwProviderStatus, NormalizedVehicleState, RawEventRecord, parse_dt, utcnow
from bmw_storage import BmwStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BmwOperation:
    name: str
    path_template: str
    stage: str

    def resolve(self, vin: str | None = None) -> str:
        if "{vin}" in self.path_template:
            if not vin:
                raise ValueError("vin required for operation")
            return self.path_template.replace("{vin}", vin)
        return self.path_template


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

    def rest_operations(self) -> list[_BmwOperation]:
        ops = [
            _BmwOperation(name="vehicle_mappings", path_template="/customers/vehicles/mappings", stage="discover"),
            _BmwOperation(name="vehicle_basic_data", path_template="/customers/vehicles/{vin}/basicData", stage="vehicle"),
            _BmwOperation(name="containers", path_template="/customers/containers", stage="discover"),
            _BmwOperation(name="vehicle_telematic_data", path_template="/customers/vehicles/{vin}/telematicData", stage="vehicle"),
        ]
        if bool(self.config.get("bmw_enable_optional_chargingprofile_followup", False)):
            ops.append(
                _BmwOperation(name="vehicle_charging_profile", path_template="/customers/vehicles/{vin}/chargingprofile", stage="vehicle_optional")
            )
        return ops

    def rest_endpoints(self) -> list[str]:
        return [op.path_template for op in self.rest_operations()]

    def request_versioning_mode(self) -> str:
        return "header:X-Version=v1"

    def rest_headers(self, access_token: str, *, include_json_content_type: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "X-Version": "v1",
        }
        if include_json_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _safe_error_excerpt(self, resp: requests.Response) -> str:
        return str((resp.text or "")[:300]).replace("\n", " ").strip()

    def _mark_last_rest_result(self, endpoint: str, status_code: int | None, error_excerpt: str | None = None) -> None:
        self.status.last_rest_endpoint = endpoint
        self.status.last_rest_status_code = status_code
        self.status.last_rest_error_excerpt = error_excerpt

    def _raise_http_error(self, endpoint: str, resp: requests.Response) -> None:
        body_excerpt = self._safe_error_excerpt(resp)
        self._mark_last_rest_result(endpoint, resp.status_code, body_excerpt)
        logger.error(
            "BMW provider REST error endpoint=%s status=%s body_excerpt=%s",
            endpoint,
            resp.status_code,
            body_excerpt,
        )
        raise RuntimeError(
            f"BMW REST call failed: status={resp.status_code} endpoint={endpoint} body={body_excerpt} auth_mode=Bearer access_token versioning={self.request_versioning_mode()}"
        )

    def _request_json(
        self,
        *,
        base: str,
        path: str,
        headers: dict[str, str],
        aggregate_payload: dict[str, Any],
        capture_paths: list[str],
        stage: str,
        optional: bool,
        capture_endpoint_path: str | None = None,
    ) -> Any:
        endpoint = f"{base}{path}"
        self.status.refresh_sequence_endpoints.append(path)
        resp = requests.get(endpoint, headers=headers, timeout=20)
        self._mark_last_rest_result(endpoint, resp.status_code)
        if "/telematicData" in path:
            self.status.last_telematic_status_code = resp.status_code
        if resp.status_code >= 400:
            safe_error = self._safe_error_excerpt(resp)
            self._mark_last_rest_result(endpoint, resp.status_code, safe_error)
            aggregate_payload[path] = {"_error": {"status": resp.status_code, "excerpt": safe_error}}
            aggregate_payload["sequence"].append(
                {
                    "stage": stage,
                    "endpoint": path,
                    "ok": False,
                    "status": resp.status_code,
                    "error_excerpt": safe_error,
                    "optional": optional,
                }
            )
            if optional:
                return None
            self._raise_http_error(endpoint, resp)

        node = resp.json()
        aggregate_payload[path] = node
        aggregate_payload["sequence"].append({"stage": stage, "endpoint": path, "ok": True, "optional": optional})
        capture_paths.append(str(self.storage.store_raw_capture(capture_endpoint_path or path, node, status_code=resp.status_code)))
        return node

    def refresh_once(self) -> dict[str, Any]:
        if not bool(self.config.get("bmw_enabled", False)):
            self.status.provider_status = "disabled"
            return {"ok": False, "reason": "disabled"}

        token = self.auth.refresh_if_possible(self.auth.load_token())
        if not token.access_token:
            self.status.provider_status = "auth_required"
            self.status.last_error = "Missing access_token"
            return {"ok": False, "reason": "auth_required"}

        headers = self.rest_headers(token.access_token)
        self.status.stream_connected = False
        self.status.provider_status = "polling"
        self.status.stream_status = "disabled" if not bool(self.config.get("bmw_stream_enabled", False)) else "not_implemented"
        self.status.request_versioning_mode = self.request_versioning_mode()

        aggregate_payload: dict[str, Any] = {"sequence": []}
        capture_paths: list[str] = []
        base = self.rest_base_url()
        self.status.refresh_sequence_endpoints = []
        self.status.capture_files_written = []
        self.status.mapping_diagnostics = []
        self.status.discovered_container_ids = []
        self.status.active_container_id = None
        self.status.container_diagnostics = []
        self.status.last_telematic_url = None
        self.status.last_telematic_status_code = None
        self.status.active_vehicle_id = None
        self.status.vehicle_data_mode = "unknown"

        try:
            discovery_op = self.rest_operations()[0]
            mappings_payload = self._request_json(
                base=base,
                path=discovery_op.resolve(),
                headers=headers,
                aggregate_payload=aggregate_payload,
                capture_paths=capture_paths,
                stage=discovery_op.stage,
                optional=False,
            )
            discovered_ids, mapping_diagnostics = self._discover_vehicle_ids(mappings_payload)
            self.status.discovered_vehicle_ids = list(discovered_ids)
            self.status.mapping_diagnostics = list(mapping_diagnostics)

            if not discovered_ids:
                self.status.provider_status = "degraded"
                msg = "no accessible BMW vehicle mappings"
                self.status.last_error = msg
                self.status.capture_files_written = list(capture_paths)
                self.status.vehicle_data_mode = "none"
                self.storage.append_raw_event(
                    RawEventRecord(provider="bmw_cardata", received_at=utcnow(), payload=aggregate_payload, parse_ok=False, parse_error=msg)
                )
                return {
                    "ok": False,
                    "reason": "no_vehicles",
                    "message": msg,
                    "endpoints": list(self.status.refresh_sequence_endpoints),
                    "capture_files": capture_paths,
                    "request_versioning_mode": self.status.request_versioning_mode,
                    "rest_token_mode": "access_token",
                    "mapping_diagnostics": list(mapping_diagnostics),
                    "discovered_container_ids": list(self.status.discovered_container_ids),
                    "active_container_id": self.status.active_container_id,
                    "container_diagnostics": list(self.status.container_diagnostics),
                }

            target_vehicle_id = self._select_active_vehicle(discovered_ids, mapping_diagnostics)
            self.status.active_vehicle_id = target_vehicle_id
            logger.debug(
                "BMW mapping discovery discovered_vehicle_ids=%s active_vehicle_id=%s diagnostics=%s",
                self.status.discovered_vehicle_ids,
                self.status.active_vehicle_id,
                self.status.mapping_diagnostics,
            )

            basic_path = "/customers/vehicles/{vin}/basicData".replace("{vin}", target_vehicle_id)
            self._request_json(
                base=base,
                path=basic_path,
                headers=headers,
                aggregate_payload=aggregate_payload,
                capture_paths=capture_paths,
                stage="vehicle",
                optional=False,
            )

            containers_payload = self._request_json(
                base=base,
                path="/customers/containers",
                headers=headers,
                aggregate_payload=aggregate_payload,
                capture_paths=capture_paths,
                stage="discover",
                optional=True,
            )
            container_ids, container_diags = self._discover_containers(containers_payload)
            self.status.discovered_container_ids = list(container_ids)
            self.status.container_diagnostics = list(container_diags)
            active_container_id = self._select_active_container(container_diags)
            self.status.active_container_id = active_container_id

            telematic_path = f"/customers/vehicles/{target_vehicle_id}/telematicData"
            if active_container_id:
                telematic_path = f"{telematic_path}?containerId={active_container_id}"
            self.status.last_telematic_url = f"{base}{telematic_path}"

            self._request_json(
                base=base,
                path=telematic_path,
                headers=headers,
                aggregate_payload=aggregate_payload,
                capture_paths=capture_paths,
                stage="vehicle",
                optional=True,
                capture_endpoint_path=f"/customers/vehicles/{target_vehicle_id}/telematicData",
            )

            for op in self.rest_operations():
                if op.name != "vehicle_charging_profile":
                    continue
                try:
                    self._request_json(
                        base=base,
                        path=op.resolve(target_vehicle_id),
                        headers=headers,
                        aggregate_payload=aggregate_payload,
                        capture_paths=capture_paths,
                        stage=op.stage,
                        optional=True,
                    )
                except Exception:
                    pass

            self.status.last_auth_refresh = token.obtained_at
            self.status.capture_files_written = list(capture_paths)
            self._ingest_payload(aggregate_payload)
            self.status.provider_status = "healthy" if self.vehicles else "degraded"
            telematic_path_prefix = f"/customers/vehicles/{target_vehicle_id}/telematicData"
            telematic_node = next((v for k, v in aggregate_payload.items() if isinstance(k, str) and k.startswith(telematic_path_prefix)), None)
            self.status.vehicle_data_mode = "live_telematics" if isinstance(telematic_node, dict) and "_error" not in telematic_node else "static_only"
            return {
                "ok": bool(self.vehicles),
                "vehicles": len(self.vehicles),
                "endpoints": list(self.status.refresh_sequence_endpoints),
                "capture_files": capture_paths,
                "request_versioning_mode": self.status.request_versioning_mode,
                "rest_token_mode": "access_token",
                "active_vehicle_id": self.status.active_vehicle_id,
                "discovered_vehicle_ids": list(self.status.discovered_vehicle_ids),
                "mapping_diagnostics": list(self.status.mapping_diagnostics),
                "discovered_container_ids": list(self.status.discovered_container_ids),
                "active_container_id": self.status.active_container_id,
                "container_diagnostics": list(self.status.container_diagnostics),
            }
        except Exception as exc:
            self.status.last_error = str(exc)
            self.status.provider_status = "degraded"
            self.status.capture_files_written = list(capture_paths)
            logger.warning("BMW provider poll failed: %s", exc)
            return {
                "ok": False,
                "reason": "poll_failed",
                "error": str(exc),
                "endpoints": list(self.status.refresh_sequence_endpoints),
                "request_versioning_mode": self.status.request_versioning_mode,
                "rest_token_mode": "access_token",
                "active_vehicle_id": self.status.active_vehicle_id,
                "discovered_vehicle_ids": list(self.status.discovered_vehicle_ids),
                "mapping_diagnostics": list(self.status.mapping_diagnostics),
                "discovered_container_ids": list(self.status.discovered_container_ids),
                "active_container_id": self.status.active_container_id,
                "container_diagnostics": list(self.status.container_diagnostics),
                "capture_files": capture_paths,
            }

    def _discover_vehicle_ids(self, discovery_payload: Any) -> tuple[list[str], list[dict[str, Any]]]:
        raw_rows: list[dict[str, Any]] = []
        if isinstance(discovery_payload, list):
            raw_rows.extend(x for x in discovery_payload if isinstance(x, dict))
        elif isinstance(discovery_payload, dict):
            if isinstance(discovery_payload.get("vehicleMappings"), list):
                raw_rows.extend(x for x in discovery_payload.get("vehicleMappings") if isinstance(x, dict))
            if isinstance(discovery_payload.get("vehicles"), list):
                raw_rows.extend(x for x in discovery_payload.get("vehicles") if isinstance(x, dict))

        ids: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        for row in raw_rows:
            vehicle_id = str(row.get("vin") or row.get("vehicleId") or row.get("id") or "").strip()
            if not vehicle_id:
                continue
            role = row.get("mappingType") or row.get("mappingRole") or row.get("relationshipType")
            is_primary = row.get("primary") if isinstance(row.get("primary"), bool) else None
            diag = {
                "vehicle_id": vehicle_id,
                "mapping_role": role,
                "is_primary": is_primary,
                "display_name": row.get("displayName") or row.get("name"),
            }
            diagnostics.append(diag)
            if vehicle_id not in ids:
                ids.append(vehicle_id)
        return ids, diagnostics

    def _select_active_vehicle(self, discovered_ids: list[str], mapping_diagnostics: list[dict[str, Any]]) -> str:
        configured = str(self.config.get("bmw_active_vehicle_id") or "").strip()
        if configured and configured in discovered_ids:
            return configured

        primary_ids = [
            str(diag.get("vehicle_id"))
            for diag in mapping_diagnostics
            if str(diag.get("vehicle_id") or "") in discovered_ids
            and (
                diag.get("is_primary") is True
                or str(diag.get("mapping_role") or "").upper() in {"PRIMARY", "OWNER", "MAIN_USER"}
            )
        ]
        return primary_ids[0] if primary_ids else discovered_ids[0]


    def _discover_containers(self, payload: Any) -> tuple[list[str], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        if isinstance(payload, list):
            rows.extend(x for x in payload if isinstance(x, dict))
        elif isinstance(payload, dict):
            for key in ("containers", "customerContainers", "items", "data"):
                if isinstance(payload.get(key), list):
                    rows.extend(x for x in payload.get(key) if isinstance(x, dict))
            if not rows:
                rows.append(payload)

        seen: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        for row in rows:
            container_id = str(row.get("containerId") or row.get("id") or row.get("identifier") or "").strip()
            if not container_id:
                continue
            if container_id not in seen:
                seen.append(container_id)
            diagnostics.append(
                {
                    "container_id": container_id,
                    "state": str(row.get("state") or ""),
                    "name": row.get("name") or row.get("containerName"),
                    "purpose": row.get("purpose") or row.get("type"),
                    "created_at": row.get("createdAt") or row.get("creationTime") or row.get("created"),
                    "updated_at": row.get("updatedAt") or row.get("lastUpdatedAt") or row.get("updateTime"),
                }
            )
        return seen, diagnostics

    def _select_active_container(self, diagnostics: list[dict[str, Any]]) -> str | None:
        if not diagnostics:
            return None
        active = [d for d in diagnostics if str(d.get("state") or "").upper() == "ACTIVE"]
        candidates = active if active else diagnostics
        candidates = sorted(
            candidates,
            key=lambda d: (
                parse_dt(d.get("updated_at")) or parse_dt(d.get("created_at")) or utcnow(),
                str(d.get("container_id") or ""),
            ),
            reverse=True,
        )
        chosen = candidates[0] if candidates else None
        return str(chosen.get("container_id")) if chosen else None

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
        if newest:
            age = int((utcnow() - newest).total_seconds())
            self.status.data_status = "fresh" if age < 120 else "aging" if age < 600 else "stale" if age < 1800 else "partial"
        else:
            self.status.data_status = "partial"

    def manual_refresh(self) -> dict[str, Any]:
        return self.refresh_once()
