from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

from bmw_auth import BmwAuthClient
from bmw_cardata_contract import CreateContainerRequest
from bmw_logging import bmw_log_context
from bmw_mapping import (
    CRITICAL_EV_DESCRIPTOR_FIELDS,
    apply_planner_derivations,
    build_critical_ev_field_evidence,
    extract_vehicle_telematic_payload,
    map_bmw_payload_to_vehicle_states,
)
from bmw_models import BmwProviderStatus, NormalizedVehicleState, RawEventRecord, parse_dt, utcnow
from bmw_storage import BmwStorage

logger = logging.getLogger(__name__)

CRITICAL_EV_DESCRIPTORS = list(CRITICAL_EV_DESCRIPTOR_FIELDS.keys())


def _descriptor_alias(descriptor: str) -> str:
    return descriptor.split(".")[-1].replace("stateOfCharge", "soc").replace("timeToComplete", "time_to_complete")


def _safe_preview(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:80]
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(k) for k in value.keys())[:6]}
    if isinstance(value, list):
        return {"type": "list", "len": len(value)}
    return str(value)[:80]


PHASE1_CONTAINER_DEFINITION: dict[str, Any] = {
    "name": "pvbp_phase1_ev_telematics",
    "purpose": "PV Battery Planner phase 1 EV/PHEV telematics",
    # Validated against BMW CarData generated descriptor catalog naming used during
    # this integration hardening cycle.
    "candidate_phase1_descriptors": [
        "vehicle.drivetrain.electricEngine.battery.stateOfCharge",
        "vehicle.drivetrain.electricEngine.range.electric",
        "vehicle.drivetrain.electricEngine.charging.status",
        "vehicle.drivetrain.electricEngine.charging.timeToComplete",
        "vehicle.drivetrain.electricEngine.charging.power",
        "vehicle.body.chargingPort.status",
        "vehicle.powertrain.electric.battery.charging.acLimit.selected",
    ],
}


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
            "BMW provider REST error %s endpoint=%s status=%s body_excerpt=%s",
            bmw_log_context(operation="bmw_refresh_once", bmw_operation="provider_request", phase="http_error"),
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
        method: str,
        base: str,
        path: str,
        headers: dict[str, str],
        aggregate_payload: dict[str, Any],
        capture_paths: list[str],
        stage: str,
        optional: bool,
        capture_endpoint_path: str | None = None,
        json_body: dict[str, Any] | None = None,
        raw_body: str | None = None,
    ) -> Any:
        endpoint = f"{base}{path}"
        self.status.refresh_sequence_endpoints.append(f"{method.upper()} {path}")
        try:
            if method.upper() == "POST":
                if raw_body is not None:
                    resp = requests.post(endpoint, headers=headers, data=raw_body, timeout=20)
                else:
                    resp = requests.post(endpoint, headers=headers, json=json_body, timeout=20)
            else:
                resp = requests.get(endpoint, headers=headers, timeout=20)
        except Exception as exc:
            safe_error = str(exc)[:300]
            self._mark_last_rest_result(endpoint, None, safe_error)
            logger.warning(
                "BMW provider request failed %s endpoint=%s error_excerpt=%s",
                bmw_log_context(
                    operation="bmw_refresh_once",
                    bmw_operation="provider_request",
                    phase=stage,
                    method=method.upper(),
                    optional=optional,
                ),
                endpoint,
                safe_error,
            )
            aggregate_payload[f"{method.upper()} {path}"] = {"_error": {"status": None, "excerpt": safe_error}}
            aggregate_payload["sequence"].append(
                {"stage": stage, "method": method.upper(), "endpoint": path, "ok": False, "status": None, "error_excerpt": safe_error, "optional": optional}
            )
            if optional:
                return None
            raise
        self._mark_last_rest_result(endpoint, resp.status_code)
        if "/telematicData" in path:
            self.status.last_telematic_status_code = resp.status_code
        if resp.status_code >= 400:
            safe_error = self._safe_error_excerpt(resp)
            self._mark_last_rest_result(endpoint, resp.status_code, safe_error)
            logger.warning(
                "BMW provider request non-2xx %s endpoint=%s status=%s optional=%s",
                bmw_log_context(
                    operation="bmw_refresh_once",
                    bmw_operation="provider_request",
                    phase=stage,
                    method=method.upper(),
                ),
                endpoint,
                resp.status_code,
                optional,
            )
            aggregate_payload[f"{method.upper()} {path}"] = {"_error": {"status": resp.status_code, "excerpt": safe_error}}
            aggregate_payload["sequence"].append(
                {
                    "stage": stage,
                    "method": method.upper(),
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
        aggregate_payload[f"{method.upper()} {path}"] = node
        aggregate_payload["sequence"].append({"stage": stage, "method": method.upper(), "endpoint": path, "ok": True, "optional": optional})
        capture_paths.append(str(self.storage.store_raw_capture(capture_endpoint_path or path, node, status_code=resp.status_code)))
        return node

    def _phase1_candidate_descriptors(self) -> list[str]:
        profile = dict(PHASE1_CONTAINER_DEFINITION)
        candidate_descriptor_ids = profile.get("candidate_phase1_descriptors") if isinstance(profile.get("candidate_phase1_descriptors"), list) else []
        return self._build_technical_descriptors(candidate_descriptor_ids)

    def _phase1_container_create_request(self, technical_descriptors: list[str] | None = None) -> CreateContainerRequest:
        profile = dict(PHASE1_CONTAINER_DEFINITION)
        descriptors = list(technical_descriptors) if isinstance(technical_descriptors, list) else self._phase1_candidate_descriptors()
        return CreateContainerRequest(
            name=str(profile.get("name") or ""),
            purpose=str(profile.get("purpose") or ""),
            technicalDescriptors=self._build_technical_descriptors(descriptors),
        )

    def _build_technical_descriptors(self, descriptor_ids: list[Any]) -> list[str]:
        return [str(x).strip() for x in descriptor_ids if str(x).strip()]

    def _technical_descriptor_shape_summary(self, technical_descriptors: list[Any]) -> str:
        if not technical_descriptors:
            return "empty"
        summaries: list[str] = []
        for item in technical_descriptors:
            summaries.append(type(item).__name__)
        uniq = sorted(set(summaries))
        return uniq[0] if len(uniq) == 1 else f"mixed({','.join(uniq)})"

    def _load_persisted_container(self) -> tuple[str | None, list[dict[str, Any]]]:
        persisted = self.storage.load_container_state()
        container_id = str(persisted.get("active_container_id") or "").strip() if isinstance(persisted, dict) else ""
        diagnostics: list[dict[str, Any]] = []
        if isinstance(persisted, dict) and isinstance(persisted.get("containers"), list):
            diagnostics = [x for x in persisted.get("containers") if isinstance(x, dict)]
        return (container_id or None), diagnostics

    def _load_descriptor_validation_state(self) -> dict[str, Any]:
        state = self.storage.load_descriptor_validation_state()
        return state if isinstance(state, dict) else {}

    def _persist_descriptor_validation_state(
        self,
        *,
        accepted_descriptors: list[str],
        rejected_descriptors: dict[str, Any],
        probe_results: list[dict[str, Any]],
    ) -> None:
        payload = {
            "accepted_descriptors": list(accepted_descriptors),
            "rejected_descriptors": rejected_descriptors,
            "probe_results": probe_results,
            "last_tested_at": utcnow().replace(microsecond=0).isoformat(),
        }
        self.storage.save_descriptor_validation_state(payload)

    def _persist_container_state(
        self,
        *,
        active_container_id: str | None,
        diagnostics: list[dict[str, Any]],
        source: str,
        accepted_descriptors: list[str] | None = None,
    ) -> None:
        descriptor_state = self._load_descriptor_validation_state()
        persisted_accepted = descriptor_state.get("accepted_descriptors") if isinstance(descriptor_state.get("accepted_descriptors"), list) else []
        final_accepted = self._build_technical_descriptors(accepted_descriptors) if isinstance(accepted_descriptors, list) else self._build_technical_descriptors(persisted_accepted)
        self.storage.save_container_state(
            {
                "active_container_id": active_container_id,
                "containers": diagnostics,
                "source": source,
                "updated_at": utcnow().replace(microsecond=0).isoformat(),
                "accepted_descriptors": final_accepted,
                "descriptor_profile": self._phase1_container_create_request(final_accepted).to_json_body(),
            }
        )

    def _critical_missing_descriptors(self, accepted_descriptors: list[str]) -> list[str]:
        accepted = set(self._build_technical_descriptors(accepted_descriptors))
        return [d for d in CRITICAL_EV_DESCRIPTORS if d not in accepted]

    def _build_bmw_ev_diagnostics(
        self,
        *,
        aggregate_payload: dict[str, Any],
        target_vehicle_id: str | None,
        accepted_before: list[str],
        accepted_after: list[str],
        missing_before: list[str],
        missing_after: list[str],
        reprobe_triggered: bool,
        reprobe_reason: str,
        active_container_id: str | None,
        refresh_attempted: bool,
        refresh_succeeded: bool,
        fallback_to_cached_state: bool,
        fallback_reason: str | None,
    ) -> dict[str, Any]:
        telematic = extract_vehicle_telematic_payload(aggregate_payload, target_vehicle_id)
        per_descriptor = build_critical_ev_field_evidence(telematic, accepted_after)
        fields: dict[str, Any] = {}
        for descriptor, field_name in CRITICAL_EV_DESCRIPTOR_FIELDS.items():
            alias = f"{field_name}_evidence"
            fields[alias] = dict(per_descriptor.get(descriptor) or {})
            fields[alias]["descriptor"] = descriptor

        vehicle_state = self.vehicles.get(str(target_vehicle_id or ""))
        normalized = {
            "soc_pct": vehicle_state.soc_pct if vehicle_state else None,
            "range_km": vehicle_state.range_km if vehicle_state else None,
            "is_charging": vehicle_state.is_charging if vehicle_state else None,
            "time_to_full_min": vehicle_state.time_to_full_min if vehicle_state else None,
            "charge_power_kw": vehicle_state.charge_power_kw if vehicle_state else None,
            "is_plugged": vehicle_state.is_plugged if vehicle_state else None,
            "expected_full_charge_ts": vehicle_state.expected_full_charge_ts.replace(microsecond=0).isoformat() if vehicle_state and vehicle_state.expected_full_charge_ts else None,
        }
        return {
            "critical_candidate_descriptors": list(CRITICAL_EV_DESCRIPTORS),
            "accepted_descriptors": list(accepted_after),
            "missing_critical_descriptors": list(missing_after),
            "accepted_before": list(accepted_before),
            "accepted_after": list(accepted_after),
            "missing_before": list(missing_before),
            "missing_after": list(missing_after),
            "reprobe_triggered": reprobe_triggered,
            "reprobe_reason": reprobe_reason,
            "active_container_id": active_container_id,
            "refresh_attempted": refresh_attempted,
            "refresh_succeeded": refresh_succeeded,
            "fallback_to_cached_state": fallback_to_cached_state,
            "fallback_reason": fallback_reason,
            "raw_field_evidence": fields,
            "normalized_outputs": normalized,
            "telematic_payload_seen": bool(telematic),
            "telematic_payload_keys_preview": _safe_preview(sorted(telematic.keys())[:10]) if telematic else [],
        }

    def _execute_container_create_attempt(
        self,
        *,
        base: str,
        headers: dict[str, str],
        aggregate_payload: dict[str, Any],
        capture_paths: list[str],
        technical_descriptors: list[str],
        mode: str,
        capture_endpoint_path: str,
    ) -> tuple[str | None, dict[str, Any], Any]:
        create_request_model = self._phase1_container_create_request(technical_descriptors)
        payload = create_request_model.to_json_body()
        serialized_body = create_request_model.to_json_string()
        create_headers = dict(headers)
        create_headers["Content-Type"] = "application/json"
        endpoint_path = "/customers/containers"
        request_field_names = sorted(payload.keys())
        descriptor_count = len(payload.get("technicalDescriptors", []))
        descriptor_sample = list(payload.get("technicalDescriptors", []))[:3]
        descriptor_shape_summary = self._technical_descriptor_shape_summary(list(payload.get("technicalDescriptors", [])))
        serialized_body_sample = serialized_body[:600]

        response = self._request_json(
            method="POST",
            base=base,
            path=endpoint_path,
            headers=create_headers,
            aggregate_payload=aggregate_payload,
            capture_paths=capture_paths,
            stage="discover",
            optional=True,
            capture_endpoint_path=capture_endpoint_path,
            raw_body=serialized_body,
        )
        error_node = aggregate_payload.get(f"POST {endpoint_path}") if isinstance(aggregate_payload.get(f"POST {endpoint_path}"), dict) else {}
        status = (error_node.get("_error") or {}).get("status")
        if status is None:
            status = self.status.last_rest_status_code
        response_excerpt = (error_node.get("_error") or {}).get("excerpt")

        create_request_diag = {
            "mode": mode,
            "endpoint": f"{base}{endpoint_path}",
            "method": "POST",
            "content_type": create_headers.get("Content-Type"),
            "serialized_body": serialized_body,
            "top_level_field_names": request_field_names,
            "descriptor_count": descriptor_count,
            "descriptor_item_type_summary": descriptor_shape_summary,
            "descriptor_sample": descriptor_sample,
            "technical_descriptors": list(payload.get("technicalDescriptors", [])),
            "status": status,
            "response_excerpt": response_excerpt,
        }
        capture_payload = {
            "attempt_mode": mode,
            "endpoint": f"{base}{endpoint_path}",
            "method": "POST",
            "content_type": create_headers.get("Content-Type"),
            "serialized_body": serialized_body,
            "top_level_field_names": request_field_names,
            "descriptor_count": descriptor_count,
            "descriptor_item_type_summary": descriptor_shape_summary,
            "descriptor_sample": descriptor_sample,
            "technical_descriptors": list(payload.get("technicalDescriptors", [])),
            "status": status,
            "response_excerpt": response_excerpt,
            "response_payload": response,
        }
        capture_paths.append(str(self.storage.store_raw_capture(f"/customers/containers_create_attempt_{mode}", capture_payload, status_code=status)))

        container_id = None
        if isinstance(response, dict):
            container_id = str(response.get("containerId") or response.get("id") or response.get("identifier") or "").strip() or None
        diag = {
            "container_id": container_id,
            "state": str(response.get("state") or "") if isinstance(response, dict) else "",
            "name": (response.get("name") if isinstance(response, dict) else None) or payload.get("name"),
            "purpose": (response.get("purpose") if isinstance(response, dict) else None) or payload.get("purpose"),
            "created_at": (response.get("createdAt") if isinstance(response, dict) else None) or utcnow().replace(microsecond=0).isoformat(),
            "updated_at": response.get("updatedAt") if isinstance(response, dict) else None,
            "descriptor_profile": payload,
            "create_request": create_request_diag,
            "raw": response if isinstance(response, dict) else None,
        }
        return container_id, diag, response

    def _delete_probe_container(self, *, base: str, headers: dict[str, str], container_id: str) -> dict[str, Any]:
        endpoint_path = f"/customers/containers/{container_id}"
        endpoint = f"{base}{endpoint_path}"
        try:
            resp = requests.delete(endpoint, headers=headers, timeout=20)
            ok = resp.status_code < 400
            excerpt = self._safe_error_excerpt(resp) if not ok else ""
            return {"endpoint": endpoint, "status": resp.status_code, "ok": ok, "response_excerpt": excerpt}
        except Exception as exc:
            return {"endpoint": endpoint, "status": None, "ok": False, "response_excerpt": str(exc)[:300]}

    def _probe_descriptors(
        self,
        *,
        base: str,
        headers: dict[str, str],
        aggregate_payload: dict[str, Any],
        capture_paths: list[str],
        candidates: list[str],
    ) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
        accepted: list[str] = []
        rejected: dict[str, Any] = {}
        probe_results: list[dict[str, Any]] = []
        for descriptor in candidates:
            container_id, create_diag, _ = self._execute_container_create_attempt(
                base=base,
                headers=headers,
                aggregate_payload=aggregate_payload,
                capture_paths=capture_paths,
                technical_descriptors=[descriptor],
                mode="probe",
                capture_endpoint_path="/customers/containers_probe_create",
            )
            status = create_diag.get("create_request", {}).get("status")
            result = {
                "tested_descriptors": [descriptor],
                "status": status,
                "success": bool(container_id),
                "container_id": container_id,
                "response_excerpt": create_diag.get("create_request", {}).get("response_excerpt"),
            }
            if container_id:
                accepted.append(descriptor)
                deletion = self._delete_probe_container(base=base, headers=headers, container_id=container_id)
                result["probe_container_deleted"] = deletion
            else:
                rejected[descriptor] = {
                    "status": status,
                    "response_excerpt": create_diag.get("create_request", {}).get("response_excerpt"),
                }
            probe_results.append(result)
        return accepted, rejected, probe_results

    def _create_container_if_needed(
        self,
        *,
        base: str,
        headers: dict[str, str],
        aggregate_payload: dict[str, Any],
        capture_paths: list[str],
        force_reprobe: bool = False,
    ) -> tuple[str | None, dict[str, Any] | None]:
        self.status.container_auto_create_attempted = True

        descriptor_state = self._load_descriptor_validation_state()
        persisted_accepted = self._build_technical_descriptors(
            descriptor_state.get("accepted_descriptors") if isinstance(descriptor_state.get("accepted_descriptors"), list) else []
        )
        persisted_rejected_raw = descriptor_state.get("rejected_descriptors") if isinstance(descriptor_state.get("rejected_descriptors"), dict) else {}
        persisted_rejected = {str(k): v for k, v in persisted_rejected_raw.items()}
        candidate_descriptors = self._phase1_candidate_descriptors()
        next_probe_descriptor = "vehicle.drivetrain.electricEngine.charging.level"
        should_probe_next = bool(
            persisted_accepted
            and next_probe_descriptor not in persisted_accepted
            and (
                next_probe_descriptor not in persisted_rejected
                or force_reprobe
            )
        )
        if should_probe_next:
            accepted_next, rejected_next, probe_results_next = self._probe_descriptors(
                base=base,
                headers=headers,
                aggregate_payload=aggregate_payload,
                capture_paths=capture_paths,
                candidates=[next_probe_descriptor],
            )
            if accepted_next:
                persisted_accepted = self._build_technical_descriptors(persisted_accepted + accepted_next)
            persisted_rejected.update(rejected_next)
            existing_probe_results = descriptor_state.get("probe_results") if isinstance(descriptor_state.get("probe_results"), list) else []
            descriptor_state["probe_results"] = [*existing_probe_results, *probe_results_next]
            descriptor_state["accepted_descriptors"] = list(persisted_accepted)
            descriptor_state["rejected_descriptors"] = dict(persisted_rejected)
            descriptor_state["last_forced_reprobe"] = bool(force_reprobe)
            self._persist_descriptor_validation_state(
                accepted_descriptors=persisted_accepted,
                rejected_descriptors=persisted_rejected,
                probe_results=descriptor_state.get("probe_results") if isinstance(descriptor_state.get("probe_results"), list) else [],
            )

        candidate_descriptors = self._phase1_candidate_descriptors()
        initial_descriptors = persisted_accepted or candidate_descriptors

        container_id, diag, _ = self._execute_container_create_attempt(
            base=base,
            headers=headers,
            aggregate_payload=aggregate_payload,
            capture_paths=capture_paths,
            technical_descriptors=initial_descriptors,
            mode="production_initial",
            capture_endpoint_path="/customers/containers_create",
        )

        if container_id:
            self.status.container_auto_create_succeeded = True
            self._persist_descriptor_validation_state(
                accepted_descriptors=initial_descriptors,
                rejected_descriptors=persisted_rejected,
                probe_results=descriptor_state.get("probe_results") if isinstance(descriptor_state.get("probe_results"), list) else [],
            )
            return container_id, diag

        status = diag.get("create_request", {}).get("status")
        if status != 400:
            return None, diag

        accepted, rejected, probe_results = self._probe_descriptors(
            base=base,
            headers=headers,
            aggregate_payload=aggregate_payload,
            capture_paths=capture_paths,
            candidates=candidate_descriptors,
        )
        self._persist_descriptor_validation_state(
            accepted_descriptors=accepted,
            rejected_descriptors=rejected,
            probe_results=probe_results,
        )
        diag["bootstrap_probe"] = {
            "triggered": True,
            "accepted_descriptors": accepted,
            "rejected_descriptors": rejected,
            "probe_results": probe_results,
        }
        if not accepted:
            return None, diag

        final_container_id, final_diag, _ = self._execute_container_create_attempt(
            base=base,
            headers=headers,
            aggregate_payload=aggregate_payload,
            capture_paths=capture_paths,
            technical_descriptors=accepted,
            mode="production_final",
            capture_endpoint_path="/customers/containers_create_final",
        )
        final_diag["bootstrap_probe"] = diag.get("bootstrap_probe")
        if final_container_id:
            self.status.container_auto_create_succeeded = True
        return final_container_id, final_diag

    def refresh_once(self, *, force_reprobe: bool = False) -> dict[str, Any]:
        logger.info(
            "BMW provider refresh start %s",
            bmw_log_context(operation="bmw_refresh_once", bmw_operation="provider_refresh", phase="start", force_reprobe=bool(force_reprobe)),
        )
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
        self.status.container_auto_create_attempted = False
        self.status.container_auto_create_succeeded = False
        self.status.bmw_ev_diagnostics = {}
        self.status.force_reprobe_diagnostics = {
            "force_mode": bool(force_reprobe),
            "target_descriptor": "vehicle.drivetrain.electricEngine.charging.level",
            "descriptor_reprobe_attempted": False,
            "descriptor_accepted": False,
            "descriptor_rejected": False,
            "production_container_rebuilt": False,
            "old_active_container_id": None,
            "new_active_container_id": None,
            "old_container_cleanup": None,
        }

        try:
            discovery_op = self.rest_operations()[0]
            mappings_payload = self._request_json(
                method="GET",
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
                logger.warning(
                    "BMW provider refresh no vehicles %s",
                    bmw_log_context(operation="bmw_refresh_once", bmw_operation="provider_refresh", phase="discover", reason="no_vehicles"),
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
                    "bmw_ev_diagnostics": dict(self.status.bmw_ev_diagnostics),
                }

            target_vehicle_id = self._select_active_vehicle(discovered_ids, mapping_diagnostics)
            self.status.active_vehicle_id = target_vehicle_id

            basic_path = f"/customers/vehicles/{target_vehicle_id}/basicData"
            self._request_json(
                method="GET",
                base=base,
                path=basic_path,
                headers=headers,
                aggregate_payload=aggregate_payload,
                capture_paths=capture_paths,
                stage="vehicle",
                optional=False,
            )

            containers_payload = self._request_json(
                method="GET",
                base=base,
                path="/customers/containers",
                headers=headers,
                aggregate_payload=aggregate_payload,
                capture_paths=capture_paths,
                stage="discover",
                optional=True,
            )
            container_ids, container_diags = self._discover_containers(containers_payload)

            persisted_active_id, persisted_diags = self._load_persisted_container()
            if persisted_active_id and persisted_active_id not in container_ids:
                container_ids.append(persisted_active_id)
            for diag in persisted_diags:
                if str(diag.get("container_id") or "") and all(str(x.get("container_id") or "") != str(diag.get("container_id") or "") for x in container_diags):
                    container_diags.append(diag)

            descriptor_state_before = self._load_descriptor_validation_state()
            accepted_before = self._build_technical_descriptors(
                descriptor_state_before.get("accepted_descriptors") if isinstance(descriptor_state_before.get("accepted_descriptors"), list) else []
            )
            missing_before = self._critical_missing_descriptors(accepted_before)

            active_container_id = self._select_active_container(container_diags)
            previous_active_container_id = active_container_id
            self.status.force_reprobe_diagnostics["old_active_container_id"] = previous_active_container_id
            reprobe_triggered = False
            reprobe_reason = "none"
            should_rebuild = bool(force_reprobe or not active_container_id or bool(missing_before))
            if force_reprobe:
                reprobe_reason = "force_reprobe"
            elif not active_container_id:
                reprobe_reason = "missing_active_container"
            elif missing_before:
                reprobe_reason = "missing_critical_descriptors"

            if should_rebuild:
                reprobe_triggered = True
                created_container_id, created_diag = self._create_container_if_needed(
                    base=base,
                    headers=headers,
                    aggregate_payload=aggregate_payload,
                    capture_paths=capture_paths,
                    force_reprobe=force_reprobe or bool(missing_before),
                )
                if created_diag:
                    container_diags = [
                        x for x in container_diags
                        if str(x.get("container_id") or "") != str(created_diag.get("container_id") or "")
                    ]
                    container_diags.append(created_diag)
                if created_container_id and created_container_id not in container_ids:
                    container_ids.append(created_container_id)
                if created_container_id:
                    active_container_id = created_container_id
                    rebuilt = bool(force_reprobe and previous_active_container_id and previous_active_container_id != created_container_id)
                    self.status.force_reprobe_diagnostics["production_container_rebuilt"] = rebuilt
                    if rebuilt:
                        self.status.force_reprobe_diagnostics["old_container_cleanup"] = self._delete_probe_container(
                            base=base,
                            headers=headers,
                            container_id=str(previous_active_container_id),
                        )
                elif force_reprobe:
                    active_container_id = previous_active_container_id

            self.status.discovered_container_ids = list(container_ids)
            self.status.container_diagnostics = list(container_diags)
            self.status.active_container_id = active_container_id
            self.status.force_reprobe_diagnostics["new_active_container_id"] = active_container_id
            self._persist_container_state(active_container_id=active_container_id, diagnostics=container_diags, source="refresh")

            if active_container_id:
                telematic_query = urlencode({"containerId": active_container_id})
                telematic_path = f"/customers/vehicles/{target_vehicle_id}/telematicData?{telematic_query}"
                self.status.last_telematic_url = f"{base}{telematic_path}"
                self._request_json(
                    method="GET",
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
                        method="GET",
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
            telematic_key = f"GET /customers/vehicles/{target_vehicle_id}/telematicData?containerId={active_container_id}" if active_container_id else None
            telematic_node = aggregate_payload.get(telematic_key) if telematic_key else None
            self.status.vehicle_data_mode = "live_telematics" if isinstance(telematic_node, dict) and "_error" not in telematic_node else "static_only"
            if self.status.vehicle_data_mode != "live_telematics":
                self.status.data_status = "partial"
            descriptor_state_after = self._load_descriptor_validation_state()
            accepted_after = self._build_technical_descriptors(
                descriptor_state_after.get("accepted_descriptors") if isinstance(descriptor_state_after.get("accepted_descriptors"), list) else []
            )
            missing_after = self._critical_missing_descriptors(accepted_after)
            probe_results = descriptor_state_after.get("probe_results") if isinstance(descriptor_state_after.get("probe_results"), list) else []
            target_descriptor = str(self.status.force_reprobe_diagnostics.get("target_descriptor") or "")
            target_probe_entries = [
                x for x in probe_results
                if isinstance(x, dict) and x.get("tested_descriptors") == [target_descriptor]
            ]
            if target_probe_entries:
                latest_target_probe = target_probe_entries[-1]
                self.status.force_reprobe_diagnostics["descriptor_reprobe_attempted"] = bool(
                    force_reprobe or len(target_probe_entries) >= 1
                )
                self.status.force_reprobe_diagnostics["descriptor_accepted"] = bool(latest_target_probe.get("success"))
                self.status.force_reprobe_diagnostics["descriptor_rejected"] = not bool(latest_target_probe.get("success"))
            elif force_reprobe:
                self.status.force_reprobe_diagnostics["descriptor_reprobe_attempted"] = True
            self.status.force_reprobe_diagnostics["accepted_before"] = list(accepted_before)
            self.status.force_reprobe_diagnostics["accepted_after"] = list(accepted_after)
            self.status.force_reprobe_diagnostics["missing_before"] = list(missing_before)
            self.status.force_reprobe_diagnostics["missing_after"] = list(missing_after)
            self.status.force_reprobe_diagnostics["reprobe_triggered"] = reprobe_triggered
            self.status.force_reprobe_diagnostics["reprobe_reason"] = reprobe_reason

            self.status.bmw_ev_diagnostics = self._build_bmw_ev_diagnostics(
                aggregate_payload=aggregate_payload,
                target_vehicle_id=target_vehicle_id,
                accepted_before=accepted_before,
                accepted_after=accepted_after,
                missing_before=missing_before,
                missing_after=missing_after,
                reprobe_triggered=reprobe_triggered,
                reprobe_reason=reprobe_reason,
                active_container_id=active_container_id,
                refresh_attempted=True,
                refresh_succeeded=bool(self.vehicles),
                fallback_to_cached_state=False,
                fallback_reason=None,
            )

            result = {
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
                "container_auto_create_attempted": self.status.container_auto_create_attempted,
                "container_auto_create_succeeded": self.status.container_auto_create_succeeded,
                "force_reprobe_diagnostics": dict(self.status.force_reprobe_diagnostics),
                "bmw_ev_diagnostics": dict(self.status.bmw_ev_diagnostics),
            }
            logger.info(
                "BMW provider refresh complete %s",
                bmw_log_context(
                    operation="bmw_refresh_once",
                    bmw_operation="provider_refresh",
                    phase="complete",
                    ok=bool(result.get("ok")),
                    vehicles=len(self.vehicles),
                    active_vehicle_id=self.status.active_vehicle_id,
                ),
            )
            return result
        except Exception as exc:
            self.status.last_error = str(exc)
            self.status.provider_status = "degraded"
            self.status.capture_files_written = list(capture_paths)
            logger.warning(
                "BMW provider refresh failed %s error=%s",
                bmw_log_context(operation="bmw_refresh_once", bmw_operation="provider_refresh", phase="exception"),
                exc,
            )
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
                "container_auto_create_attempted": self.status.container_auto_create_attempted,
                "container_auto_create_succeeded": self.status.container_auto_create_succeeded,
                "force_reprobe_diagnostics": dict(self.status.force_reprobe_diagnostics),
                "bmw_ev_diagnostics": dict(self.status.bmw_ev_diagnostics),
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
            if not rows and any(payload.get(k) is not None for k in ("containerId", "id", "identifier")):
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
                    "raw": row,
                }
            )
        return seen, diagnostics

    def _select_active_container(self, diagnostics: list[dict[str, Any]]) -> str | None:
        if not diagnostics:
            return None
        active = [d for d in diagnostics if str(d.get("state") or "").upper() in {"ACTIVE", "ENABLED", "READY"}]
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

    def manual_refresh(self, *, force_reprobe: bool = False) -> dict[str, Any]:
        return self.refresh_once(force_reprobe=force_reprobe)
