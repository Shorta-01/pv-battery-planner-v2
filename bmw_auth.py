from __future__ import annotations

import datetime as dt
import base64
import hashlib
import json
import logging
import secrets
from pathlib import Path
from typing import Any

import requests

from bmw_models import BmwTokenData

logger = logging.getLogger(__name__)


class BmwAuthClient:
    DEVICE_FLOW_SCOPE = "authenticate_user openid cardata:streaming:read cardata:api:read"

    def __init__(self, client_id: str, token_cache_path: str, auth_base_url: str = "https://customer.bmwgroup.com/gcdm/oauth") -> None:
        self.client_id = client_id
        self.token_cache_path = Path(token_cache_path)
        self.device_flow_session_path = self.token_cache_path.with_name(f"{self.token_cache_path.stem}_device_flow_session.json")
        self.auth_base_url = auth_base_url.rstrip("/")
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)

    def device_flow_start_url(self) -> str:
        return f"{self.auth_base_url}/device/code"

    def device_flow_poll_url(self) -> str:
        return f"{self.auth_base_url}/token"

    def load_token(self) -> BmwTokenData:
        if not self.token_cache_path.exists():
            return BmwTokenData()
        try:
            payload = json.loads(self.token_cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return BmwTokenData()
        return BmwTokenData.from_dict(payload if isinstance(payload, dict) else {})

    def save_token(self, token: BmwTokenData) -> None:
        self.token_cache_path.write_text(json.dumps(token.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _raise_http_error(self, endpoint: str, resp: requests.Response, context: str) -> None:
        body_excerpt = resp.text[:300]
        logger.error(
            "BMW auth: %s error status=%s endpoint=%s body_excerpt=%s",
            context,
            resp.status_code,
            endpoint,
            body_excerpt,
        )
        raise RuntimeError(
            f"BMW {context} failed: status={resp.status_code} endpoint={endpoint} body={body_excerpt}"
        )

    @staticmethod
    def _generate_code_verifier() -> str:
        return secrets.token_urlsafe(64)

    @staticmethod
    def _pkce_code_challenge(code_verifier: str) -> str:
        digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def _save_device_flow_session(self, payload: dict[str, Any]) -> None:
        self.device_flow_session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_device_flow_session(self) -> dict[str, Any]:
        if not self.device_flow_session_path.exists():
            return {}
        try:
            payload = json.loads(self.device_flow_session_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def start_device_flow(self, scope: str = DEVICE_FLOW_SCOPE) -> dict[str, Any]:
        url = self.device_flow_start_url()
        code_verifier = self._generate_code_verifier()
        payload = {
            "client_id": self.client_id,
            "code_challenge": self._pkce_code_challenge(code_verifier),
            "code_challenge_method": "S256",
            "scope": scope,
        }
        logger.info(
            "BMW auth: device flow start request endpoint=%s form_encoded=%s param_keys=%s",
            url,
            True,
            sorted(payload.keys()),
        )
        resp = requests.post(
            url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        logger.info("BMW auth: device flow start response status=%s endpoint=%s", resp.status_code, url)
        if resp.status_code >= 400:
            self._raise_http_error(url, resp, "device flow start")
        response_payload = resp.json()
        self._save_device_flow_session(
            {
                "client_id": self.client_id,
                "code_verifier": code_verifier,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "device_code": response_payload.get("device_code"),
            }
        )
        logger.info("BMW auth: device flow started")
        return response_payload

    def poll_device_token(self, device_code: str, interval_seconds: int = 5, scope: str = DEVICE_FLOW_SCOPE) -> BmwTokenData:
        session = self._load_device_flow_session()
        code_verifier = session.get("code_verifier")
        if not code_verifier:
            raise RuntimeError("BMW token poll failed: missing device flow session code_verifier")
        url = self.device_flow_poll_url()
        payload = {
            "client_id": str(session.get("client_id") or self.client_id),
            "code_verifier": code_verifier,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "scope": scope,
        }
        logger.info(
            "BMW auth: token poll request endpoint=%s form_encoded=%s param_keys=%s verifier_preview=%s",
            url,
            True,
            sorted(payload.keys()),
            f"{code_verifier[:6]}...",
        )
        resp = requests.post(
            url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        logger.info("BMW auth: token poll response status=%s endpoint=%s", resp.status_code, url)
        if resp.status_code >= 400:
            self._raise_http_error(url, resp, "token poll")
        payload = resp.json()
        expires_in = int(payload.get("expires_in", 3600))
        token = BmwTokenData(
            access_token=payload.get("access_token"),
            refresh_token=payload.get("refresh_token"),
            id_token=payload.get("id_token"),
            token_type=payload.get("token_type"),
            obtained_at=dt.datetime.now(dt.timezone.utc),
            expires_at=dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=expires_in),
        )
        self.save_token(token)
        logger.info("BMW auth: token obtained and cached")
        return token

    def refresh_if_possible(self, token: BmwTokenData) -> BmwTokenData:
        if token.is_fresh():
            return token
        if not token.refresh_token:
            return token
        url = f"{self.auth_base_url}/token"
        resp = requests.post(
            url,
            data={
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": token.refresh_token,
            },
            timeout=20,
        )
        if resp.status_code >= 400:
            logger.warning("BMW auth: refresh failed (status=%s)", resp.status_code)
            return token
        payload = resp.json()
        expires_in = int(payload.get("expires_in", 3600))
        refreshed = BmwTokenData(
            access_token=payload.get("access_token") or token.access_token,
            refresh_token=payload.get("refresh_token") or token.refresh_token,
            id_token=payload.get("id_token") or token.id_token,
            token_type=payload.get("token_type") or token.token_type,
            obtained_at=dt.datetime.now(dt.timezone.utc),
            expires_at=dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=expires_in),
        )
        self.save_token(refreshed)
        logger.info("BMW auth: token refreshed")
        return refreshed
