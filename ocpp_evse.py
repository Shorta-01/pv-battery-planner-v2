from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import uuid
from typing import Any

from fastapi import WebSocket


class OcppEvseManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._connected = False
        self._status = "unknown"
        self._is_charging = False
        self._transaction_id: int | None = None
        self._last_seen_iso: str | None = None
        self._last_error: str | None = None
        self._auth_mode = "unknown"
        self._last_power_kw: float | None = None
        self._energy_total_kwh: float | None = None
        self._energy_session_start_kwh: float | None = None
        self._energy_session_kwh: float | None = None
        self._ws: WebSocket | None = None
        self._pending_calls: dict[str, asyncio.Future] = {}

    def _now_iso(self) -> str:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    async def _set_state(self, **kwargs: Any) -> None:
        async with self._state_lock:
            for key, value in kwargs.items():
                setattr(self, f"_{key}", value)
            self._last_seen_iso = self._now_iso()

    async def _send_json(self, payload: list[Any]) -> None:
        if self._ws is None:
            raise RuntimeError("EVSE not connected")
        async with self._send_lock:
            await self._ws.send_text(json.dumps(payload, separators=(",", ":")))

    async def send_call(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None or not self._connected:
            raise RuntimeError("EVSE not connected")
        unique_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_calls[unique_id] = fut
        await self._send_json([2, unique_id, action, payload])
        try:
            result = await asyncio.wait_for(fut, timeout=10.0)
        finally:
            self._pending_calls.pop(unique_id, None)
        return result

    def _plugged_from_status(self, status: str) -> bool:
        return status in {"Preparing", "Charging", "SuspendedEV", "SuspendedEVSE", "Finishing"}

    async def _handle_call(self, unique_id: str, action: str, payload: dict[str, Any]) -> None:
        if action == "BootNotification":
            await self._send_json([3, unique_id, {"status": "Accepted", "currentTime": self._now_iso(), "interval": 30}])
            await self._set_state(last_error=None)
            return

        if action == "Heartbeat":
            await self._send_json([3, unique_id, {"currentTime": self._now_iso()}])
            return

        if action == "StatusNotification":
            status = str(payload.get("status", "Unknown"))
            is_charging = status == "Charging"
            await self._set_state(status=status, is_charging=is_charging, last_error=None)
            await self._send_json([3, unique_id, {}])
            return

        if action == "StartTransaction":
            txid = payload.get("transactionId")
            if txid is None:
                txid = int(dt.datetime.now().timestamp())
            self._energy_session_start_kwh = None
            self._energy_session_kwh = None
            await self._set_state(transaction_id=int(txid), is_charging=True, status="Charging", last_error=None)
            await self._send_json([3, unique_id, {"idTagInfo": {"status": "Accepted"}, "transactionId": int(txid)}])
            return

        if action == "StopTransaction":
            self._energy_session_start_kwh = None
            self._energy_session_kwh = None
            await self._set_state(transaction_id=None, is_charging=False, last_error=None)
            await self._send_json([3, unique_id, {"idTagInfo": {"status": "Accepted"}}])
            return

        if action == "MeterValues":
            meter_values = payload.get("meterValue", [])
            if isinstance(meter_values, list):
                for meter_value in meter_values:
                    if not isinstance(meter_value, dict):
                        continue
                    sampled_values = meter_value.get("sampledValue", [])
                    if not isinstance(sampled_values, list):
                        continue
                    for sampled in sampled_values:
                        if not isinstance(sampled, dict):
                            continue
                        value_raw = sampled.get("value")
                        try:
                            value = float(value_raw)
                        except (TypeError, ValueError):
                            continue

                        measurand = str(sampled.get("measurand", "") or "")
                        unit = str(sampled.get("unit", "") or "")

                        if (not measurand) or measurand in {"Power.Active.Import"}:
                            if unit == "W":
                                self._last_power_kw = value / 1000.0
                            elif unit == "kW":
                                self._last_power_kw = value

                        if "Energy.Active.Import" in measurand:
                            energy_total_kwh: float | None = None
                            if unit == "Wh":
                                energy_total_kwh = value / 1000.0
                            elif unit == "kWh":
                                energy_total_kwh = value

                            if energy_total_kwh is not None:
                                self._energy_total_kwh = energy_total_kwh
                                if self._energy_session_start_kwh is None:
                                    self._energy_session_start_kwh = energy_total_kwh
                                self._energy_session_kwh = max(0.0, energy_total_kwh - self._energy_session_start_kwh)

            await self._set_state(last_error=None)
            await self._send_json([3, unique_id, {}])
            return

        if action == "Authorize":
            await self._send_json([3, unique_id, {"idTagInfo": {"status": "Accepted"}}])
            return

        await self._send_json([4, unique_id, "NotSupported", f"Unsupported action: {action}", {}])

    async def _handle_incoming_text(self, text: str) -> None:
        msg = json.loads(text)
        if not isinstance(msg, list) or len(msg) < 3:
            return
        message_type = msg[0]
        if message_type == 2 and len(msg) >= 4:
            unique_id = str(msg[1])
            action = str(msg[2])
            payload = msg[3] if isinstance(msg[3], dict) else {}
            await self._handle_call(unique_id, action, payload)
            return

        if message_type == 3 and len(msg) >= 3:
            unique_id = str(msg[1])
            fut = self._pending_calls.get(unique_id)
            if fut is not None and not fut.done():
                fut.set_result({"ok": True, "result": msg[2] if isinstance(msg[2], dict) else {}})
            return

        if message_type == 4 and len(msg) >= 5:
            unique_id = str(msg[1])
            fut = self._pending_calls.get(unique_id)
            if fut is not None and not fut.done():
                fut.set_result({"ok": False, "errorCode": msg[2], "errorDescription": msg[3], "details": msg[4]})

    def _auth_ok(self, websocket: WebSocket, *, basic_user: str, basic_pass: str) -> bool:
        auth = websocket.headers.get("authorization", "")
        if not auth.startswith("Basic "):
            return False
        token = auth.split(" ", 1)[1].strip()
        try:
            decoded = base64.b64decode(token).decode("utf-8")
        except Exception:
            return False
        if ":" not in decoded:
            return False
        user, pw = decoded.split(":", 1)
        if user != basic_user or pw != basic_pass:
            return False
        return True

    async def handle_websocket(self, websocket: WebSocket, *, enabled: bool, basic_user: str, basic_pass: str) -> None:
        if not enabled:
            await websocket.accept(subprotocol="ocpp1.6")
            self._auth_mode = "unknown"
            await self._set_state(last_error="car_charger is disabled")
            await websocket.close(code=1008, reason="car_charger is disabled")
            return

        user = (basic_user or "").strip()
        pw = basic_pass or ""

        if user == "" and pw == "":
            self._auth_mode = "none"
        elif (user == "") ^ (pw == ""):
            await websocket.accept(subprotocol="ocpp1.6")
            self._auth_mode = "unknown"
            await self._set_state(last_error="ocpp_auth_misconfigured: username/password must both be set or both empty")
            await websocket.close(code=1008, reason="ocpp_auth_misconfigured")
            return
        else:
            if not self._auth_ok(websocket, basic_user=user, basic_pass=pw):
                await websocket.accept(subprotocol="ocpp1.6")
                self._auth_mode = "basic"
                await self._set_state(last_error="ocpp_auth_failed")
                await websocket.close(code=1008, reason="ocpp_auth_failed")
                return
            self._auth_mode = "basic"

        await websocket.accept(subprotocol="ocpp1.6")
        async with self._lock:
            self._ws = websocket
            await self._set_state(connected=True, last_error=None)
            try:
                while True:
                    text = await websocket.receive_text()
                    await self._set_state(connected=True)
                    await self._handle_incoming_text(text)
            except Exception:
                await self._set_state(connected=False, transaction_id=None, is_charging=False)
                self._ws = None
                for fut in self._pending_calls.values():
                    if not fut.done():
                        fut.set_result({"ok": False, "error": "disconnected"})
                self._pending_calls.clear()

    async def remote_resume(self, connector_id: int = 1, id_tag: str = "LOCAL") -> dict[str, Any]:
        if not self._connected or self._ws is None:
            return {"ok": False, "error": "evse_not_connected"}
        if not self._plugged_from_status(self._status):
            return {"ok": False, "error": "car_not_plugged", "status": self._status}
        try:
            return await self.send_call("RemoteStartTransaction", {"connectorId": int(connector_id), "idTag": str(id_tag)})
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def remote_stop(self) -> dict[str, Any]:
        if not self._connected or self._ws is None:
            return {"ok": False, "error": "evse_not_connected"}
        if not self._plugged_from_status(self._status):
            return {"ok": False, "error": "car_not_plugged", "status": self._status}
        if not self._is_charging:
            return {"ok": False, "error": "not_charging", "status": self._status}

        if self._transaction_id is not None:
            try:
                return await self.send_call("RemoteStopTransaction", {"transactionId": int(self._transaction_id)})
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        try:
            res1 = await self.send_call("ChangeAvailability", {"connectorId": 1, "type": "Inoperative"})
            res2 = await self.send_call("ChangeAvailability", {"connectorId": 1, "type": "Operative"})
            return {"ok": bool(res1.get("ok") and res2.get("ok")), "fallback": "change_availability", "step1": res1, "step2": res2}
        except Exception as exc:
            return {"ok": False, "error": f"no_transaction_id_and_fallback_failed: {exc}"}

    def status_dict(self) -> dict[str, Any]:
        status = self._status if self._connected else "disconnected"
        plugged = self._plugged_from_status(status)
        return {
            "connected": bool(self._connected),
            "status": status,
            "is_plugged": bool(plugged),
            "is_charging": bool(self._is_charging or status == "Charging"),
            "last_seen_iso": self._last_seen_iso,
            "transaction_id": self._transaction_id,
            "auth_mode": self._auth_mode,
            "last_error": self._last_error,
            "power_kw": self._last_power_kw,
            "energy_total_kwh": self._energy_total_kwh,
            "energy_session_kwh": self._energy_session_kwh,
        }
