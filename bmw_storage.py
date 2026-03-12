from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from threading import RLock
from typing import Any

from bmw_models import NormalizedVehicleState, RawEventRecord


class BmwStorage:
    def __init__(self, raw_event_store_path: str, vehicle_state_store_path: str) -> None:
        self.raw_path = Path(raw_event_store_path)
        self.state_path = Path(vehicle_state_store_path)
        self._lock = RLock()
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def append_raw_event(self, rec: RawEventRecord) -> None:
        with self._lock:
            with self.raw_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    def save_vehicle_states(self, states: dict[str, NormalizedVehicleState]) -> None:
        with self._lock:
            payload = {vid: st.to_dict() for vid, st in states.items()}
            self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_vehicle_states(self) -> dict[str, NormalizedVehicleState]:
        with self._lock:
            if not self.state_path.exists():
                return {}
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
            if not isinstance(payload, dict):
                return {}
            out: dict[str, NormalizedVehicleState] = {}
            for vid, row in payload.items():
                if isinstance(row, dict):
                    out[str(vid)] = NormalizedVehicleState.from_dict(row)
            return out



    def store_raw_capture(self, endpoint_path: str, payload: dict[str, Any]) -> Path:
        safe_endpoint = endpoint_path.strip().strip("/").replace("/", "_") or "root"
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_name = f"bmw_cardata_live_{safe_endpoint}_{ts}.json"
        out_path = self.raw_path.parent / "bmw_payload_captures" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wrapped = {
            "captured_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "endpoint": endpoint_path,
            "payload": payload,
        }
        with self._lock:
            out_path.write_text(json.dumps(wrapped, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def list_raw_captures(self, limit: int = 20) -> list[str]:
        capture_dir = self.raw_path.parent / "bmw_payload_captures"
        if not capture_dir.exists():
            return []
        items = sorted(capture_dir.glob("bmw_cardata_live_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(path) for path in items[: max(1, limit)]]

    def load_recent_raw_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if not self.raw_path.exists():
                return []
            lines = self.raw_path.read_text(encoding="utf-8").splitlines()[-max(1, limit):]
            out: list[dict[str, Any]] = []
            for line in lines:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        out.append(row)
                except json.JSONDecodeError:
                    continue
            return out
