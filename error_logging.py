from __future__ import annotations

import datetime as dt
import hashlib
import json
import traceback

MAX_ERROR_BODY_CHARS = 200_000


ERROR_TYPES = {
    "exception",
    "http_error",
    "network",
    "validation",
    "external_service",
    "ui_state",
    "unknown",
}


def iso_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def trim(s: str, max_chars: int = MAX_ERROR_BODY_CHARS) -> str:
    raw = str(s or "")
    if max_chars <= 0:
        return ""
    if len(raw) <= max_chars:
        return raw
    suffix = f"\n\n[truncated {len(raw) - max_chars} chars]"
    keep = max(0, max_chars - len(suffix))
    return raw[:keep] + suffix


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def compute_dedupe_key(*, source: str, error_type: str, where: str, title: str, body: str) -> str:
    body_head = str(body or "")[:10_000]
    normalized = "\n".join(
        [
            _normalize_text(source),
            _normalize_text(error_type),
            _normalize_text(where),
            _normalize_text(title),
            _normalize_text(body_head),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def format_exception_body(*, title: str, where: str, exc: BaseException, extra: dict | None = None) -> str:
    pieces = [
        f"Title: {title}",
        f"Where: {where}",
        f"Exception: {type(exc).__name__}: {exc}",
        "",
        "Traceback:",
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
    ]
    if isinstance(extra, dict) and extra:
        try:
            extra_text = json.dumps(extra, indent=2, sort_keys=True, default=str)
        except Exception:
            extra_text = str(extra)
        pieces.extend(["", "Context:", extra_text])
    return trim("\n".join(pieces))


def classify_exception(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "validation" in name or "422" in message:
        return "validation"
    if any(term in message for term in ("timeout", "dns", "connection", "refused", "unreachable")):
        return "network"
    if any(term in message for term in ("duplicatewidgetid", "streamlit", "session state", "widget")):
        return "ui_state"
    if "http" in name:
        return "http_error"
    if "externalservice" in name:
        return "external_service"
    return "exception"
