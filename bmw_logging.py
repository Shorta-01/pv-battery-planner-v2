from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


_BMW_REQUEST_CONTEXT: ContextVar[dict[str, str | None]] = ContextVar("bmw_request_context", default={})


def get_bmw_request_context() -> dict[str, str | None]:
    current = _BMW_REQUEST_CONTEXT.get()
    return dict(current) if isinstance(current, dict) else {}


@contextmanager
def bmw_request_context(**fields: str | None) -> Iterator[None]:
    merged = get_bmw_request_context()
    for key, value in fields.items():
        if value is None:
            continue
        merged[key] = str(value)
    token = _BMW_REQUEST_CONTEXT.set(merged)
    try:
        yield
    finally:
        _BMW_REQUEST_CONTEXT.reset(token)


def bmw_log_context(**fields: object) -> str:
    merged: dict[str, object] = {}
    merged.update(get_bmw_request_context())
    merged.update(fields)
    parts: list[str] = []
    for key, value in merged.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)
