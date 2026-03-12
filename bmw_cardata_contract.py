from __future__ import annotations

"""Contract models for BMW CarData container creation.

Source of truth: BMW CarData API specification
https://bmw-cardata.bmwgroup.com/customer/public/api-specification

Endpoint: POST /customers/containers
Content-Type: application/json
X-Version: v1
"""

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CreateContainerRequest:
    """BMW OpenAPI `CreateContainer` request body for POST /customers/containers."""

    name: str
    purpose: str
    technicalDescriptors: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.purpose, str):
            raise TypeError("CreateContainerRequest.name and .purpose must be strings")
        if not isinstance(self.technicalDescriptors, list):
            raise TypeError("CreateContainerRequest.technicalDescriptors must be a list[str]")
        if any(not isinstance(descriptor, str) for descriptor in self.technicalDescriptors):
            raise TypeError("CreateContainerRequest.technicalDescriptors must contain only strings")

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "technicalDescriptors": list(self.technicalDescriptors),
        }

    def to_json_body(self) -> dict[str, Any]:
        """Serialize through JSON encoder to mirror request-on-the-wire shape."""
        return json.loads(self.to_json_string())

    def to_json_string(self) -> str:
        return json.dumps(self.to_wire(), ensure_ascii=False, separators=(",", ":"))


# Backward-compatible name retained for older imports.
BmwCreateContainerRequest = CreateContainerRequest
