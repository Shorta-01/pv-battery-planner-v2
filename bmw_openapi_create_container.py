from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CreateContainerTechnicalDescriptor:
    """Faithful typed reproduction of BMW OpenAPI `TechnicalDescriptors` item."""

    technicalDescriptorId: str

    def to_wire(self) -> dict[str, str]:
        return {"technicalDescriptorId": self.technicalDescriptorId}


@dataclass(frozen=True)
class CreateContainerRequest:
    """Faithful typed reproduction of BMW OpenAPI `CreateContainer` request body."""

    name: str
    purpose: str
    technicalDescriptors: list[CreateContainerTechnicalDescriptor]

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "technicalDescriptors": [descriptor.to_wire() for descriptor in self.technicalDescriptors],
        }

    def to_json_body(self) -> dict[str, Any]:
        return json.loads(self.to_json_string())

    def to_json_string(self) -> str:
        return json.dumps(self.to_wire(), ensure_ascii=False, separators=(",", ":"))
