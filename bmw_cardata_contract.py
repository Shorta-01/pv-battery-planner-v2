from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BmwTechnicalDescriptor:
    """Local typed equivalent of BMW/OpenAPI `TechnicalDescriptor` request schema."""

    id: str

    def to_wire(self) -> dict[str, str]:
        return {"id": self.id}


@dataclass(frozen=True)
class BmwCreateContainerRequest:
    """Local typed equivalent of BMW/OpenAPI `CreateContainerRequest` schema."""

    name: str
    purpose: str
    technical_descriptors: list[BmwTechnicalDescriptor]

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "technicalDescriptors": [descriptor.to_wire() for descriptor in self.technical_descriptors],
        }

    def to_json_body(self) -> dict[str, Any]:
        """Serialize through JSON encoder to mirror request-on-the-wire shape."""
        return json.loads(json.dumps(self.to_wire(), ensure_ascii=False))
