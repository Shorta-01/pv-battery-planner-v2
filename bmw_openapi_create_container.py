"""Deprecated compatibility module.

Use ``bmw_cardata_contract`` as the single source of truth for
POST /customers/containers request schema.
"""

from bmw_cardata_contract import CreateContainerRequest, CreateContainerTechnicalDescriptor

__all__ = ["CreateContainerRequest", "CreateContainerTechnicalDescriptor"]
