"""Imported-mesh 3D CFD tier (snappyHexMesh around the uploaded surface)."""

from __future__ import annotations

from backend.simulation.cfd3d.models import LEVELS, ImportedCFDLevel
from backend.simulation.cfd3d.runner import (
    ImportedCFDProvenance,
    ImportedCFDResult,
    run_imported_cfd,
    solver_available,
)

__all__ = [
    "ImportedCFDLevel",
    "LEVELS",
    "ImportedCFDProvenance",
    "ImportedCFDResult",
    "solver_available",
    "run_imported_cfd",
]
