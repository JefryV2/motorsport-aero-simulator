"""Models and constants for the imported-mesh 3D CFD tier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ImportedCFDLevel(str, Enum):
    """Background-mesh resolution levels for the imported-mesh tier."""

    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"


@dataclass(frozen=True)
class LevelParams:
    """Per-level snappyHexMesh settings.

    ``cell_share`` is the background cell size as a share of the mesh's own
    streamwise length, so the tunnel scales with the object rather than
    assuming a car.
    """

    cell_share: float
    surface_levels: int
    n_layers: int


LEVELS: dict[ImportedCFDLevel, LevelParams] = {
    ImportedCFDLevel.COARSE: LevelParams(cell_share=1 / 24.0, surface_levels=2, n_layers=3),
    ImportedCFDLevel.MEDIUM: LevelParams(cell_share=1 / 36.0, surface_levels=2, n_layers=4),
    ImportedCFDLevel.FINE: LevelParams(cell_share=1 / 54.0, surface_levels=3, n_layers=5),
}

#: run-up before the nose, in car lengths
INLET_MARGIN_LENGTHS = 1.5
#: wake space behind the tail, in car lengths
OUTLET_MARGIN_LENGTHS = 2.5
#: tunnel height, in car heights
TOP_MARGIN_LENGTHS = 2.5
#: total tunnel width, in car widths (1.0 = exactly the mesh's own width)
SIDE_MARGIN_LENGTHS = 3.0
#: same as the narrow axis above, in lateral sizing
LATERAL_CELL_SHRINK = 2.0
