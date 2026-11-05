"""API models for the Phase 5 lap simulation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.models.aero_config import FormulaAeroConfig
from backend.simulation.lap.vehicle import Vehicle


class LapTrackInput(BaseModel):
    """Centreline input. Provide raw points or CSV text, never both required."""

    points: list[list[float]] | None = Field(
        default=None,
        min_length=8,
        description="[[x, y], ...] or [[x, y, z], ...] in metres",
    )
    csv: str | None = Field(
        default=None,
        max_length=4_000_000,
        description="CSV/TSV centreline text; first two numeric columns are x, y",
    )
    closed: bool = True
    ds_m: float | None = Field(default=None, gt=0.0, le=10.0)


class LapVehicleInput(BaseModel):
    """Point-mass parameters; defaults describe a generic formula car."""

    mass_kg: float = Field(default=798.0, gt=0.0, le=3000.0)
    wheelbase_m: float = Field(default=3.35, gt=0.0, le=6.0)
    cg_height_m: float = Field(default=0.30, ge=0.05, le=1.0)
    front_static_pct: float = Field(default=0.46, gt=0.0, lt=1.0)
    track_width_m: float = Field(default=1.60, gt=0.0, le=3.0)
    power_kw: float = Field(default=750.0, ge=0.0, le=2000.0)
    drivetrain_efficiency: float = Field(default=0.90, gt=0.0, le=1.0)
    tyre_mu: float = Field(default=1.60, gt=0.0, le=5.0)
    tyre_load_sensitivity: float = Field(default=0.10, ge=0.0, le=1.0)
    tyre_reference_load_N: float | None = Field(default=None, gt=0.0)
    rolling_resistance: float = Field(default=0.012, ge=0.0, le=0.1)
    roll_stiffness_front_pct: float = Field(default=0.55, gt=0.0, lt=1.0)
    max_speed_ms: float = Field(default=100.0, gt=0.0, le=150.0)
    max_tractive_force_N: float | None = Field(default=None, gt=0.0)
    rear_wheel_drive: bool = True

    def to_vehicle(self) -> Vehicle:
        return Vehicle(**self.model_dump())


class LapRequest(BaseModel):
    config: FormulaAeroConfig
    track: LapTrackInput
    vehicle: LapVehicleInput = Field(default_factory=LapVehicleInput)
    method: str = Field(
        default="analytical_enhanced",
        description="analytical, analytical_enhanced or panel_vlm; CFD is point-only",
    )
    initial_speed_ms: float | None = Field(default=None, ge=0.0, le=150.0)


class LapResponse(BaseModel):
    lap_time_s: float
    sector_times_s: list[float]
    length_m: float
    track_points: int
    closed: bool
    ds_m: float
    top_speed_kmh: float
    min_speed_kmh: float
    mean_speed_kmh: float
    iterations: int
    converged: bool
    entry_speed_limited: bool
    max_utilisation: float
    s_m: list[float]
    speed_kmh: list[float]
    ay_ms2: list[float]
    ax_ms2: list[float]
    curvature_1pm: list[float]
    downforce_N: list[float]
    drag_N: list[float]
    aero_source: str
    aero_fidelity: str
    track_source: str
    provenance: dict[str, Any]
