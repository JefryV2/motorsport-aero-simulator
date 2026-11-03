"""Formula open-wheel aerodynamic configuration models."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SimulationMethod(str, Enum):
    """Supported simulation methods."""

    ANALYTICAL = "analytical"
    ANALYTICAL_ENHANCED = "analytical_enhanced"
    CFD_STUB = "cfd_stub"
    # Legacy aliases for backwards compatibility
    PANEL_VLM = "panel_vlm"
    CFD = "cfd"


class WingElement(BaseModel):
    """Single wing element (main plane or flap)."""

    angle_deg: float = Field(ge=-5.0, le=35.0, description="Incidence angle in degrees")
    chord_m: float = Field(gt=0, description="Chord length in metres")
    span_m: float = Field(gt=0, description="Span in metres")
    gurney_mm: float = Field(default=0.0, ge=0, le=50, description="Gurney flap height in mm")


class FormulaAeroConfig(BaseModel):
    """Complete aero setup for a formula open-wheel car."""

    front_wing_main: WingElement = Field(
        default_factory=lambda: WingElement(angle_deg=2.0, chord_m=0.35, span_m=1.80)
    )
    front_wing_flap: WingElement = Field(
        default_factory=lambda: WingElement(angle_deg=8.0, chord_m=0.12, span_m=1.75)
    )
    rear_wing_main: WingElement = Field(
        default_factory=lambda: WingElement(angle_deg=5.0, chord_m=0.40, span_m=0.95)
    )
    rear_wing_flap: WingElement = Field(
        default_factory=lambda: WingElement(angle_deg=15.0, chord_m=0.15, span_m=0.90)
    )
    drs_active: bool = Field(
        default=False,
        description="Rear wing flap folded toward the main plane (DRS open)",
    )
    ride_height_front_mm: float = Field(default=25.0, ge=5.0, le=200.0)
    ride_height_rear_mm: float = Field(default=45.0, ge=10.0, le=250.0)
    rake_deg: float = Field(default=1.5, ge=-2.0, le=4.0, description="Nose-up rake angle")
    diffuser_angle_deg: float = Field(default=12.0, ge=5.0, le=20.0)
    diffuser_ride_height_mm: float = Field(default=30.0, ge=10.0, le=60.0)
    speed_kmh: float = Field(default=200.0, ge=50.0, le=350.0)


class AeroForces(BaseModel):
    """Aerodynamic force breakdown."""

    drag_N: float
    downforce_total_N: float
    downforce_front_N: float
    downforce_rear_N: float
    balance_front_pct: float
    cop_x_m: float = Field(description="Center of pressure from front axle, metres")
    cl_total: float
    cd_total: float
    efficiency: float = Field(description="L/D ratio (downforce/drag)")


class SimulationResult(BaseModel):
    """Full simulation output.

    ``fidelity`` and ``provenance`` are additive fields used to keep the UI and
    downstream consumers honest about what produced the numbers. They never
    change the aerodynamic values themselves.
    """

    config: FormulaAeroConfig
    forces: AeroForces
    lap_time_delta_s: float = Field(description="Estimated lap time delta vs baseline")
    method: str
    fidelity: str = Field(
        default="unknown",
        description="One of: empirical, low_fidelity_vlm, synthetic_model, rans",
    )
    provenance: dict[str, Any] = Field(default_factory=dict)


class OptimizationObjective(str, Enum):
    MAX_DOWNFORCE = "max_downforce"
    MIN_DRAG = "min_drag"
    MAX_EFFICIENCY = "max_efficiency"
    BALANCED = "balanced"


class OptimizationRequest(BaseModel):
    config: FormulaAeroConfig
    objective: OptimizationObjective = OptimizationObjective.BALANCED
    max_drag_N: float | None = None
    target_balance_pct: float | None = Field(default=42.0, ge=35.0, le=50.0)
    n_trials: int = Field(default=100, ge=10, le=500)


class OptimizationResult(BaseModel):
    best_config: FormulaAeroConfig
    best_forces: AeroForces
    improvement_pct: float = Field(
        description=(
            "Relative improvement in the selected scalar objective score "
            "(not a lap-time gain and not a physically calibrated percentage)"
        )
    )
    trials_run: int
    pareto_front: list[dict] = Field(
        default_factory=list,
        description="Non-dominated drag/downforce points from the sampled trials",
    )
    target_balance_pct: float | None = Field(
        default=None,
        description="Balance target used by the objective, if any",
    )
    achievable_balance_pct: list[float] = Field(
        default_factory=list,
        description="[min, max] front balance observed among sampled trials",
    )
    target_reachable: bool | None = Field(
        default=None,
        description=(
            "Whether the balance target lies inside the sampled balance range; "
            "false means the target is outside the model's sampled feasible region"
        ),
    )


class AdvisorRequest(BaseModel):
    config: FormulaAeroConfig
    simulation: SimulationResult
    optimization: OptimizationResult | None = None
    question: str | None = None
