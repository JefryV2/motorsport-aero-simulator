"""High-level VLM simulation coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.simulation.vlm.forces import SectionForceResult, compute_vlm_forces
from backend.simulation.vlm.geometry import Panel, VLMSurface, generate_panels
from backend.simulation.vlm.solver import solve_circulations


@dataclass
class VLMSolution:
    """Complete VLM simulation result across all lifting surfaces."""

    sections: dict[str, SectionForceResult]
    total_downforce_N: float
    total_lift_N: float
    total_drag_N: float
    total_cl: float
    total_cd: float
    efficiency: float
    cop_x_m: float
    circulations: list[float]
    warnings: list[str] = field(default_factory=list)


def solve_vlm(
    surfaces: list[VLMSurface],
    speed_kmh: float = 200.0,
    ground_height: float | None = None,
    use_symmetry: bool = True,
    rho: float = 1.225,
) -> VLMSolution:
    """Run full coupled Vortex Lattice Method simulation over multiple wing surfaces.

    Args:
        surfaces: List of VLMSurface objects (main wing, flaps, rear wing, etc.)
        speed_kmh: Freestream air speed in km/h
        ground_height: Ground plane clearance (m) for mirror image ground effect
        use_symmetry: Whether to exploit half-span y >= 0 mirror symmetry
        rho: Air density (kg/m^3)
    """
    speed_ms = speed_kmh / 3.6
    v_inf = np.array([speed_ms, 0.0, 0.0], dtype=np.float64)

    # Discretize all surfaces into a single unified panel collection
    all_panels: list[Panel] = []
    for surf in surfaces:
        panels = generate_panels(surf, full_span=(not use_symmetry))
        all_panels.extend(panels)

    # Solve coupled circulation system
    gamma, _ = solve_circulations(
        all_panels,
        freestream_vel=v_inf,
        use_symmetry=use_symmetry,
        ground_height=ground_height,
    )

    # Evaluate force breakdown per component
    section_results = compute_vlm_forces(
        all_panels,
        gamma,
        speed_ms=speed_ms,
        rho=rho,
        use_symmetry=use_symmetry,
        ground_height=ground_height,
    )

    total_df = sum(s.downforce_N for s in section_results.values())
    total_lift = sum(s.lift_N for s in section_results.values())
    total_drag = sum(s.drag_total_N for s in section_results.values())

    # Combined Center of Pressure
    cop_moment = sum(s.lift_N * s.cop_x_m for s in section_results.values())
    cop_x = cop_moment / total_lift if abs(total_lift) > 1e-6 else 0.0

    ref_area = sum(s.chord * s.span for s in surfaces)
    q = 0.5 * rho * speed_ms**2
    total_cl = total_lift / (q * ref_area) if q > 0 and ref_area > 0 else 0.0
    total_cd = total_drag / (q * ref_area) if q > 0 and ref_area > 0 else 0.0
    efficiency = total_df / total_drag if total_drag > 0 else 0.0

    all_warnings: list[str] = []
    for s in section_results.values():
        all_warnings.extend(s.warnings)

    return VLMSolution(
        sections=section_results,
        total_downforce_N=round(total_df, 2),
        total_lift_N=round(total_lift, 2),
        total_drag_N=round(total_drag, 2),
        total_cl=round(total_cl, 4),
        total_cd=round(total_cd, 4),
        efficiency=round(efficiency, 2),
        cop_x_m=round(cop_x, 4),
        circulations=gamma.tolist(),
        warnings=all_warnings,
    )
