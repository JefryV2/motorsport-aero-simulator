"""Aerodynamic maps used by the lap simulation.

The live aero models in this project are quasi-steady and scale with dynamic
pressure (the VLM is linear, the floor and body terms are ``q * coefficient``),
so one evaluation at a reference speed is enough to build a map that is exact
for those models at any other speed. Nothing here calls CFD per track point:
the CFD tier is a single-point verification, not a lap-sweep tool.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.models.aero_config import FormulaAeroConfig


@dataclass(frozen=True)
class AeroPoint:
    """Aerodynamic loads at one operating point (all forces in newtons)."""

    downforce_N: float
    drag_N: float
    front_share: float = 0.5


@dataclass(frozen=True)
class ConstantAeroMap:
    """Analytic map for validation cases.

    ``downforce_area_m2`` and ``drag_area_m2`` are ``Cl * A`` and ``Cd * A`` in
    square metres, so the loads are ``q * area``.
    """

    downforce_area_m2: float = 0.0
    drag_area_m2: float = 0.0
    front_share: float = 0.5
    rho: float = 1.225

    def at(self, speed_ms: float) -> AeroPoint:
        q = 0.5 * self.rho * speed_ms * speed_ms
        return AeroPoint(
            downforce_N=q * self.downforce_area_m2,
            drag_N=q * self.drag_area_m2,
            front_share=self.front_share,
        )


class AeroMap:
    """Dynamic-pressure-scaled map from the analytical or VLM aero model."""

    def __init__(self, config: FormulaAeroConfig, method: str = "analytical_enhanced"):
        if method in ("cfd", "cfd_stub"):
            raise ValueError(
                "Lap simulation needs an aero map, not a single CFD point. "
                "Use the analytical/VLM map, and keep CFD for point verification."
            )
        if method not in ("analytical", "analytical_enhanced", "panel_vlm"):
            raise ValueError(f"Unknown aero method for the lap map: {method}")

        from backend.simulation.analytical import simulate
        from backend.simulation.panel_method import simulate_panel

        if method == "analytical":
            result = simulate(config)
        else:
            result = simulate_panel(config, method=method)

        forces = result.forces
        q = 0.5 * 1.225 * max(config.speed_kmh / 3.6, 0.1) ** 2
        self.downforce_coefficient = forces.downforce_total_N / q if q > 0 else 0.0
        self.drag_coefficient = forces.drag_N / q if q > 0 else 0.0
        if forces.downforce_total_N > 0:
            self.front_share = float(
                min(1.0, max(0.0, forces.downforce_front_N / forces.downforce_total_N))
            )
        else:
            self.front_share = 0.5

        self.method = method
        self.fidelity = str(result.fidelity)
        self.source = str(result.provenance.get("source", "aero model"))
        self.reference_speed_ms = max(config.speed_kmh / 3.6, 0.1)

    def at(self, speed_ms: float) -> AeroPoint:
        q = 0.5 * 1.225 * speed_ms * speed_ms
        return AeroPoint(
            downforce_N=q * self.downforce_coefficient,
            drag_N=q * self.drag_coefficient,
            front_share=self.front_share,
        )
