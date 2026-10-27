"""VLM linear system assembly and circulation solver."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from backend.simulation.vlm.geometry import Panel
from backend.simulation.vlm.horseshoe import horseshoe_symmetric_and_ground_velocity


def build_influence_matrix(
    panels: list[Panel],
    use_symmetry: bool = True,
    ground_height: float | None = None,
    freestream_dir: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the N x N Aerodynamic Influence Coefficient (AIC) matrix.

    A[i, j] represents the normal velocity induced at control point i
    by a unit circulation Gamma_j = 1 on panel j.
    """
    n_panels = len(panels)
    aic = np.zeros((n_panels, n_panels), dtype=np.float64)

    for i in range(n_panels):
        cp_i = panels[i].cp
        normal_i = panels[i].normal

        for j in range(n_panels):
            p_left = panels[j].bound_p1
            p_right = panels[j].bound_p2

            v_ind = horseshoe_symmetric_and_ground_velocity(
                p_left=p_left,
                p_right=p_right,
                p_field=cp_i,
                use_symmetry=use_symmetry,
                ground_height=ground_height,
                freestream_dir=freestream_dir,
            )
            aic[i, j] = np.dot(v_ind, normal_i)

    return aic


def solve_circulations(
    panels: list[Panel],
    freestream_vel: np.ndarray,
    use_symmetry: bool = True,
    ground_height: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the VLM linear system A * gamma = b for panel circulations.

    Returns (gamma, aic_matrix).
    """
    n_panels = len(panels)
    v_inf = np.asarray(freestream_vel, dtype=np.float64)
    speed = np.linalg.norm(v_inf)
    u_inf = v_inf / speed if speed > 1e-12 else np.array([1.0, 0.0, 0.0])

    aic = build_influence_matrix(
        panels,
        use_symmetry=use_symmetry,
        ground_height=ground_height,
        freestream_dir=u_inf,
    )

    # Right hand side b[i] = - V_inf . normal_i
    rhs = np.zeros(n_panels, dtype=np.float64)
    for i in range(n_panels):
        rhs[i] = -np.dot(v_inf, panels[i].normal)

    # Solve linear system
    gamma = linalg.solve(aic, rhs)

    return gamma, aic
