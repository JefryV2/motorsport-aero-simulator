"""VLM force evaluation: Kutta-Joukowski, Trefftz plane induced drag, and envelope warnings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backend.simulation.vlm.geometry import Panel


@dataclass
class SectionForceResult:
    """Aerodynamic breakdown for a specific wing component."""

    surface_id: str
    lift_N: float  # Vertical force (N, positive upwards)
    downforce_N: float  # Positive downward (N)
    drag_induced_N: float  # Trefftz induced drag (N)
    drag_profile_N: float  # Section viscous drag (N)
    drag_total_N: float  # Induced + profile drag (N)
    cl: float  # Component lift coefficient
    cd_induced: float  # Component induced drag coefficient
    cd_profile: float  # Component profile drag coefficient
    cd_total: float  # Total drag coefficient
    cop_x_m: float  # Longitudinal center of pressure (m)
    efficiency: float  # L/D or DF/Drag ratio
    oswald_e: float  # Span efficiency factor
    warnings: list[str]  # Flow separation / stall envelope warnings


def compute_trefftz_induced_drag(
    panels: list[Panel],
    gamma: np.ndarray,
    speed_ms: float,
    rho: float,
    use_symmetry: bool = True,
    ground_height: float | None = None,
) -> float:
    """Compute induced drag in the Trefftz plane far downstream (x -> inf).

    In the Trefftz plane, bound vortices have zero influence and trailing vortices
    become infinite 2D point vortices in the crossflow plane (y, z).
    D_i = -0.5 * rho * sum(w_ind_i * Gamma_i * Delta_y_i)
    """
    if len(panels) == 0 or speed_ms < 1e-6:
        return 0.0

    # Collect spanwise trailing horseshoe endpoints and circulations
    # For a multi-chord surface, group panels into spanwise strips
    strip_map: dict[float, list[int]] = {}
    for idx, p in enumerate(panels):
        y_key = round(float(p.cp[1]), 4)
        strip_map.setdefault(y_key, []).append(idx)

    total_di = 0.0
    for idx_i, p_i in enumerate(panels):
        y_i = p_i.cp[1]
        z_i = p_i.cp[2]
        w_ind_z = 0.0

        for idx_j, p_j in enumerate(panels):
            gam_j = gamma[idx_j]
            y1, z1 = p_j.bound_p1[1], p_j.bound_p1[2]
            y2, z2 = p_j.bound_p2[1], p_j.bound_p2[2]

            # 2D point vortex downwash at (y_i, z_i) from trailing legs at (y1, z1) and (y2, z2)
            # Leg 1: circulation -gam_j at (y1, z1)
            dy1 = y_i - y1
            dz1 = z_i - z1
            r1_sq = dy1**2 + dz1**2 + 1e-8
            w_ind_z += (-gam_j / (2.0 * np.pi)) * (dy1 / r1_sq)

            # Leg 2: circulation +gam_j at (y2, z2)
            dy2 = y_i - y2
            dz2 = z_i - z2
            r2_sq = dy2**2 + dz2**2 + 1e-8
            w_ind_z += (gam_j / (2.0 * np.pi)) * (dy2 / r2_sq)

            # Symmetry mirror at (-y, z)
            if use_symmetry:
                dy1_s = y_i - (-y2)
                dz1_s = z_i - z2
                r1s_sq = dy1_s**2 + dz1_s**2 + 1e-8
                w_ind_z += (-gam_j / (2.0 * np.pi)) * (dy1_s / r1s_sq)

                dy2_s = y_i - (-y1)
                dz2_s = z_i - z1
                r2s_sq = dy2_s**2 + dz2_s**2 + 1e-8
                w_ind_z += (gam_j / (2.0 * np.pi)) * (dy2_s / r2s_sq)

            # Ground effect mirror (opposite sign) at (y, 2h - z)
            if ground_height is not None:
                zg = ground_height
                z_img1 = 2.0 * zg - z1
                z_img2 = 2.0 * zg - z2

                dy1_g = y_i - y1
                dz1_g = z_i - z_img1
                r1g_sq = dy1_g**2 + dz1_g**2 + 1e-8
                # Opposite circulation: -(-gam_j) = +gam_j
                w_ind_z += (gam_j / (2.0 * np.pi)) * (dy1_g / r1g_sq)

                dy2_g = y_i - y2
                dz2_g = z_i - z_img2
                r2g_sq = dy2_g**2 + dz2_g**2 + 1e-8
                w_ind_z += (-gam_j / (2.0 * np.pi)) * (dy2_g / r2g_sq)

        # Trefftz drag increment for this panel
        # F_ind = - 0.5 * rho * w_ind * Gamma * dy
        panel_di = -0.5 * rho * w_ind_z * gamma[idx_i] * p_i.span
        total_di += max(0.0, panel_di)

    # Double if lateral symmetry was used
    if use_symmetry:
        total_di *= 2.0

    return float(total_di)


def compute_vlm_forces(
    panels: list[Panel],
    gamma: np.ndarray,
    speed_ms: float,
    rho: float = 1.225,
    use_symmetry: bool = True,
    ground_height: float | None = None,
    ref_area: float | None = None,
) -> dict[str, SectionForceResult]:
    """Compute forces per surface element using Kutta-Joukowski and Trefftz plane analysis."""
    q = 0.5 * rho * speed_ms**2
    results: dict[str, SectionForceResult] = {}

    # Group panels by surface_id
    surface_groups: dict[str, list[int]] = {}
    for idx, p in enumerate(panels):
        surface_groups.setdefault(p.surface_id, []).append(idx)

    for surf_id, indices in surface_groups.items():
        surf_lift = 0.0
        cop_moment = 0.0
        warnings: list[str] = []

        surf_panels = [panels[i] for i in indices]
        surf_gamma = gamma[indices]
        surf_area = sum(p.area for p in surf_panels) * (2.0 if use_symmetry else 1.0)
        s_ref = ref_area if ref_area is not None else max(surf_area, 1e-4)

        # Kutta-Joukowski lift: L = rho * V_inf * Gamma * span
        for _i_local, i_global in enumerate(indices):
            p = panels[i_global]
            gam = gamma[i_global]

            dL = rho * speed_ms * gam * p.span
            surf_lift += dL
            cop_moment += dL * p.cp[0]

            # Sectional Cl check for flow separation (P1.5 Envelope warnings)
            panel_cl = (2.0 * gam) / (speed_ms * p.chord) if (speed_ms * p.chord) > 1e-6 else 0.0
            if abs(panel_cl) > 2.2:
                warning_msg = f"Local panel Cl ({panel_cl:.2f}) exceeds attached-flow limit (2.2) on {surf_id}."
                if warning_msg not in warnings:
                    warnings.append(warning_msg)

        if use_symmetry:
            surf_lift *= 2.0
            cop_moment *= 2.0

        cop_x = cop_moment / surf_lift if abs(surf_lift) > 1e-6 else surf_panels[0].cp[0]

        # Trefftz plane induced drag for this surface
        d_induced = compute_trefftz_induced_drag(
            surf_panels,
            surf_gamma,
            speed_ms=speed_ms,
            rho=rho,
            use_symmetry=use_symmetry,
            ground_height=ground_height,
        )

        cl = surf_lift / (q * s_ref) if q > 0 else 0.0
        cd_i = d_induced / (q * s_ref) if q > 0 else 0.0

        # Viscous profile drag polar (P1.4): Cd_p = Cd_0 + k * Cl^2
        # Racing wing section polar: Cd0 ~ 0.012, k ~ 0.008
        cd_p = 0.012 + 0.008 * (cl**2)
        d_profile = cd_p * q * s_ref

        d_total = d_induced + d_profile
        cd_total = cd_i + cd_p

        # Inverted race wing convention: downforce is positive downward (-lift)
        downforce = -surf_lift if surf_lift < 0 else surf_lift
        efficiency = downforce / d_total if d_total > 0 else 0.0

        # Calculate Oswald efficiency factor e = Cl^2 / (pi * AR * Cd_i)
        span_total = max(p.bound_p2[1] for p in surf_panels) * (2.0 if use_symmetry else 1.0)
        ar = (span_total**2) / surf_area if surf_area > 0 else 1.0
        oswald_e = (cl**2) / (math.pi * ar * cd_i) if (cd_i > 1e-6 and ar > 0) else 1.0

        results[surf_id] = SectionForceResult(
            surface_id=surf_id,
            lift_N=round(surf_lift, 2),
            downforce_N=round(downforce, 2),
            drag_induced_N=round(d_induced, 2),
            drag_profile_N=round(d_profile, 2),
            drag_total_N=round(d_total, 2),
            cl=round(cl, 4),
            cd_induced=round(cd_i, 4),
            cd_profile=round(cd_p, 4),
            cd_total=round(cd_total, 4),
            cop_x_m=round(cop_x, 4),
            efficiency=round(efficiency, 2),
            oswald_e=round(oswald_e, 3),
            warnings=warnings,
        )

    return results
