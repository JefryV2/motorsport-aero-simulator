"""Biot-Savart vortex kernel for horseshoe vortices, symmetry, and ground effect."""

from __future__ import annotations

import numpy as np


def vortex_segment_velocity(
    p1: np.ndarray,
    p2: np.ndarray,
    p0: np.ndarray,
    gamma: float = 1.0,
    core_radius: float = 1e-5,
) -> np.ndarray:
    """Compute velocity induced by a finite straight vortex filament from p1 to p2 at p0.

    Uses standard regularized Biot-Savart law to prevent divide-by-zero singularities.
    """
    ra = p0 - p1
    rb = p0 - p2
    r_segment = p2 - p1

    ra_mag = np.linalg.norm(ra)
    rb_mag = np.linalg.norm(rb)

    if ra_mag < 1e-12 or rb_mag < 1e-12:
        return np.zeros(3, dtype=np.float64)

    cross_prod = np.cross(ra, rb)
    cross_mag_sq = np.dot(cross_prod, cross_prod)

    # Core radius cutoff for numerical stability
    denom = 4.0 * np.pi * (cross_mag_sq + core_radius**2)

    # Projection factor: (r2 - r1) . (ra / |ra| - rb / |rb|)
    geom_factor = np.dot(r_segment, (ra / ra_mag) - (rb / rb_mag))

    vel: np.ndarray = np.asarray((gamma * geom_factor / denom) * cross_prod, dtype=np.float64)
    return vel


def semi_infinite_trailing_velocity(
    p_start: np.ndarray,
    direction: np.ndarray,
    p0: np.ndarray,
    gamma: float = 1.0,
    core_radius: float = 1e-5,
) -> np.ndarray:
    """Compute velocity induced by a semi-infinite line vortex starting at p_start extending along direction."""
    u_inf = direction / np.linalg.norm(direction)
    ra = p0 - p_start
    ra_mag = np.linalg.norm(ra)

    if ra_mag < 1e-12:
        return np.zeros(3, dtype=np.float64)

    cross_prod = np.cross(u_inf, ra)
    cross_mag_sq = np.dot(cross_prod, cross_prod)

    denom = 4.0 * np.pi * (cross_mag_sq + core_radius**2)
    geom_factor = 1.0 + np.dot(u_inf, ra) / ra_mag

    vel: np.ndarray = np.asarray((gamma * geom_factor / denom) * cross_prod, dtype=np.float64)
    return vel


def horseshoe_induced_velocity(
    p_left: np.ndarray,
    p_right: np.ndarray,
    p_field: np.ndarray,
    freestream_dir: np.ndarray | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Velocity induced by a horseshoe vortex with bound segment [p_left, p_right] at field point p_field.

    Trailing legs extend downstream parallel to freestream_dir (+x by default).
    Circulation follows right-hand rule from p_left to p_right.
    """
    if freestream_dir is None:
        u_inf = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u_inf = freestream_dir / np.linalg.norm(freestream_dir)

    # 1. Left trailing leg: extends from p_left downstream along +u_inf
    # Direction of vortex filament is from infinity to p_left, so circulation is -gamma along +u_inf
    v_left = semi_infinite_trailing_velocity(p_left, u_inf, p_field, gamma=-gamma)

    # 2. Bound vortex filament: from p_left to p_right
    v_bound = vortex_segment_velocity(p_left, p_right, p_field, gamma=gamma)

    # 3. Right trailing leg: extends from p_right downstream along +u_inf
    # Direction of vortex filament is from p_right to infinity, so circulation is +gamma along +u_inf
    v_right = semi_infinite_trailing_velocity(p_right, u_inf, p_field, gamma=gamma)

    v_total: np.ndarray = np.asarray(v_left + v_bound + v_right, dtype=np.float64)
    return v_total


def horseshoe_symmetric_and_ground_velocity(
    p_left: np.ndarray,
    p_right: np.ndarray,
    p_field: np.ndarray,
    use_symmetry: bool = True,
    ground_height: float | None = None,
    freestream_dir: np.ndarray | None = None,
) -> np.ndarray:
    """Compute total induced velocity including lateral symmetry (y=0) and image ground effect.

    - use_symmetry: Adds mirror horseshoe reflected across y=0 plane with same circulation sense.
    - ground_height: If specified, adds mirror image horseshoe reflected across z = ground_height with
      OPPOSITE circulation (-gamma) to satisfy zero-normal-flow on ground plane.
    """
    v_total = np.zeros(3, dtype=np.float64)

    # 1. Primary horseshoe
    v_total += horseshoe_induced_velocity(p_left, p_right, p_field, freestream_dir, gamma=1.0)

    # 2. Symmetric mirror across y = 0
    if use_symmetry:
        p_left_sym = np.array([p_right[0], -p_right[1], p_right[2]])
        p_right_sym = np.array([p_left[0], -p_left[1], p_left[2]])
        v_total += horseshoe_induced_velocity(
            p_left_sym, p_right_sym, p_field, freestream_dir, gamma=1.0
        )

    # 3. Ground effect image system (opposite circulation -1.0)
    if ground_height is not None:
        zg = ground_height
        # Mirror z across z = zg: z_image = 2 * zg - z
        p_left_img = np.array([p_left[0], p_left[1], 2.0 * zg - p_left[2]])
        p_right_img = np.array([p_right[0], p_right[1], 2.0 * zg - p_right[2]])
        v_total += horseshoe_induced_velocity(
            p_left_img, p_right_img, p_field, freestream_dir, gamma=-1.0
        )

        if use_symmetry:
            p_left_sym_img = np.array([p_right[0], -p_right[1], 2.0 * zg - p_right[2]])
            p_right_sym_img = np.array([p_left[0], -p_left[1], 2.0 * zg - p_left[2]])
            v_total += horseshoe_induced_velocity(
                p_left_sym_img, p_right_sym_img, p_field, freestream_dir, gamma=-1.0
            )

    return v_total
