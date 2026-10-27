"""Compute aerodynamic surface fields mapped onto 3D mesh vertices."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.models.aero_config import FormulaAeroConfig
from backend.simulation.panel_method import simulate_panel


def _normalize_coords(
    vertices: np.ndarray,
    bounds: dict[str, list[float]] | None = None,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Normalize vertices to 0-1 car frame (x=front→rear, y=left→right, z=ground→top)."""
    if bounds is None:
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        bounds = {"min": vmin.tolist(), "max": vmax.tolist()}
    else:
        vmin = np.array(bounds["min"])
        vmax = np.array(bounds["max"])

    span = np.maximum(vmax - vmin, 1e-6)
    norm = (vertices - vmin) / span
    return norm, bounds


def _region_weight(x: float, y: float, z: float) -> dict[str, float]:
    """Soft region masks for formula car surfaces."""
    front_wing = max(0.0, 1.0 - x / 0.18) * max(0.0, 1.0 - z / 0.35)
    rear_wing = max(0.0, (x - 0.78) / 0.22) * max(0.0, (z - 0.55) / 0.45)
    underbody = max(0.0, 1.0 - z / 0.28) * (1.0 - abs(y - 0.5) * 1.6) * max(0.0, 1.0 - abs(x - 0.5) * 1.2)
    diffuser = max(0.0, (x - 0.62) / 0.38) * max(0.0, 1.0 - z / 0.22)
    body = max(0.0, 1.0 - front_wing - rear_wing * 0.5 - underbody * 0.3)
    wheels = max(0.0, 1.0 - abs(y - 0.5) * 3.5) * (1.0 if z < 0.35 else 0.0) * (
        1.0 if x < 0.25 or x > 0.72 else 0.3
    )
    return {
        "front_wing": min(1.0, front_wing),
        "rear_wing": min(1.0, rear_wing),
        "underbody": min(1.0, underbody),
        "diffuser": min(1.0, diffuser),
        "body": min(1.0, max(0.0, body)),
        "wheels": min(1.0, wheels),
    }


def compute_cp_field(
    vertices: list[list[float]] | np.ndarray,
    normals: list[list[float]] | np.ndarray | None,
    config: FormulaAeroConfig,
    bounds: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """
    Map pressure coefficient (Cp) to each vertex.
    Negative Cp = suction (downforce). Positive Cp = pressure (drag).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    norms = np.asarray(normals, dtype=np.float64) if normals is not None else None
    norm_verts, bounds = _normalize_coords(verts, bounds)

    sim = simulate_panel(config)
    cl_scale = sim.forces.cl_total / 3.0
    cd_scale = sim.forces.cd_total / 1.2

    fw_angle = config.front_wing_main.angle_deg + config.front_wing_flap.angle_deg * 0.6
    rw_angle = config.rear_wing_main.angle_deg + config.rear_wing_flap.angle_deg * 0.6
    ge_front = math.exp(-config.ride_height_front_mm / 1000.0 / 0.015)
    ge_rear = math.exp(-config.ride_height_rear_mm / 1000.0 / 0.020)
    rake = 1.0 + config.rake_deg * 0.1
    diff_boost = math.sin(math.radians(config.diffuser_angle_deg)) * 0.8

    cp_values: list[float] = []
    for i, (x, y, z) in enumerate(norm_verts):
        regions = _region_weight(float(x), float(y), float(z))

        cp = 0.05 * cd_scale  # base stagnation

        if norms is not None and i < len(norms):
            nx, ny, nz = norms[i]
            cp += -0.15 * nz * cl_scale
            cp += 0.08 * nx * cd_scale

        cp -= regions["front_wing"] * (fw_angle / 15.0) * 1.8 * ge_front * cl_scale
        cp -= regions["rear_wing"] * (rw_angle / 20.0) * 1.6 * cl_scale
        cp -= regions["underbody"] * ge_rear * rake * 1.2 * cl_scale
        cp -= regions["diffuser"] * diff_boost * ge_rear * rake * cl_scale
        cp += regions["body"] * 0.25 * cd_scale
        cp += regions["wheels"] * 0.45 * cd_scale

        cp = max(-3.0, min(cp, 1.5))
        cp_values.append(round(cp, 4))

    return {
        "cp": cp_values,
        "bounds": bounds,
        "simulation": sim.model_dump(),
        "legend": {"min": -2.5, "max": 1.0, "unit": "Cp"},
    }


def generate_streamlines(
    config: FormulaAeroConfig,
    bounds: dict[str, list[float]] | None = None,
    n_lines: int = 12,
    n_points: int = 40,
) -> list[list[list[float]]]:
    """Generate simplified streamlines around the car for visualization."""
    if bounds is None:
        bounds = {"min": [-2.0, -1.0, 0.0], "max": [2.0, 1.0, 1.2]}

    vmin = np.array(bounds["min"])
    vmax = np.array(bounds["max"])
    center = (vmin + vmax) / 2
    span = vmax - vmin

    speed = config.speed_kmh / 3.6
    wake_strength = config.rear_wing_flap.angle_deg / 30.0

    lines: list[list[list[float]]] = []
    for i in range(n_lines):
        t = (i + 1) / (n_lines + 1)
        y_off = vmin[1] + span[1] * t
        z_off = vmin[2] + span[2] * (0.15 + 0.5 * (i % 3) / 2)

        points: list[list[float]] = []
        for step in range(n_points):
            frac = step / (n_points - 1)
            x = vmin[0] - span[0] * 0.5 + frac * span[0] * 1.8
            y = y_off
            z = z_off

            nx = (x - center[0]) / max(span[0], 0.1)
            ny = (y - center[1]) / max(span[1], 0.1)
            nz = (z - center[2]) / max(span[2], 0.1)
            dist = math.sqrt(nx**2 + ny**2 + nz**2)

            if dist < 0.35:
                lift_deflect = -0.12 * (config.front_wing_flap.angle_deg / 15.0)
                z += lift_deflect * span[2] * (1.0 - dist / 0.35)

            if nx > 0.2:
                z -= wake_strength * 0.06 * span[2] * (nx - 0.2)

            points.append([float(x), float(y), float(z)])

        lines.append(points)

    return lines


def force_vectors(config: FormulaAeroConfig) -> dict[str, Any]:
    """Return drag and downforce vectors for 3D arrow display."""
    sim = simulate_panel(config)
    f = sim.forces
    return {
        "drag": {"origin": [0, 0, 0.5], "direction": [-1, 0, 0], "magnitude": f.drag_N},
        "downforce": {"origin": [0, 0, 0.8], "direction": [0, 0, -1], "magnitude": f.downforce_total_N},
        "front_downforce": {"origin": [-0.8, 0, 0.3], "direction": [0, 0, -1], "magnitude": f.downforce_front_N},
        "rear_downforce": {"origin": [0.8, 0, 0.5], "direction": [0, 0, -1], "magnitude": f.downforce_rear_N},
    }
