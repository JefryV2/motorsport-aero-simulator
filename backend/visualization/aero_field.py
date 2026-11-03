"""Compute aerodynamic surface fields mapped onto 3D mesh vertices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.models.aero_config import FormulaAeroConfig
from backend.simulation.analytical import (
    CAR_HALF_WIDTH_M,
    CAR_HEIGHT_M,
    CAR_LENGTH_M,
    DIFFUSER_LENGTH_M,
    FLOOR_HALF_WIDTH_M,
    FLOOR_X_END_M,
    FLOOR_X_START_M,
    FRONT_AXLE_M,
    GE_DECAY_DIFFUSER_M,
    GE_DECAY_FRONT_M,
    GE_DECAY_REAR_M,
    RAKE_COEF_PER_DEG,
    REAR_AXLE_M,
    REAR_WING_BOTTOM_M,
    REAR_WING_TOP_M,
    TRACK_M,
    WHEEL_RADIUS_FRONT_M,
    WHEEL_RADIUS_REAR_M,
    WHEEL_WIDTH_FRONT_M,
    WHEEL_WIDTH_REAR_M,
    diffuser_exit_height_m,
    diffuser_throat_height_m,
    effective_rear_flap,
    floor_height_at_m,
    wing_stations,
)
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


def _ramp(v: float, lo: float, hi: float) -> float:
    """0 at `lo`, 1 at `hi`, linear between, clamped outside."""
    if hi <= lo:
        return 1.0 if v >= hi else 0.0
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))


def _band(v: float, lo: float, hi: float) -> float:
    """Membership of `v` in the window [lo, hi]: full inside, gone just outside.

    The fade is 10 % of the window's own length rather than a fixed distance, so
    a vertex on a surface that sits at the edge of its station window - the rear
    flap at the endplate crown, a wing leading edge - keeps its full weight while
    a coarse mesh still blends across the station instead of stamping a hard seam.
    """
    span = hi - lo
    if span <= 0.0:
        return 1.0 if lo <= v <= hi else 0.0
    fade = 0.1 * span
    return _ramp(v, lo - fade, lo) * _ramp(hi + fade - v, 0.0, fade)


def _inside(v: float, edge: float) -> float:
    """Full near the centre line, 0 by `edge` (a half-width in metres)."""
    return _ramp(edge - abs(v), 0.0, 0.5 * edge)


@dataclass(frozen=True)
class SurfaceRegions:
    """Surface windows for the pressure map, in car metres (nose x=0, ground z=0).

    Every edge is a station the solver or the drawn body actually has: the four
    lifting surfaces come from `analytical.wing_stations`, the floor and diffuser
    from the shared floor stations and the throat/exit heights, the tyres from the
    axles. A vertex on the reference car therefore shades the surface it sits on
    rather than a fraction of whatever bounding box was uploaded with it.

    The caller's mesh is still normalised into 0-1 by its own box first, which is
    what makes the same layout work for an uploaded model: the box is read as a
    formula car of `CAR_LENGTH_M` x `CAR_HEIGHT_M` x 2`CAR_HALF_WIDTH_M`.
    """

    config: FormulaAeroConfig
    front_wing_x: tuple[float, float]
    front_wing_top_m: float
    rear_wing_x: tuple[float, float]
    floor_x: tuple[float, float]
    diffuser_x: tuple[float, float]
    diffuser_throat_m: float
    diffuser_exit_m: float

    @classmethod
    def from_config(cls, config: FormulaAeroConfig) -> SurfaceRegions:
        stations = {s.name: s for s in wing_stations(config)}
        front = (stations["front_wing_main"], stations["front_wing_flap"])
        rear = (stations["rear_wing_main"], stations["rear_wing_flap"])
        diffuser_start = FLOOR_X_END_M - DIFFUSER_LENGTH_M
        return cls(
            config=config,
            front_wing_x=(front[0].x_le_m, front[1].x_te_m),
            front_wing_top_m=max(s.height_m for s in front),
            rear_wing_x=(rear[0].x_le_m, rear[1].x_te_m),
            floor_x=(FLOOR_X_START_M, diffuser_start),
            diffuser_x=(diffuser_start, FLOOR_X_END_M),
            diffuser_throat_m=diffuser_throat_height_m(config),
            diffuser_exit_m=diffuser_exit_height_m(config),
        )

    def weights(self, x: float, y: float, z: float) -> dict[str, float]:
        """Soft membership of a car-metres point in each charged surface."""
        abs_y = abs(y)

        # Ground-effect channel: the floor line at this station, one channel
        # height above it, and the floor planform's own half width. Each surface
        # region is full weight up to its own line and gone one line-height
        # above, so a vertex *on* the surface always reads as that surface.
        h_floor = floor_height_at_m(self.config, x)
        ramp_top = self.diffuser_throat_m + (self.diffuser_exit_m - self.diffuser_throat_m) * _ramp(
            x, self.diffuser_x[0], self.diffuser_x[1]
        )

        front_wing = _band(x, *self.front_wing_x) * _ramp(
            2.0 * self.front_wing_top_m - z, 0.0, self.front_wing_top_m
        )
        rear_wing = _band(x, *self.rear_wing_x) * _band(z, REAR_WING_BOTTOM_M, REAR_WING_TOP_M)
        underbody = (
            _band(x, *self.floor_x) * _ramp(2.0 * h_floor - z, 0.0, h_floor) * _inside(y, FLOOR_HALF_WIDTH_M)
        )
        diffuser = (
            _band(x, *self.diffuser_x) * _ramp(2.0 * ramp_top - z, 0.0, ramp_top) * _inside(y, FLOOR_HALF_WIDTH_M)
        )
        wheels = max(
            self._wheel_weight(FRONT_AXLE_M, WHEEL_RADIUS_FRONT_M, WHEEL_WIDTH_FRONT_M, x, abs_y, z),
            self._wheel_weight(REAR_AXLE_M, WHEEL_RADIUS_REAR_M, WHEEL_WIDTH_REAR_M, x, abs_y, z),
        )
        body = max(0.0, 1.0 - front_wing - rear_wing * 0.5 - underbody * 0.3 - diffuser * 0.3)

        return {
            "front_wing": min(1.0, front_wing),
            "rear_wing": min(1.0, rear_wing),
            "underbody": min(1.0, underbody),
            "diffuser": min(1.0, diffuser),
            "body": min(1.0, body),
            "wheels": min(1.0, wheels),
        }

    @staticmethod
    def _wheel_weight(
        axle_x: float, radius: float, width: float, x: float, abs_y: float, z: float
    ) -> float:
        """One tyre: the axle station, the outboard half of the track, tyre height."""
        top = 2.0 * radius
        return (
            _band(x, axle_x - radius, axle_x + radius)
            * _band(abs_y, TRACK_M / 2.0 - width, TRACK_M / 2.0)
            * _ramp(top - z, 0.0, 0.5 * top)
        )


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
    rw_angle = config.rear_wing_main.angle_deg + effective_rear_flap(config).angle_deg * 0.6
    ge_front = math.exp(-config.ride_height_front_mm / 1000.0 / GE_DECAY_FRONT_M)
    ge_rear = math.exp(-config.ride_height_rear_mm / 1000.0 / GE_DECAY_REAR_M)
    ge_diffuser = math.exp(-diffuser_throat_height_m(config) / GE_DECAY_DIFFUSER_M)
    rake = 1.0 + config.rake_deg * RAKE_COEF_PER_DEG
    diff_boost = math.sin(math.radians(config.diffuser_angle_deg)) * 0.8

    cp_values: list[float] = []
    regions_frame = SurfaceRegions.from_config(config)
    for i, (x, y, z) in enumerate(norm_verts):
        # Normalised 0-1 -> car metres on the reference envelope, so the region
        # edges below are stations and not fractions of this mesh's bounding box.
        x_m = float(x) * CAR_LENGTH_M
        y_m = (float(y) - 0.5) * 2.0 * CAR_HALF_WIDTH_M
        z_m = float(z) * CAR_HEIGHT_M
        regions = regions_frame.weights(x_m, y_m, z_m)

        cp = 0.05 * cd_scale  # base stagnation

        if norms is not None and i < len(norms):
            nx, ny, nz = norms[i]
            cp += -0.15 * nz * cl_scale
            cp += 0.08 * nx * cd_scale

        cp -= regions["front_wing"] * (fw_angle / 15.0) * 1.8 * ge_front * cl_scale
        cp -= regions["rear_wing"] * (rw_angle / 20.0) * 1.6 * cl_scale
        cp -= regions["underbody"] * ge_rear * rake * 1.2 * cl_scale
        cp -= regions["diffuser"] * diff_boost * ge_diffuser * rake * cl_scale
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

    wake_strength = effective_rear_flap(config).angle_deg / 30.0

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
        "downforce": {
            "origin": [0, 0, 0.8],
            "direction": [0, 0, -1],
            "magnitude": f.downforce_total_N,
        },
        "front_downforce": {
            "origin": [-0.8, 0, 0.3],
            "direction": [0, 0, -1],
            "magnitude": f.downforce_front_N,
        },
        "rear_downforce": {
            "origin": [0.8, 0, 0.5],
            "direction": [0, 0, -1],
            "magnitude": f.downforce_rear_N,
        },
    }
