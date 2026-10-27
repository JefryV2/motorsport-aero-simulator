"""VLM geometry definition and surface discretization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Panel:
    """Single quadrilateral vortex lattice panel."""

    p1: np.ndarray  # Leading Edge - Left [x, y, z]
    p2: np.ndarray  # Leading Edge - Right [x, y, z]
    p3: np.ndarray  # Trailing Edge - Right [x, y, z]
    p4: np.ndarray  # Trailing Edge - Left [x, y, z]

    bound_p1: np.ndarray  # Bound vortex left point (c/4)
    bound_p2: np.ndarray  # Bound vortex right point (c/4)
    cp: np.ndarray  # Collocation / Control point (3c/4)
    normal: np.ndarray  # Unit normal vector
    area: float  # Panel area (m^2)
    span: float  # Panel span delta y (m)
    chord: float  # Panel chord (m)
    surface_id: str  # Surface name/ID


@dataclass
class VLMSurface:
    """Parametric definition of a lifting surface."""

    name: str
    origin: tuple[float, float, float]  # LE center (x, y, z) in metres
    span: float  # Total span b (m)
    chord: float  # Chord length (m)
    angle_deg: float  # Incidence angle (degrees)
    camber: float = 0.0  # Camber maximum height ratio (e.g. 0.04)
    n_span: int = 16  # Number of spanwise panels per half-span
    n_chord: int = 4  # Number of chordwise panels


def generate_panels(surface: VLMSurface, full_span: bool = False) -> list[Panel]:
    """Discretize a VLMSurface into panels with cosine spanwise spacing.

    If full_span is False, generates panels for the positive semi-span (y >= 0)
    for use with mirror symmetry formulations.
    """
    x0, y0, z0 = surface.origin
    b = surface.span
    c = surface.chord
    alpha_rad = math.radians(surface.angle_deg)

    # Cosine spacing along span
    if full_span:
        eta = np.linspace(-1.0, 1.0, 2 * surface.n_span + 1)
        y_nodes = (b / 2.0) * np.sin(eta * np.pi / 2.0)
    else:
        # Semi-span y in [0, b/2]
        theta = np.linspace(0.0, np.pi / 2.0, surface.n_span + 1)
        y_nodes = (b / 2.0) * (1.0 - np.cos(theta))

    # Chordwise discretization (uniform or cosine)
    x_frac = np.linspace(0.0, 1.0, surface.n_chord + 1)

    panels: list[Panel] = []
    n_y_strips = len(y_nodes) - 1

    for j in range(n_y_strips):
        y_left = y0 + y_nodes[j]
        y_right = y0 + y_nodes[j + 1]
        dy = abs(y_right - y_left)

        for i in range(surface.n_chord):
            # Panel chordwise boundaries [x_i, x_i+1]
            c_start = x_frac[i] * c
            c_end = x_frac[i + 1] * c
            panel_chord = c_end - c_start

            # Quarter-chord and three-quarter chord positions relative to panel
            x_b = c_start + 0.25 * panel_chord
            x_cp = c_start + 0.75 * panel_chord

            # Apply incidence rotation: x_rot = x * cos(alpha), z_rot = -x * sin(alpha)
            # Inverted race wing convention: positive angle increases downforce (downward normal tilt)
            def transform_point(xc: float, yc: float) -> np.ndarray:
                # Add parabolic mean camber line: z_camber = 4 * camber * xc * (1 - xc/c)
                zc = 4.0 * surface.camber * xc * (1.0 - xc / c) if c > 0 else 0.0
                xp = x0 + xc * math.cos(alpha_rad) + zc * math.sin(alpha_rad)
                zp = z0 - xc * math.sin(alpha_rad) + zc * math.cos(alpha_rad)
                return np.array([xp, yc, zp], dtype=np.float64)

            p1 = transform_point(c_start, y_left)
            p2 = transform_point(c_start, y_right)
            p3 = transform_point(c_end, y_right)
            p4 = transform_point(c_end, y_left)

            bound_p1 = transform_point(x_b, y_left)
            bound_p2 = transform_point(x_b, y_right)

            y_mid = 0.5 * (y_left + y_right)
            cp = transform_point(x_cp, y_mid)

            # Panel normal from diagonals: cross(p3 - p1, p2 - p4)
            d1 = p3 - p1
            d2 = p2 - p4
            norm_raw = np.cross(d1, d2)
            norm_len = np.linalg.norm(norm_raw)
            normal = norm_raw / norm_len if norm_len > 1e-12 else np.array([0.0, 0.0, 1.0])

            # Ensure normal points generally upward or in the direction of positive lift
            if normal[2] < 0:
                normal = -normal

            area = 0.5 * norm_len

            panels.append(
                Panel(
                    p1=p1,
                    p2=p2,
                    p3=p3,
                    p4=p4,
                    bound_p1=bound_p1,
                    bound_p2=bound_p2,
                    cp=cp,
                    normal=normal,
                    area=area,
                    span=dy,
                    chord=panel_chord,
                    surface_id=surface.name,
                )
            )

    return panels
