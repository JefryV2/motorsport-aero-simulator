"""Track geometry built from a user-supplied centreline.

No circuit coordinates are stored in this module. A :class:`Track` is always
constructed from points the caller supplies; the library never invents,
downloads, or hard-codes a real circuit. Analytic geometries used by the
validation cases live in the validation script, not in the production path.

The centreline is resampled to uniform arc length and the signed curvature is
estimated from the circumscribed circle of three points spaced a few metres
apart. Positive curvature is a left-hand turn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TrackError(ValueError):
    """Raised when a centreline cannot be turned into a usable track."""


@dataclass(frozen=True)
class Track:
    """Uniformly resampled closed or open centreline."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None
    closed: bool
    ds_m: float
    length_m: float
    heading_rad: np.ndarray
    curvature_1pm: np.ndarray
    source: str = "user-supplied centreline"

    @property
    def count(self) -> int:
        return int(self.x.size)

    @property
    def segment_count(self) -> int:
        return self.count if self.closed else self.count - 1


def _resample(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    closed: bool,
    ds_m: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, float]:
    if closed:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        if z is not None:
            z = np.append(z, z[0])

    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(s[-1])
    if not np.isfinite(length) or length <= 0.0:
        raise TrackError("Centreline has zero length")

    if ds_m is None:
        step = float(np.clip(length / 1500.0, 0.25, 2.0))
    else:
        step = float(ds_m)
    if step <= 0.0:
        raise TrackError("ds_m must be positive")

    # Parameter-free on purpose: numpy 2.2.6 stubs type linspace as
    # floating[Any], which would not unify with the closed branch's float64.
    s_out: np.ndarray
    if closed:
        n = max(8, int(round(length / step)))
        step = length / n
        s_out = np.arange(n, dtype=float) * step
    else:
        n = max(2, int(round(length / step)) + 1)
        step = length / (n - 1)
        s_out = np.linspace(0.0, length, n)

    xo = np.interp(s_out, s, x)
    yo = np.interp(s_out, s, y)
    zo = np.interp(s_out, s, z) if z is not None else None
    return xo, yo, zo, step, length


def _circumcircle_curvature(
    x: np.ndarray, y: np.ndarray, closed: bool, window: int
) -> np.ndarray:
    n = x.size
    if closed:
        window = min(window, n // 2 - 1)
    else:
        window = min(window, (n - 1) // 2)
    window = max(1, window)

    idx = np.arange(n)
    if closed:
        i0 = (idx - window) % n
        i2 = (idx + window) % n
    else:
        i0 = np.clip(idx - window, 0, n - 1)
        i2 = np.clip(idx + window, 0, n - 1)

    ax = x - x[i0]
    ay = y - y[i0]
    bx = x[i2] - x
    by = y[i2] - y
    cx = x[i2] - x[i0]
    cy = y[i2] - y[i0]

    cross = ax * by - ay * bx
    denom = np.hypot(ax, ay) * np.hypot(bx, by) * np.hypot(cx, cy)
    kappa = np.zeros(n, dtype=float)
    good = denom > 1e-12
    kappa[good] = 2.0 * cross[good] / denom[good]

    if not closed and good.sum() >= 2 and (~good).any():
        kappa[~good] = np.interp(idx[~good], idx[good], kappa[good])
    return kappa


def _smooth(values: np.ndarray, closed: bool) -> np.ndarray:
    if values.size < 3:
        return np.asarray(values, dtype=float)
    if closed:
        smoothed = 0.25 * (np.roll(values, 1) + 2.0 * values + np.roll(values, -1))
        return np.asarray(smoothed, dtype=float)
    out = values.copy()
    out[1:-1] = 0.25 * (values[:-2] + 2.0 * values[1:-1] + values[2:])
    return np.asarray(out, dtype=float)


def _heading(x: np.ndarray, y: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        dx = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
        dy = 0.5 * (np.roll(y, -1) - np.roll(y, 1))
    else:
        dx = np.gradient(x)
        dy = np.gradient(y)
    return np.unwrap(np.arctan2(dy, dx))


def build_track(
    points: list[list[float]] | np.ndarray,
    *,
    closed: bool = True,
    ds_m: float | None = None,
    curvature_window_m: float = 3.0,
) -> Track:
    """Build a track from centreline points in metres.

    Parameters
    ----------
    points:
        ``[[x, y], ...]`` or ``[[x, y, z], ...]``. Coordinates are metres in a
        local frame; no geographic projection is applied.
    closed:
        Whether the last point connects back to the first.
    ds_m:
        Resampling step. Defaults to ``length / 1500`` clipped to 0.25-2 m.
    curvature_window_m:
        Half-width of the three-point curvature stencil.
    """
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise TrackError("points must be an (N, 2) or (N, 3) array of metres")
    if not np.all(np.isfinite(arr)):
        raise TrackError("points contain non-finite values")

    xy = arr[:, :2]
    z_in = arr[:, 2] if arr.shape[1] == 3 else None

    if closed and xy.shape[0] > 1:
        gap = float(np.hypot(xy[0, 0] - xy[-1, 0], xy[0, 1] - xy[-1, 1]))
        if gap < 1e-9:
            xy = xy[:-1]
            if z_in is not None:
                z_in = z_in[:-1]

    keep = np.ones(xy.shape[0], dtype=bool)
    if xy.shape[0] > 1:
        step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        keep[1:] = step > 1e-9
    if z_in is not None:
        z_in = np.asarray(z_in, dtype=float)[keep]
    xy = xy[keep]

    if xy.shape[0] < 8:
        raise TrackError("A centreline needs at least 8 distinct points")
    if not np.all(np.isfinite(xy)):
        raise TrackError("points contain non-finite values")

    x, y, z, ds, length = _resample(xy[:, 0], xy[:, 1], z_in, closed, ds_m)
    if length < 10.0:
        raise TrackError("Track length must be at least 10 m")

    window_m = max(0.5, curvature_window_m)
    window = max(1, int(round(window_m / ds)))
    kappa = _circumcircle_curvature(x, y, closed, window)
    kappa = _smooth(kappa, closed)

    return Track(
        x=x,
        y=y,
        z=z,
        closed=closed,
        ds_m=ds,
        length_m=length,
        heading_rad=_heading(x, y, closed),
        curvature_1pm=kappa,
    )
