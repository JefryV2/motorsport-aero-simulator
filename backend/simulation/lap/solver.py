"""Quasi-steady lap-time solver.

The speed profile is found the standard way: start from the cornering limit at
every point, then run forward (acceleration / power / traction) and backward
(braking) passes until the profile stops changing. The longitudinal and lateral
limits are combined with a g-g ellipse, so a tyre that is already using grip
sideways cannot also deliver full acceleration or braking.

Closed tracks wrap the passes; open tracks use an entry speed supplied by the
caller (never an assumed one) and report if braking makes that entry
infeasible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from backend.simulation.lap.track import Track
from backend.simulation.lap.vehicle import Vehicle


@dataclass(frozen=True)
class LapSettings:
    max_iterations: int = 60
    tolerance_mps: float = 1e-4
    speed_floor_mps: float = 0.5
    initial_speed_ms: float | None = None


@dataclass
class LapResult:
    lap_time_s: float
    sector_times_s: list[float]
    speed_ms: np.ndarray
    speed_kmh: np.ndarray
    s_m: np.ndarray
    curvature_1pm: np.ndarray
    ay_ms2: np.ndarray
    ax_ms2: np.ndarray
    downforce_N: np.ndarray
    drag_N: np.ndarray
    envelope_utilisation: np.ndarray
    segment_time_s: np.ndarray
    length_m: float
    ds_m: float
    top_speed_kmh: float
    min_speed_kmh: float
    mean_speed_kmh: float
    iterations: int
    converged: bool
    entry_speed_limited: bool
    max_utilisation: float
    aero_source: str
    aero_fidelity: str
    track_source: str
    provenance: dict = field(default_factory=dict)


def _cornering_limit(
    speed_cap: float, curvature: float, vehicle: Vehicle, aero_map
) -> float:
    """Speed at which lateral demand equals lateral grip at this point."""
    kappa = abs(curvature)
    if kappa < 1e-12:
        return speed_cap

    def demand_minus_grip(speed: float) -> float:
        aero = aero_map.at(speed)
        return vehicle.lateral_limit(speed, aero) - kappa * speed * speed

    if demand_minus_grip(speed_cap) >= 0.0:
        return speed_cap

    speed = min(speed_cap, math.sqrt(max(vehicle.lateral_limit(0.0, aero_map.at(0.0)), 1e-6) / kappa))
    for _ in range(40):
        aero = aero_map.at(speed)
        ay = vehicle.lateral_limit(speed, aero)
        new_speed = min(speed_cap, math.sqrt(max(ay, 1e-6) / kappa))
        if abs(new_speed - speed) < 1e-5:
            return new_speed
        speed = new_speed

    low, high = 0.0, speed_cap
    for _ in range(60):
        mid = 0.5 * (low + high)
        if demand_minus_grip(mid) > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _ellipse_factor(lateral_demand: float, lateral_limit: float) -> float:
    if lateral_limit <= 0.0:
        return 0.0
    ratio = lateral_demand / lateral_limit
    if ratio <= 0.0:
        return 1.0
    if ratio >= 1.0:
        return 0.0
    return math.sqrt(max(0.0, 1.0 - ratio * ratio))


def solve_lap(
    track: Track,
    vehicle: Vehicle,
    aero_map,
    settings: LapSettings | None = None,
) -> LapResult:
    """Compute the speed profile and lap time for one track and one setup."""
    settings = settings or LapSettings()
    n = track.count
    ds = track.ds_m
    kappa = track.curvature_1pm
    speed_cap = float(min(vehicle.max_speed_ms, 120.0))
    floor = float(settings.speed_floor_mps)

    v_lat = np.array(
        [_cornering_limit(speed_cap, kappa[i], vehicle, aero_map) for i in range(n)],
        dtype=float,
    )
    speed = np.minimum(v_lat, speed_cap)
    speed = np.maximum(speed, floor)

    entry_limited = False
    if track.closed:
        start = int(np.argmin(speed))
        order = [(start + k) % n for k in range(n)]
    else:
        order = list(range(n))
        entry = v_lat[0] if settings.initial_speed_ms is None else settings.initial_speed_ms
        entry = min(float(max(entry, 0.0)), v_lat[0], speed_cap)
        speed[0] = max(entry, floor)

    iterations = 0
    converged = False
    for sweep in range(1, settings.max_iterations + 1):
        iterations = sweep
        changed = 0.0

        for idx in order:
            j = (idx + 1) % n if track.closed else idx + 1
            if not track.closed and j >= n:
                break
            vi = speed[idx]
            aero = aero_map.at(vi)
            lateral = vi * vi * abs(kappa[idx])
            lateral_limit = vehicle.lateral_limit(vi, aero)
            accel = vehicle.accel_limit(vi, aero) * _ellipse_factor(
                lateral, lateral_limit
            )
            reachable = math.sqrt(max(vi * vi + 2.0 * accel * ds, 0.0))
            if reachable < speed[j] - settings.tolerance_mps:
                changed = max(changed, speed[j] - reachable)
                speed[j] = max(reachable, floor)

        for idx in reversed(order):
            j = (idx + 1) % n if track.closed else idx + 1
            if not track.closed and j >= n:
                continue
            vj = speed[j]
            aero = aero_map.at(vj)
            lateral = vj * vj * abs(kappa[j])
            lateral_limit = vehicle.lateral_limit(vj, aero)
            decel = vehicle.brake_limit(vj, aero) * _ellipse_factor(
                lateral, lateral_limit
            )
            reachable = math.sqrt(max(vj * vj + 2.0 * decel * ds, 0.0))
            if reachable < speed[idx] - settings.tolerance_mps:
                changed = max(changed, speed[idx] - reachable)
                speed[idx] = max(reachable, floor)

        if changed <= settings.tolerance_mps:
            converged = True
            break

    if not track.closed and settings.initial_speed_ms is not None:
        entry_limited = speed[0] < settings.initial_speed_ms - 1e-3

    # Segment times: constant-acceleration time over a segment is ds / mean(v).
    segment_speed = np.empty(track.segment_count, dtype=float)
    segment_ax = np.empty(track.segment_count, dtype=float)
    segment_time = np.empty(track.segment_count, dtype=float)
    for i in range(track.segment_count):
        j = (i + 1) % n if track.closed else i + 1
        mean_speed = 0.5 * (speed[i] + speed[j])
        segment_speed[i] = mean_speed
        segment_time[i] = ds / max(mean_speed, floor)
        segment_ax[i] = (speed[j] * speed[j] - speed[i] * speed[i]) / (2.0 * ds)

    lap_time = float(segment_time.sum())
    sectors = np.array_split(np.arange(track.segment_count), 3)
    sector_times = [float(segment_time[idx].sum()) for idx in sectors if idx.size]

    ax = np.empty(n, dtype=float)
    for i in range(n):
        ax[i] = segment_ax[i] if i < track.segment_count else segment_ax[-1]

    ay = speed * speed * kappa

    downforce = np.array([aero_map.at(v).downforce_N for v in speed], dtype=float)
    drag = np.array([aero_map.at(v).drag_N for v in speed], dtype=float)

    utilisation = np.zeros(n, dtype=float)
    for i in range(n):
        aero = aero_map.at(speed[i])
        lateral_limit = vehicle.lateral_limit(speed[i], aero)
        if ax[i] >= 0.0:
            long_limit = vehicle.accel_limit(speed[i], aero)
        else:
            long_limit = vehicle.brake_limit(speed[i], aero)
        long_ratio = 0.0 if long_limit <= 0.0 else abs(ax[i]) / long_limit
        lat_ratio = 0.0 if lateral_limit <= 0.0 else abs(ay[i]) / lateral_limit
        utilisation[i] = long_ratio * long_ratio + lat_ratio * lat_ratio

    return LapResult(
        lap_time_s=lap_time,
        sector_times_s=sector_times,
        speed_ms=speed,
        speed_kmh=speed * 3.6,
        s_m=np.arange(n, dtype=float) * ds,
        curvature_1pm=kappa,
        ay_ms2=ay,
        ax_ms2=ax,
        downforce_N=downforce,
        drag_N=drag,
        envelope_utilisation=utilisation,
        segment_time_s=segment_time,
        length_m=track.length_m,
        ds_m=ds,
        top_speed_kmh=float(speed.max() * 3.6),
        min_speed_kmh=float(speed.min() * 3.6),
        mean_speed_kmh=float(track.length_m / lap_time * 3.6),
        iterations=iterations,
        converged=converged,
        entry_speed_limited=entry_limited,
        max_utilisation=float(utilisation.max()),
        aero_source=str(getattr(aero_map, "source", "aero map")),
        aero_fidelity=str(getattr(aero_map, "fidelity", "unknown")),
        track_source=track.source,
        provenance={
            "model": "quasi-steady point-mass lap simulation",
            "speed_profile": "forward/backward passes with a g-g ellipse",
            "tyre_model": (
                "load-sensitive friction ellipse per axle; lateral transfer split "
                "by roll stiffness; inside-wheel lift included"
            ),
            "aero_coupling": (
                "downforce/drag from the live aero model, scaled with dynamic "
                "pressure; CoP split sets front/rear load"
            ),
            "not_modelled": [
                "transient tyre dynamics and relaxation length",
                "suspension compliance, damping and bump stops",
                "differential and traction-control behaviour",
                "driver model and reaction time",
                "fuel mass, tyre wear and temperature",
                "DRS zones, braking lock-up and aero map variation over a lap",
            ],
            "track_source": track.source,
        },
    )
