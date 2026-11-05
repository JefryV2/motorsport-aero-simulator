"""Lap simulation: track geometry, vehicle model and quasi-steady solver."""

from backend.simulation.lap.aero_map import AeroMap, AeroPoint, ConstantAeroMap
from backend.simulation.lap.solver import LapResult, LapSettings, solve_lap
from backend.simulation.lap.track import Track, TrackError, build_track
from backend.simulation.lap.track_io import parse_centerline_csv
from backend.simulation.lap.vehicle import Vehicle

__all__ = [
    "AeroMap",
    "AeroPoint",
    "ConstantAeroMap",
    "LapResult",
    "LapSettings",
    "Track",
    "TrackError",
    "Vehicle",
    "build_track",
    "parse_centerline_csv",
    "solve_lap",
]
