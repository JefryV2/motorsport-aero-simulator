"""Vortex Lattice Method (VLM) physics package for aerodynamic simulation."""

from backend.simulation.vlm.core import VLMSolution, VLMSurface, solve_vlm

__all__ = ["VLMSurface", "VLMSolution", "solve_vlm"]
