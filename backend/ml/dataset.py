"""Aerodynamic dataset generator and Parquet storage using Latin Hypercube DoE.

Generates paired low-fidelity (VLM) and high-fidelity (CFD) evaluations across
the bounded formula aero design space with complete provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from backend.models.aero_config import (
    FormulaAeroConfig,
    WingElement,
)
from backend.simulation.cfd.openfoam_runner import run_cfd_with_provenance
from backend.simulation.panel_method import simulate_panel


@dataclass(frozen=True)
class ParameterBound:
    name: str
    min_val: float
    max_val: float
    unit: str
    description: str


DESIGN_SPACE: list[ParameterBound] = [
    ParameterBound("fw_main_angle", 0.0, 8.0, "deg", "Front wing main element angle"),
    ParameterBound("fw_flap_angle", 4.0, 20.0, "deg", "Front wing flap angle"),
    ParameterBound("ride_height_front_mm", 15.0, 50.0, "mm", "Front ride height"),
    ParameterBound("rear_wing_main_angle", 5.0, 25.0, "deg", "Rear wing main element angle"),
    ParameterBound("speed_kmh", 100.0, 300.0, "km/h", "Freestream vehicle speed"),
]

FEATURE_COLUMNS = [p.name for p in DESIGN_SPACE]
LF_TARGETS = ["cl_lf", "cd_lf", "df_total_lf", "drag_total_lf", "balance_lf"]
HF_TARGETS = ["cl_hf", "cd_hf", "df_total_hf", "drag_total_hf", "balance_hf"]

DEFAULT_DATASET_PATH = Path("data/datasets/aero_surrogate_v1.parquet")


def sample_design_space(n_samples: int = 100, seed: int = 42) -> np.ndarray:
    """Generate Latin Hypercube samples over the bounded design space.

    Returns:
        np.ndarray of shape (n_samples, len(DESIGN_SPACE)) in physical units.
    """
    sampler = qmc.LatinHypercube(d=len(DESIGN_SPACE), seed=seed)
    unit_samples = sampler.random(n=n_samples)

    lower_bounds = np.array([p.min_val for p in DESIGN_SPACE])
    upper_bounds = np.array([p.max_val for p in DESIGN_SPACE])

    physical_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)
    return np.asarray(physical_samples, dtype=float)


def sample_vector_to_config(sample: np.ndarray | dict[str, float]) -> FormulaAeroConfig:
    """Map a design parameter vector or dict to a typed FormulaAeroConfig."""
    if isinstance(sample, np.ndarray):
        values = dict(zip(FEATURE_COLUMNS, sample.tolist(), strict=True))
    else:
        values = sample

    fw_main = WingElement(
        angle_deg=float(values["fw_main_angle"]),
        chord_m=0.35,
        span_m=1.80,
    )
    fw_flap = WingElement(
        angle_deg=float(values["fw_flap_angle"]),
        chord_m=0.12,
        span_m=1.75,
    )
    rw_main = WingElement(
        angle_deg=float(values["rear_wing_main_angle"]),
        chord_m=0.40,
        span_m=0.95,
    )
    rw_flap = WingElement(
        angle_deg=12.0,
        chord_m=0.15,
        span_m=0.90,
    )

    return FormulaAeroConfig(
        speed_kmh=float(values["speed_kmh"]),
        ride_height_front_mm=float(values["ride_height_front_mm"]),
        ride_height_rear_mm=float(values["ride_height_front_mm"]) + 15.0,  # 15mm rake
        front_wing_main=fw_main,
        front_wing_flap=fw_flap,
        rear_wing_main=rw_main,
        rear_wing_flap=rw_flap,
    )


def compute_config_hash(values: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint for a parameter configuration."""
    raw = json.dumps(values, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evaluate_design_point(
    sample_values: dict[str, float],
    mesh_level: str = "medium",
) -> dict[str, Any]:
    """Evaluate both low-fidelity (VLM) and high-fidelity (CFD-family) physics at a design point.

    Provenance columns are read from the run that actually executed. When no
    OpenFOAM binary is available the high-fidelity column is explicitly marked
    as a synthetic model, never as an OpenFOAM result.
    """
    config = sample_vector_to_config(sample_values)
    config_hash = compute_config_hash(sample_values)

    # 1. Low-Fidelity: 3D VLM with ground mirror and Trefftz induced drag
    vlm_result = simulate_panel(config)
    lf_forces = vlm_result.forces

    # 2. High-Fidelity: OpenFOAM when available, otherwise a labelled surrogate
    cfd_result, provenance = run_cfd_with_provenance(config)
    hf_forces = cfd_result.forces

    record: dict[str, Any] = {
        "config_hash": config_hash,
        "timestamp_epoch": time.time(),
        "mesh_level": mesh_level,
        "solver_version": provenance.solver_version,
        "solver_used": provenance.solver_used,
        "hf_fidelity": provenance.fidelity,
        "is_synthetic": provenance.is_synthetic,
        "convergence_flag": provenance.is_converged,
        "convergence_status": provenance.convergence_status,
        # Feature columns
        **sample_values,
        # LF Targets
        "cl_lf": lf_forces.cl_total,
        "cd_lf": lf_forces.cd_total,
        "df_total_lf": lf_forces.downforce_total_N,
        "drag_total_lf": lf_forces.drag_N,
        "balance_lf": lf_forces.balance_front_pct,
        # HF Targets
        "cl_hf": hf_forces.cl_total,
        "cd_hf": hf_forces.cd_total,
        "df_total_hf": hf_forces.downforce_total_N,
        "drag_total_hf": hf_forces.drag_N,
        "balance_hf": hf_forces.balance_front_pct,
    }
    return record


def generate_surrogate_dataset(
    n_samples: int = 100,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_workers: int = 6,
) -> pd.DataFrame:
    """Generate a multi-fidelity dataset using Latin Hypercube Sampling.

    Assigns deterministic train / val / test splits.
    """
    samples = sample_design_space(n_samples=n_samples, seed=seed)

    def _eval(i: int) -> dict[str, Any]:
        val_dict = dict(zip(FEATURE_COLUMNS, samples[i].tolist(), strict=True))
        rec = evaluate_design_point(val_dict)
        rec["sample_id"] = i
        return rec

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(_eval, range(n_samples)))

    df = pd.DataFrame(records)

    # Deterministic split assignment using RNG permutation
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))

    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)

    split = np.empty(len(df), dtype=object)
    split[indices[:n_train]] = "train"
    split[indices[n_train : n_train + n_val]] = "val"
    split[indices[n_train + n_val :]] = "test"
    df["split"] = split

    return df


def save_dataset_parquet(df: pd.DataFrame, file_path: Path = DEFAULT_DATASET_PATH) -> Path:
    """Save aerodynamic dataset to Apache Parquet columnar format."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, engine="pyarrow", index=False)
    return file_path


def load_dataset_parquet(file_path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load aerodynamic dataset from Apache Parquet format."""
    if not file_path.exists():
        raise FileNotFoundError(f"Aerodynamic dataset not found at {file_path}")
    return pd.read_parquet(file_path, engine="pyarrow")


def summarize_provenance(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise where the high-fidelity column actually came from.

    Used by validation scripts so a synthetic dataset can never be silently
    reported as an OpenFOAM study.
    """
    if "solver_used" not in df.columns:
        return {
            "has_provenance": False,
            "all_synthetic": None,
            "solver_counts": {},
            "warning": "Legacy dataset without provenance columns; treat HF as unknown.",
        }
    counts = {str(k): int(v) for k, v in df["solver_used"].value_counts().items()}
    is_synth = df.get("is_synthetic")
    all_synthetic = bool(is_synth.all()) if is_synth is not None else False
    converged = df.get("convergence_flag")
    return {
        "has_provenance": True,
        "all_synthetic": all_synthetic,
        "solver_counts": counts,
        "n_rows": int(len(df)),
        "n_converged": int(converged.fillna(False).astype(bool).sum()) if converged is not None else None,
        "warning": (
            "HIGH-FIDELITY COLUMN IS A SYNTHETIC MODEL: results validate the "
            "surrogate machinery against a deterministic generator, not physics."
            if all_synthetic
            else None
        ),
    }


def get_or_generate_dataset(
    file_path: Path = DEFAULT_DATASET_PATH,
    n_samples: int = 120,
    seed: int = 42,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """Get existing cached Parquet dataset or generate a new one if missing."""
    if file_path.exists() and not force_regenerate:
        return load_dataset_parquet(file_path)

    df = generate_surrogate_dataset(n_samples=n_samples, seed=seed)
    save_dataset_parquet(df, file_path=file_path)
    return df
