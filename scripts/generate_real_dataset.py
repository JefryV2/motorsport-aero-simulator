"""Generate a small, real-CFD high-fidelity dataset for the surrogate study.

The old 100-row dataset was produced by the calibrated synthetic fallback. This
script replaces it with a deliberately smaller LHS design evaluated by the real
OpenFOAM hybrid path. Failed points are retained with NaN HF targets and a
``hf_failed`` flag rather than being filled with invented numbers.

Typical runtime is roughly 20-30 minutes for 24-28 medium-mesh points on WSL.
The run is serial by default because each OpenFOAM case already uses multiple
cores; launching several at once makes every case slower and less reliable.

Usage:
    .venv\\Scripts\\python.exe scripts\\generate_real_dataset.py --samples 28
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from backend.ml.dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    generate_surrogate_dataset,
    save_dataset_parquet,
    summarize_provenance,
)
from backend.simulation.cfd.models import MeshLevel  # noqa: E402
from backend.simulation.cfd.runner import detect_cfd_backend, get_solver_version  # noqa: E402

BACKUP_PATH = Path("data/datasets/aero_surrogate_v1_synthetic_backup.parquet")
MANIFEST_PATH = Path("data/datasets/aero_surrogate_v1_provenance.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=28, help="number of LHS points")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mesh-level",
        choices=[level.value for level in MeshLevel],
        default=MeshLevel.MEDIUM.value,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="parallel CFD cases; keep at 1 unless the solver host is dedicated",
    )
    args = parser.parse_args()

    backend = detect_cfd_backend()
    if backend not in {"wsl", "native"}:
        print(
            "No OpenFOAM backend detected. Refusing to overwrite the dataset with "
            "synthetic fallback rows. Install OpenFOAM first.",
            file=sys.stderr,
        )
        return 2

    if DEFAULT_DATASET_PATH.exists() and not BACKUP_PATH.exists():
        BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DEFAULT_DATASET_PATH, BACKUP_PATH)
        print(f"Backed up existing dataset to {BACKUP_PATH}")

    solver_version = get_solver_version()
    started = time.time()
    print(
        f"Generating {args.samples} real-CFD points | backend={backend} | "
        f"solver={solver_version} | mesh={args.mesh_level} | workers={args.max_workers}",
        flush=True,
    )

    df = generate_surrogate_dataset(
        n_samples=args.samples,
        seed=args.seed,
        max_workers=max(1, args.max_workers),
        mesh_level=MeshLevel(args.mesh_level),
    )
    elapsed_s = time.time() - started
    save_dataset_parquet(df, file_path=DEFAULT_DATASET_PATH)

    provenance = summarize_provenance(df)
    manifest = {
        "dataset_path": str(DEFAULT_DATASET_PATH),
        "backup_path": str(BACKUP_PATH),
        "created_at_epoch": time.time(),
        "elapsed_s": round(elapsed_s, 1),
        "n_samples_requested": args.samples,
        "seed": args.seed,
        "mesh_level": args.mesh_level,
        "sampling": {
            "design_space": "DESIGN_SPACE (backend/ml/dataset.py)",
            "cfd_validated": True,
            "total_incidence_envelope_deg": 8.0,
            "ride_height_mm": [71.0, 150.0],
            "note": (
                "Element angles re-drawn so main+flap <= 8 deg and ride height "
                "71-150 mm (h/c 0.15-0.32): the envelope V5/V6 measured on the "
                "O-grid family. Out-of-envelope designs are not measured."
            ),
        },
        "backend": backend,
        "solver_version": solver_version,
        "provenance": provenance,
        "columns": list(df.columns),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")

    print("\n=== Real-CFD dataset complete ===")
    print(f"Rows: {len(df)} | usable HF: {provenance['n_used']} | failed: {provenance['n_failed']}")
    print(f"HF sources: {provenance['solver_counts']}")
    print(f"Converged HF rows: {provenance['n_converged']}")
    print(f"Saved: {DEFAULT_DATASET_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Elapsed: {elapsed_s / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
