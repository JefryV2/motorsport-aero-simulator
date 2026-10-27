"""OpenFOAM CFD integration scaffold.

Requires OpenFOAM installed and on PATH. Generates case files,
runs simpleFoam/pimpleFoam, and parses force coefficients.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from backend.models.aero_config import FormulaAeroConfig, SimulationResult
from backend.simulation.analytical import simulate as simulate_analytical

CASE_TEMPLATE = Path(__file__).parent / "templates" / "formula_car"
WORK_DIR = Path(__file__).parent.parent.parent / "data" / "cfd_cases"


def is_openfoam_available() -> bool:
    """Check if OpenFOAM is installed."""
    return shutil.which("simpleFoam") is not None or shutil.which("foamRun") is not None


def generate_case(config: FormulaAeroConfig, case_id: str) -> Path:
    """Generate OpenFOAM case directory from config parameters."""
    case_dir = WORK_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "front_wing_angle": config.front_wing_main.angle_deg + config.front_wing_flap.angle_deg,
        "rear_wing_angle": config.rear_wing_main.angle_deg + config.rear_wing_flap.angle_deg,
        "ride_height_front": config.ride_height_front_mm / 1000.0,
        "ride_height_rear": config.ride_height_rear_mm / 1000.0,
        "speed_ms": config.speed_kmh / 3.6,
        "rake_deg": config.rake_deg,
    }

    (case_dir / "aero_config.json").write_text(json.dumps(meta, indent=2))

    # Write system/controlDict stub
    system_dir = case_dir / "system"
    system_dir.mkdir(exist_ok=True)
    (system_dir / "controlDict").write_text(
        f"""FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     simpleFoam;
startFrom       startTime;
stopAt          endTime;
endTime         500;
deltaT          1;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    )

    return case_dir


def run_cfd(config: FormulaAeroConfig, case_id: str = "latest") -> SimulationResult:
    """
    Run CFD simulation. Falls back to analytical if OpenFOAM unavailable.
    """
    if not is_openfoam_available():
        result = simulate_analytical(config, method="analytical_fallback")
        result.method = "cfd_stub_unavailable_fallback_analytical"
        return result

    case_dir = generate_case(config, case_id)

    try:
        subprocess.run(
            ["simpleFoam"],
            cwd=case_dir,
            check=True,
            capture_output=True,
            timeout=3600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        result = simulate_analytical(config, method="analytical_fallback")
        result.method = "cfd_stub_failed_fallback_analytical"
        return result

    # Parse force coefficients from postProcessing (stub — extend with actual parser)
    result = simulate_analytical(config, method="cfd_stub")
    result.method = "cfd_stub"
    return result
