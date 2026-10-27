"""Verify B1: panel_method produces identical results to analytical."""
from backend.simulation.analytical import simulate
from backend.simulation.panel_method import simulate_panel
from backend.models.aero_config import FormulaAeroConfig

cfg = FormulaAeroConfig()
r1 = simulate(cfg)
r2 = simulate_panel(cfg)

fields = [
    "drag_N", "downforce_total_N", "downforce_front_N",
    "downforce_rear_N", "balance_front_pct", "cop_x_m",
    "cl_total", "cd_total", "efficiency",
]

all_same = True
for f in fields:
    v1 = getattr(r1.forces, f)
    v2 = getattr(r2.forces, f)
    status = "SAME" if v1 == v2 else "DIFF"
    if v1 != v2:
        all_same = False
    print(f"  {f}: {v1} vs {v2} -> {status}")

print(f"  lap_delta: {r1.lap_time_delta_s} vs {r2.lap_time_delta_s}")
if r1.lap_time_delta_s != r2.lap_time_delta_s:
    all_same = False
print(f"\nB1 CONFIRMED: panel_method == analytical: {all_same}")
