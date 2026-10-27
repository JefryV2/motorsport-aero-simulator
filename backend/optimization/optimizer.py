"""Bayesian optimization for formula car aero setup."""

from __future__ import annotations

import copy

import optuna

from backend.models.aero_config import (
    AeroForces,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
)
from backend.simulation.analytical import simulate
from backend.simulation.panel_method import simulate_panel


def _objective_score(forces: AeroForces, objective: OptimizationObjective, target_balance: float) -> float:
    """Convert forces to a scalar to maximize."""
    if objective == OptimizationObjective.MAX_DOWNFORCE:
        return float(forces.downforce_total_N)
    if objective == OptimizationObjective.MIN_DRAG:
        return float(-forces.drag_N)
    if objective == OptimizationObjective.MAX_EFFICIENCY:
        return float(forces.efficiency)
    # Balanced: downforce + efficiency + balance penalty
    balance_penalty = abs(forces.balance_front_pct - target_balance) * 5.0
    return float(forces.downforce_total_N * 0.6 + forces.efficiency * 200 - balance_penalty)


def optimize(request: OptimizationRequest, use_panel: bool = True) -> OptimizationResult:
    """Run Bayesian optimization over aero parameters."""
    baseline = simulate_panel(request.config) if use_panel else simulate(request.config)
    baseline_score = _objective_score(
        baseline.forces, request.objective, request.target_balance_pct or 42.0
    )

    sim_fn = simulate_panel if use_panel else simulate
    target_balance = request.target_balance_pct or 42.0
    pareto: list[dict] = []

    def optuna_objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(request.config)
        cfg.front_wing_main.angle_deg = trial.suggest_float("fw_main_angle", -2.0, 8.0)
        cfg.front_wing_flap.angle_deg = trial.suggest_float("fw_flap_angle", 2.0, 20.0)
        cfg.rear_wing_main.angle_deg = trial.suggest_float("rw_main_angle", 0.0, 12.0)
        cfg.rear_wing_flap.angle_deg = trial.suggest_float("rw_flap_angle", 5.0, 30.0)
        cfg.ride_height_front_mm = trial.suggest_float("rh_front", 10.0, 50.0)
        cfg.ride_height_rear_mm = trial.suggest_float("rh_rear", 20.0, 70.0)
        cfg.rake_deg = trial.suggest_float("rake", 0.0, 3.5)
        cfg.diffuser_angle_deg = trial.suggest_float("diffuser_angle", 8.0, 18.0)
        cfg.diffuser_ride_height_mm = trial.suggest_float("diffuser_rh", 15.0, 50.0)
        cfg.front_wing_main.gurney_mm = trial.suggest_float("fw_gurney", 0.0, 30.0)
        cfg.rear_wing_main.gurney_mm = trial.suggest_float("rw_gurney", 0.0, 20.0)

        result = sim_fn(cfg)
        forces = result.forces

        if request.max_drag_N and forces.drag_N > request.max_drag_N:
            return -1e6

        score = _objective_score(forces, request.objective, target_balance)

        if len(pareto) < 50:
            pareto.append({
                "drag_N": forces.drag_N,
                "downforce_N": forces.downforce_total_N,
                "efficiency": forces.efficiency,
                "balance_pct": forces.balance_front_pct,
                "score": score,
            })

        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(optuna_objective, n_trials=request.n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_cfg = copy.deepcopy(request.config)
    best_cfg.front_wing_main.angle_deg = best_params["fw_main_angle"]
    best_cfg.front_wing_flap.angle_deg = best_params["fw_flap_angle"]
    best_cfg.rear_wing_main.angle_deg = best_params["rw_main_angle"]
    best_cfg.rear_wing_flap.angle_deg = best_params["rw_flap_angle"]
    best_cfg.ride_height_front_mm = best_params["rh_front"]
    best_cfg.ride_height_rear_mm = best_params["rh_rear"]
    best_cfg.rake_deg = best_params["rake"]
    best_cfg.diffuser_angle_deg = best_params["diffuser_angle"]
    best_cfg.diffuser_ride_height_mm = best_params["diffuser_rh"]
    best_cfg.front_wing_main.gurney_mm = best_params["fw_gurney"]
    best_cfg.rear_wing_main.gurney_mm = best_params["rw_gurney"]

    best_result = sim_fn(best_cfg)
    best_score = _objective_score(best_result.forces, request.objective, target_balance)
    improvement = ((best_score - baseline_score) / abs(baseline_score) * 100) if baseline_score else 0.0

    return OptimizationResult(
        best_config=best_cfg,
        best_forces=best_result.forces,
        improvement_pct=round(improvement, 2),
        trials_run=request.n_trials,
        pareto_front=sorted(pareto, key=lambda x: x["score"], reverse=True)[:20],
    )
