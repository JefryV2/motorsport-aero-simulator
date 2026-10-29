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


def _objective_score(
    forces: AeroForces, objective: OptimizationObjective, target_balance: float
) -> float:
    """Convert forces to a scalar to maximize.

    The balanced objective is a weighted sum of *normalised* terms. Without
    normalisation the raw downforce in newtons dominates efficiency and the
    balance target by orders of magnitude, so "balanced" collapses into
    "maximum downforce". The weights here are a design choice, not a claim of
    physical optimality: downforce 35%, efficiency 25%, balance 40% with a
    25-percentage-point tolerance around the target.
    """
    if objective == OptimizationObjective.MAX_DOWNFORCE:
        return float(forces.downforce_total_N)
    if objective == OptimizationObjective.MIN_DRAG:
        return float(-forces.drag_N)
    if objective == OptimizationObjective.MAX_EFFICIENCY:
        return float(forces.efficiency)

    downforce_term = forces.downforce_total_N / 3000.0
    efficiency_term = forces.efficiency / 3.0
    balance_error = abs(forces.balance_front_pct - target_balance)
    balance_term = max(0.0, 1.0 - balance_error / 25.0)
    return float(0.35 * downforce_term + 0.25 * efficiency_term + 0.40 * balance_term)


def _pareto_front(points: list[dict], limit: int = 20) -> list[dict]:
    """Return the non-dominated drag/downforce points.

    The previous implementation sorted the first 50 trials by score and called
    that a Pareto front. This filters points that are actually dominated on the
    drag/downforce plane, then sorts the front by drag for plotting.
    """
    front: list[dict] = []
    seen: set[tuple[float, float]] = set()
    for point in points:
        key = (round(point["drag_N"], 1), round(point["downforce_N"], 1))
        if key in seen:
            continue
        dominated = any(
            other["downforce_N"] >= point["downforce_N"]
            and other["drag_N"] <= point["drag_N"]
            and (
                other["downforce_N"] > point["downforce_N"]
                or other["drag_N"] < point["drag_N"]
            )
            for other in points
        )
        if not dominated:
            seen.add(key)
            front.append(point)
    return sorted(front, key=lambda item: (item["drag_N"], -item["downforce_N"]))[:limit]


def optimize(request: OptimizationRequest, use_panel: bool = True) -> OptimizationResult:
    """Run Bayesian optimization over aero parameters."""
    baseline = simulate_panel(request.config) if use_panel else simulate(request.config)
    baseline_score = _objective_score(
        baseline.forces, request.objective, request.target_balance_pct or 42.0
    )

    sim_fn = simulate_panel if use_panel else simulate
    target_balance = request.target_balance_pct or 42.0
    trial_points: list[dict] = []

    def optuna_objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(request.config)
        cfg.front_wing_main.angle_deg = trial.suggest_float("fw_main_angle", 0.0, 8.0)
        cfg.front_wing_flap.angle_deg = trial.suggest_float("fw_flap_angle", 4.0, 20.0)
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
        trial_points.append(
            {
                "drag_N": forces.drag_N,
                "downforce_N": forces.downforce_total_N,
                "efficiency": forces.efficiency,
                "balance_pct": forces.balance_front_pct,
                "score": score,
            }
        )
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
    improvement = (
        ((best_score - baseline_score) / abs(baseline_score) * 100) if baseline_score else 0.0
    )
    balances = [point["balance_pct"] for point in trial_points]
    achievable_balance = (
        [round(min(balances), 1), round(max(balances), 1)] if balances else []
    )
    balanced_objective = request.objective == OptimizationObjective.BALANCED
    target_reachable = (
        bool(achievable_balance and achievable_balance[0] <= target_balance <= achievable_balance[1])
        if balanced_objective
        else None
    )

    return OptimizationResult(
        best_config=best_cfg,
        best_forces=best_result.forces,
        improvement_pct=round(improvement, 2),
        trials_run=request.n_trials,
        pareto_front=_pareto_front(trial_points),
        target_balance_pct=target_balance if balanced_objective else None,
        achievable_balance_pct=achievable_balance,
        target_reachable=target_reachable,
    )
