"""Validation Case V9: Active learning vs random vs space-filling at equal budget.

Phase 4 (P4.1/P4.2). Measures whether uncertainty-driven acquisition of
high-fidelity evaluations beats random and space-filling (Sobol/maximin)
selection **on real data**: the high-fidelity oracle is the O-grid envelope
dataset's rows (`aero_surrogate_v1.parquet`, all OpenFOAM WSL hybrid solves).
No synthetic values are substituted anywhere.

Two honest modes are measured:

- ``lenient`` (comparable with V7/V8): every pool row can return its measured
  HF value when queried, converged or not.
- ``strict``: only the 13 converged rows return data; querying one of the 15
  force-unstable designs consumes HF budget and returns nothing, exactly as a
  real CFD run at that design would. The acquisition's usable yield is then a
  measured outcome, not an assumption.

The historical claim under test is printed as a scorecard, but nothing is
forced: if active learning does not win, that measured result is the answer
(BLUEPRINT Phase 4: "Report the result even if active learning does not win").

Learning curves are test-RMSE (Cl) vs number of HF *attempts* on a fixed
deterministic test set of converged rows. Random acquisition is averaged over
several seeds; AL and space-filling are deterministic and get confidence
context from the random band.

Usage:
    .venv\\Scripts\\python.exe validation\\v9_active_learning.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from backend.ml.dataset import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    get_or_generate_dataset,
    summarize_provenance,
)
from backend.ml.metrics import compute_metrics
from backend.ml.surrogate import AR1MultiFidelitySurrogate

RANDOM_SEEDS = (7, 11, 23, 29, 31, 37, 41, 43, 53, 59, 61, 67)
TEST_SIZE = 5
INITIAL_HF = 2
BUDGETS = (4, 6, 9, 13)
CURVE_JSON = Path("data/datasets/v9_learning_curves.json")


def _fit_ar1(
    x: np.ndarray, lf: np.ndarray, hf: np.ndarray, seed: int
) -> AR1MultiFidelitySurrogate:
    mf = AR1MultiFidelitySurrogate(n_restarts_optimizer=2, random_state=seed)
    mf.fit(x, lf, hf)
    return mf


def _run_scheme(
    scheme: str,
    x_pool: np.ndarray,
    lf_pool: np.ndarray,
    oracle_pool: np.ndarray,  # nan where the oracle cannot return data (strict mode)
    x_test: np.ndarray,
    lf_test: np.ndarray,
    hf_test: np.ndarray,
    rng: np.random.Generator,
    n_restarts_seed: int,
) -> dict[int, dict[str, float]]:
    """Acquire HF points one at a time; evaluate the model at each checkpoint.

    Args:
        scheme: ``al`` (max predictive variance), ``random``, ``maximin``.
    """
    n_pool = len(x_pool)
    selected: list[int] = []

    def _select_next(excluded: set[int]) -> int | None:
        cand = [i for i in range(n_pool) if i not in excluded]
        if not cand:
            return None
        if scheme == "random":
            return int(cand[int(rng.integers(len(cand)))])
        if scheme == "maximin":
            if not selected:
                return int(cand[0])
            pts = x_pool[cand] / np.maximum(np.ptp(x_pool, axis=0), 1e-9)
            base = x_pool[selected] / np.maximum(np.ptp(x_pool, axis=0), 1e-9)
            dists = np.sqrt(((pts[:, None, :] - base[None, :, :]) ** 2).sum(-1)).min(axis=1)
            return int(cand[int(np.argmax(dists))])
        # scheme == "al": pick the candidate with the largest AR1 predictive
        # std for the HF target at that design.
        hist = [i for i in selected if np.isfinite(oracle_pool[i])]
        if hist:
            mf = _fit_ar1(
                x_pool[hist], lf_pool[hist], oracle_pool[hist], n_restarts_seed
            )
            _, std = mf.predict(x_pool[cand], lf_pool[cand])
            order = sorted(range(len(cand)), key=lambda j: (-float(std[j]), cand[j]))
            return int(cand[order[0]])
        return int(cand[0])

    selected = []
    excluded: set[int] = set()
    # Same deterministic initial design for every scheme at a given pool, so the
    # comparison is like-for-like: the schemes differ only in what they add.
    rng_initial = np.random.default_rng(42)
    for _ in range(INITIAL_HF):
        cand = [i for i in range(n_pool) if i not in excluded]
        pick = int(cand[int(rng_initial.integers(len(cand)))])
        selected.append(pick)
        excluded.add(pick)

    curve: dict[int, dict[str, float]] = {}
    attempts = -1
    for attempt in range(INITIAL_HF + 1, max(BUDGETS) + 1):
        next_pick = _select_next(excluded)
        if next_pick is None:
            break
        attempts = attempt
        selected.append(next_pick)
        excluded.add(next_pick)

        n_obs = int(sum(np.isfinite(oracle_pool[i]) for i in selected))
        if n_obs >= 2:
            obs = [i for i in selected if np.isfinite(oracle_pool[i])]
            mf = _fit_ar1(x_pool[obs], lf_pool[obs], oracle_pool[obs], n_restarts_seed)
            pred, _ = mf.predict(x_test, lf_test)
            metrics = compute_metrics(np.asarray(hf_test), np.asarray(pred))
        else:
            metrics = None
        if attempt in BUDGETS:
            curve[attempt] = {
                "rmse_cl": float(metrics.rmse) if metrics else float("nan"),
                "hf_observed": float(n_obs),
            }
    if BUDGETS and max(BUDGETS) not in curve:
        curve[max(BUDGETS)] = {
            "rmse_cl": float("nan"),
            "hf_observed": float(n_obs) if attempts >= 0 else 0.0,
        }
    return curve


def run_v9_validation() -> dict[str, float | str | bool]:
    """Run Case V9 and print the learning curves and scorecard."""
    print("=== Case V9: Active learning vs random vs space-filling (equal HF budget) ===")
    df = get_or_generate_dataset(file_path=DEFAULT_DATASET_PATH, n_samples=24, seed=42)
    provenance = summarize_provenance(df)
    print(f"Dataset rows: {len(df)} | HF provenance: {provenance['solver_counts']}")

    if provenance.get("all_synthetic"):
        print(
            "STOP: the HF column is synthetic. V9 measures acquisition strategy "
            "on real solver output only.",
        )
        return {"targets_met": 0.0, "hf_source": "synthetic_model"}

    converged = df["convergence_flag"].astype(bool).to_numpy()

    # Deterministic test set: converged rows only, so every scheme is scored on
    # the same oracle targets.
    rng = np.random.default_rng(42)
    converged_idx = np.flatnonzero(converged)
    shuffle = rng.permutation(len(converged_idx))
    test_idx = sorted(converged_idx[shuffle[:TEST_SIZE]].tolist())
    pool_idx = [i for i in range(len(df)) if i not in set(test_idx)]

    x = df[FEATURE_COLUMNS].to_numpy()
    lf = df["cl_lf"].to_numpy()
    hf = df["cl_hf"].to_numpy()
    hf_strict = np.where(converged, hf, np.nan)

    hf_test = np.asarray(hf[test_idx], dtype=float)
    x_test = x[test_idx]
    lf_test = lf[test_idx]
    print(f"Test set: {len(test_idx)} converged rows {test_idx} | pool: {len(pool_idx)} rows")

    curves: dict[str, dict[str, dict[int, dict[str, float]]]] = {
        "lenient": {},
        "strict": {},
    }
    for mode, oracle in (("lenient", hf), ("strict", hf_strict)):
        oracle_pool = oracle[pool_idx]
        x_pool = x[pool_idx]
        lf_pool = lf[pool_idx]

        curves[mode]["al"] = _run_scheme(
            "al", x_pool, lf_pool, oracle_pool, x_test, lf_test, hf_test,
            np.random.default_rng(42), 3407,
        )
        curves[mode]["maximin"] = _run_scheme(
            "maximin", x_pool, lf_pool, oracle_pool, x_test, lf_test, hf_test,
            np.random.default_rng(42), 3408,
        )

        per_seed = []
        for seed in RANDOM_SEEDS:
            rng_seed = np.random.default_rng(seed)
            per_seed.append(
                _run_scheme(
                    "random", x_pool, lf_pool, oracle_pool,
                    x_test, lf_test, hf_test, rng_seed, 3409,
                )
            )
        # average the random seeds at each budget
        agg: dict[int, dict[str, float]] = {}
        for budget in BUDGETS:
            rmses = np.array([c[budget]["rmse_cl"] for c in per_seed], dtype=float)
            obs = np.array([c[budget]["hf_observed"] for c in per_seed], dtype=float)
            agg[budget] = {
                "rmse_cl": float(np.nanmean(rmses)),
                "rmse_std": float(np.nanstd(rmses)),
                "hf_observed": float(np.mean(obs)),
            }
        curves[mode]["random"] = agg

    # ---- report ------------------------------------------------------------
    for mode in ("lenient", "strict"):
        print(f"\n--- Learning curves ({mode} oracle) ---")
        print(f"{'HF attempts':>12} | {'AL RMSE':>9} {'rand RMSE':>10} {'+-':>7} "
              f"{'space-fill':>11} | {'HF got AL/rand':>14}")
        for budget in BUDGETS:
            a = curves[mode]["al"][budget]["rmse_cl"]
            r = curves[mode]["random"][budget]["rmse_cl"]
            s = curves[mode]["random"][budget]["rmse_std"]
            f = curves[mode]["maximin"][budget]["rmse_cl"]
            ya = curves[mode]["al"][budget]["hf_observed"]
            yr = curves[mode]["random"][budget]["hf_observed"]
            print(f"{budget:>12} | {a:>9.4f} {r:>10.4f} {s:>7.4f} {f:>11.4f} | "
                  f"{ya:>6.1f} / {yr:.1f}")

    # ---- measured scorecard (not forced) ------------------------------------
    at = max(BUDGETS)
    al_final = curves["lenient"]["al"][at]["rmse_cl"]
    rand_final = curves["lenient"]["random"][at]["rmse_cl"]
    fill_final = curves["lenient"]["maximin"][at]["rmse_cl"]
    reduction_vs_random = (rand_final - al_final) / rand_final * 100.0
    reduction_vs_fill = (fill_final - al_final) / fill_final * 100.0
    strict_yield_al = curves["strict"]["al"][at]["hf_observed"]
    strict_yield_rand = curves["strict"]["random"][at]["hf_observed"]

    checks = [
        (
            f"AL beats random at N={at} by >5% RMSE",
            reduction_vs_random > 5.0,
            f"{reduction_vs_random:+.1f}%",
        ),
        (
            f"AL beats space-filling at N={at} by >5% RMSE",
            reduction_vs_fill > 5.0,
            f"{reduction_vs_fill:+.1f}%",
        ),
        (
            "Strict oracle: AL keeps more usable HF than random",
            strict_yield_al > strict_yield_rand,
            f"{strict_yield_al:.1f} vs {strict_yield_rand:.1f}",
        ),
    ]
    targets_met = all(ok for _, ok, _ in checks)

    print("\n--- V9 measured scorecard (not forced) ---")
    for label, ok, value in checks:
        print(f"{label} -> {'PASS' if ok else 'FAIL'} ({value})")
    if targets_met:
        print("\n>>> Case V9: uncertainty acquisition pays off on this dataset.\n")
    else:
        print(
            "\n>>> Case V9: targets NOT all met. That measured answer stands: active "
            "learning does not pay for itself on this dataset at these budgets.\n"
        )

    results = {
        "case": "V9 active learning vs random vs space-filling at equal HF budget",
        "dataset": "data/datasets/aero_surrogate_v1.parquet (28 real openfoam_wsl rows)",
        "hf_oracle": (
            "lenient: every pool row returns its measured HF value; strict: only "
            "converged rows return data, force-unstable queries consume budget"
        ),
        "test_rows": test_idx,
        "budgets": list(BUDGETS),
        "curves": curves,
        "scorecard": {
            "reduction_vs_random_pct": reduction_vs_random,
            "reduction_vs_maximin_pct": reduction_vs_fill,
            "strict_yield_al": strict_yield_al,
            "strict_yield_random": strict_yield_rand,
            "targets_met": bool(targets_met),
        },
    }
    CURVE_JSON.write_text(
        json.dumps(results, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Curves written to {CURVE_JSON}")

    return {
        "al_rmse_lenient": al_final,
        "random_rmse_lenient": rand_final,
        "maximin_rmse_lenient": fill_final,
        "reduction_vs_random_pct": reduction_vs_random,
        "reduction_vs_maximin_pct": reduction_vs_fill,
        "strict_yield_al": strict_yield_al,
        "strict_yield_random": strict_yield_rand,
        "targets_met": 1.0 if targets_met else 0.0,
        "hf_source": "cfd" if not provenance.get("all_synthetic") else "synthetic_model",
    }


if __name__ == "__main__":
    run_v9_validation()
