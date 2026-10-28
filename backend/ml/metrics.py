"""Evaluation and uncertainty calibration metrics for aerodynamic surrogates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    rmse: float
    mae: float
    mape: float
    coverage_95: float
    mean_ci_width_95: float

    def to_dict(self) -> dict[str, float]:
        return {
            "r2": round(self.r2, 4),
            "rmse": round(self.rmse, 5),
            "mae": round(self.mae, 5),
            "mape_pct": round(self.mape, 2),
            "coverage_95_pct": round(self.coverage_95 * 100.0, 2),
            "mean_ci_width_95": round(self.mean_ci_width_95, 4),
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None = None,
) -> RegressionMetrics:
    """Compute standard regression accuracy and probabilistic calibration metrics."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Residuals
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    # MAPE: avoid div by zero
    non_zero = np.abs(y_true) > 1e-6
    if np.any(non_zero):
        mape = float(np.mean(np.abs(residuals[non_zero] / y_true[non_zero])) * 100.0)
    else:
        mape = 0.0

    # 95% Confidence Interval Coverage (nominal coverage for Gaussian is z=1.96)
    if y_std is not None:
        y_std = np.asarray(y_std).ravel()
        ci_lower = y_pred - 1.96 * y_std
        ci_upper = y_pred + 1.96 * y_std
        covered = (y_true >= ci_lower) & (y_true <= ci_upper)
        coverage_95 = float(np.mean(covered))
        mean_width = float(np.mean(ci_upper - ci_lower))
    else:
        coverage_95 = 0.0
        mean_width = 0.0

    return RegressionMetrics(
        r2=float(r2),
        rmse=rmse,
        mae=mae,
        mape=mape,
        coverage_95=coverage_95,
        mean_ci_width_95=mean_width,
    )


def compute_reliability_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    quantiles: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Compute empirical vs nominal coverage across multiple confidence intervals.

    Used to assess over-confidence or under-confidence of the Gaussian Process.
    """
    if quantiles is None:
        # Nominal percentiles: 50%, 68%, 80%, 90%, 95%, 99%
        quantiles = np.array([0.50, 0.68, 0.80, 0.90, 0.95, 0.99])

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_std = np.maximum(np.asarray(y_std).ravel(), 1e-6)

    # Standard normal z-scores for two-sided central intervals
    from scipy.stats import norm

    empirical_coverages: list[float] = []

    for q in quantiles:
        z = norm.ppf(0.5 + q / 2.0)
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        emp_cov = float(np.mean((y_true >= lower) & (y_true <= upper)))
        empirical_coverages.append(emp_cov)

    return {
        "nominal": quantiles.tolist(),
        "empirical": empirical_coverages,
    }
