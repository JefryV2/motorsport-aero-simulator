# ADR 0003: Machine Learning Surrogate Architecture and Dependencies

- **Status:** Accepted
- **Date:** 2026-10-27
- **Author:** Computational Aerodynamics & ML Agent
- **Deciders:** Engineering Lead / User

---

## 1. Context and Problem Statement

Phase 3 of the project requires:
1. Generating a structured dataset over the bounded aerodynamic design space using Latin Hypercube Sampling (LHS) or Sobol sequences.
2. Saving evaluations with full provenance (geometry parameters, conditions, solver version, convergence flag, high-fidelity CFD result, low-fidelity VLM prediction) into an efficient columnar storage format (Parquet).
3. Training and evaluating:
   - Single-fidelity Gaussian Process (GP) on high-fidelity (HF) data.
   - Kennedy & O'Hagan (2000) AR1 autoregressive multi-fidelity model:
     $$y_H(\mathbf{x}) = \rho \cdot y_L(\mathbf{x}) + \delta(\mathbf{x})$$
     where $\rho$ is the scaling factor and $\delta(\mathbf{x}) \sim \mathcal{GP}(0, k(\mathbf{x}, \mathbf{x}'))$ models the residual discrepancy.
4. Measuring cross-validation $R^2$, RMSE, MAPE, and evaluating uncertainty calibration (coverage and reliability curves).

---

## 2. Decision

We introduce standard, industry-standard scientific Python dependencies:
- `scikit-learn`: Gaussian Process regression with Matern/RBF anisotropic kernels (`GaussianProcessRegressor`, `Matern`, `WhiteKernel`), cross-validation splits, and metrics.
- `pandas` and `pyarrow`: Columnar Parquet dataset reading, writing, and partitioning.

### Architecture

1. **Dataset Generation (`backend/ml/dataset.py`)**:
   - `LatinHypercube` sampling from `scipy.stats.qmc` across the 5 design parameters:
     - Front wing main angle $\alpha_{main} \in [0.0^\circ, 8.0^\circ]$
     - Front wing flap angle $\alpha_{flap} \in [4.0^\circ, 20.0^\circ]$
     - Ride height $h_{front} \in [15.0\,\text{mm}, 50.0\,\text{mm}]$
     - Rear wing angle $\alpha_{rear} \in [5.0^\circ, 25.0^\circ]$
     - Freestream speed $V \in [100.0\,\text{km/h}, 300.0\,\text{km/h}]$
   - Evaluates low-fidelity (VLM, $y_L$) and high-fidelity (CFD, $y_H$) at each sample point.
   - Writes immutable, reproducible train/validation/test splits to `data/datasets/aero_surrogate_v1.parquet`.

2. **Surrogate Models (`backend/ml/surrogate.py`)**:
   - `SingleFidelityGP`: Standard GP regressor with anisotropic $\mathrm{Matern}(\nu = 5/2)$ kernel and hyperparameter optimization.
   - `AR1MultiFidelitySurrogate`: Autoregressive Kennedy-O'Hagan multi-fidelity formulation combining VLM physics baseline with Gaussian process residual discrepancy learning.
   - Predictions return both expected mean $\hat{y}$ and predictive standard deviation $\sigma$.

3. **Validation & Calibration (V7 & V8)**:
   - Case V7: Cross-validation $R^2 > 0.90$ and $\mathrm{MAPE} < 5.0\%$.
   - Case V8: 95% confidence interval coverage between 90% and 98%.

---

## 3. Consequences

- **Positive**:
  - Leverages scikit-learn's robust L-BFGS-B hyperparameter optimizer and kernel library.
  - Columnar Parquet storage ensures high performance, schema enforcement, and versioning.
  - Provides rigorous predictive uncertainties required for Phase 4 active learning acquisition functions.
- **Negative**:
  - Increases virtual environment disk footprint by $\approx 60\,\text{MB}$.
