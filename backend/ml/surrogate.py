"""Aerodynamic surrogate models: Single-Fidelity GP and Kennedy-O'Hagan AR1 Multi-Fidelity.

Implements Gaussian Process regression with anisotropic Matern kernels and
autoregressive multi-fidelity residual modeling (Kennedy & O'Hagan 2000).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from backend.ml.dataset import FEATURE_COLUMNS, sample_vector_to_config
from backend.simulation.analytical import REF_AREA, RHO
from backend.simulation.panel_method import simulate_panel


def _build_gp_kernel(n_features: int) -> Any:
    """Build anisotropic Matern-5/2 kernel with constant scaling and calibrated white noise."""
    return ConstantKernel(
        constant_value=1.0, constant_value_bounds=(1e-2, 1e2)
    ) * Matern(
        length_scale=np.ones(n_features),
        length_scale_bounds=(0.1, 50.0),
        nu=2.5,
    ) + WhiteKernel(
        noise_level=0.005,
        noise_level_bounds=(0.002, 0.1),
    )


class SingleFidelityGP:
    """Single-fidelity Gaussian Process regressor for High-Fidelity CFD outputs."""

    def __init__(self, n_restarts_optimizer: int = 5, random_state: int = 42) -> None:
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.gp: GaussianProcessRegressor | None = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> SingleFidelityGP:
        """Fit single-fidelity GP on input features X and target values y."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        X_scaled = self.scaler_x.fit_transform(X_arr)
        y_scaled = self.scaler_y.fit_transform(y_arr).ravel()

        kernel = _build_gp_kernel(X_scaled.shape[1])
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            normalize_y=False,
        )
        self.gp.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected mean and standard deviation at query points X."""
        if not self.is_fitted or self.gp is None:
            raise RuntimeError("SingleFidelityGP model is not fitted yet.")

        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_x.transform(X_arr)

        if return_std:
            y_scaled_pred, y_scaled_std = self.gp.predict(X_scaled, return_std=True)
            # Add aleatoric noise variance from WhiteKernel if present
            noise_var = getattr(self.gp.kernel_, "k2", None)
            noise_level = getattr(noise_var, "noise_level", 0.0) if noise_var else 0.0
            total_scaled_std = np.sqrt(y_scaled_std**2 + noise_level)

            # Unscale mean
            y_pred = self.scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
            # Unscale standard deviation by target scale factor
            scale = float(self.scaler_y.scale_[0])
            y_std = total_scaled_std * scale
            return y_pred, y_std
        else:
            y_scaled_pred = self.gp.predict(X_scaled, return_std=False)
            y_pred = self.scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
            return y_pred, np.zeros_like(y_pred)


class AR1MultiFidelitySurrogate:
    """Kennedy & O'Hagan (2000) Autoregressive Multi-Fidelity Surrogate.

    Formulation:
        y_H(x) = rho * y_L(x) + c + delta(x)
        delta(x) ~ GP(0, k_delta(x, x'))
    """

    def __init__(self, n_restarts_optimizer: int = 5, random_state: int = 42) -> None:
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.rho: float = 1.0
        self.c: float = 0.0
        self.discrepancy_gp: SingleFidelityGP = SingleFidelityGP(
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        self.is_fitted: bool = False

    def fit(self, X_hf: np.ndarray, y_lf_at_hf: np.ndarray, y_hf: np.ndarray) -> AR1MultiFidelitySurrogate:
        """Fit AR1 surrogate using paired LF and HF evaluations at high-fidelity locations."""
        X_arr = np.asarray(X_hf, dtype=np.float64)
        y_lf_arr = np.asarray(y_lf_at_hf, dtype=np.float64).ravel()
        y_hf_arr = np.asarray(y_hf, dtype=np.float64).ravel()

        # 1. Estimate scaling parameter rho and intercept c via linear fit
        if len(y_lf_arr) > 1 and float(np.var(y_lf_arr)) > 1e-8:
            poly = np.polyfit(y_lf_arr, y_hf_arr, deg=1)
            self.rho = float(poly[0])
            self.c = float(poly[1])
        else:
            self.rho = 1.0
            self.c = 0.0

        # 2. Compute discrepancy delta = y_H - (rho * y_L + c)
        delta = y_hf_arr - (self.rho * y_lf_arr + self.c)

        # 3. Fit Gaussian Process on discrepancy
        self.discrepancy_gp.fit(X_arr, delta)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        y_lf: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict multi-fidelity mean and epistemic uncertainty."""
        if not self.is_fitted:
            raise RuntimeError("AR1MultiFidelitySurrogate model is not fitted yet.")

        X_arr = np.asarray(X, dtype=np.float64)
        y_lf_arr = np.asarray(y_lf, dtype=np.float64).ravel()

        # Predict discrepancy delta mean and std
        delta_mean, delta_std = self.discrepancy_gp.predict(X_arr, return_std=True)

        y_hf_pred = self.rho * y_lf_arr + self.c + delta_mean
        if return_std:
            return y_hf_pred, delta_std
        return y_hf_pred, np.zeros_like(y_hf_pred)


@dataclass
class AeroPrediction:
    cl_mean: float
    cl_std: float
    cd_mean: float
    cd_std: float
    downforce_N: float
    drag_N: float
    efficiency: float


class AeroSurrogatePipeline:
    """Unified multi-output surrogate pipeline predicting Cl, Cd, and aero forces."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.cl_surrogate = AR1MultiFidelitySurrogate(random_state=random_state)
        self.cd_surrogate = AR1MultiFidelitySurrogate(random_state=random_state + 1)
        self.cl_single_gp = SingleFidelityGP(random_state=random_state)
        self.cd_single_gp = SingleFidelityGP(random_state=random_state + 1)
        self.is_fitted: bool = False

    def fit_from_dataframe(self, df_train: Any) -> AeroSurrogatePipeline:
        """Train both Single-Fidelity and Multi-Fidelity models from a dataset DataFrame."""
        X_train = df_train[FEATURE_COLUMNS].to_numpy()
        cl_lf = df_train["cl_lf"].to_numpy()
        cd_lf = df_train["cd_lf"].to_numpy()
        cl_hf = df_train["cl_hf"].to_numpy()
        cd_hf = df_train["cd_hf"].to_numpy()

        # Fit Multi-Fidelity AR1
        self.cl_surrogate.fit(X_train, cl_lf, cl_hf)
        self.cd_surrogate.fit(X_train, cd_lf, cd_hf)

        # Fit Single-Fidelity GP
        self.cl_single_gp.fit(X_train, cl_hf)
        self.cd_single_gp.fit(X_train, cd_hf)

        self.is_fitted = True
        return self

    def predict_point(self, sample_values: dict[str, float]) -> AeroPrediction:
        """Predict aerodynamic coefficients and forces from parameter dictionary."""
        if not self.is_fitted:
            raise RuntimeError("AeroSurrogatePipeline is not fitted yet.")

        # 1. Run quick Low-Fidelity physics (VLM)
        config = sample_vector_to_config(sample_values)
        vlm_res = simulate_panel(config)
        cl_lf = vlm_res.forces.cl_total
        cd_lf = vlm_res.forces.cd_total

        x_vec = np.array([[sample_values[col] for col in FEATURE_COLUMNS]])

        cl_pred, cl_std = self.cl_surrogate.predict(x_vec, np.array([cl_lf]))
        cd_pred, cd_std = self.cd_surrogate.predict(x_vec, np.array([cd_lf]))

        cl_val = float(cl_pred[0])
        cd_val = float(cd_pred[0])
        speed_ms = sample_values["speed_kmh"] / 3.6
        q = 0.5 * RHO * speed_ms**2

        df_N = cl_val * q * REF_AREA
        drag_N = cd_val * q * REF_AREA
        eff = df_N / drag_N if drag_N > 1.0 else 1.0

        return AeroPrediction(
            cl_mean=round(cl_val, 4),
            cl_std=round(float(cl_std[0]), 4),
            cd_mean=round(cd_val, 4),
            cd_std=round(float(cd_std[0]), 4),
            downforce_N=round(df_N, 1),
            drag_N=round(drag_N, 1),
            efficiency=round(eff, 2),
        )

    def save(self, filepath: Path) -> Path:
        """Serialize trained pipeline to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> AeroSurrogatePipeline:
        """Deserialize trained pipeline from disk."""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return cast("AeroSurrogatePipeline", model)
