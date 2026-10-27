"""Richardson Extrapolation and Grid Convergence Index (GCI) calculations.

Follows the ASME standard procedure:
Celik, I. B., et al. (2008). "Procedure for estimation and reporting of
uncertainty due to discretization in CFD applications".
ASME Journal of Fluids Engineering, 130(7), 078001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GCIReport:
    """Complete ASME GCI verification metrics across 3 grid levels."""

    phi_fine: float
    phi_medium: float
    phi_coarse: float
    h_fine: float
    h_medium: float
    h_coarse: float
    r21: float
    r32: float
    apparent_order_p: float
    phi_extrapolated: float
    relative_error_fine_pct: float
    gci_fine_pct: float
    gci_medium_pct: float
    asymptotic_range_ratio: float
    is_in_asymptotic_range: bool


def compute_gci(
    phi_fine: float,
    phi_medium: float,
    phi_coarse: float,
    h_fine: float,
    h_medium: float,
    h_coarse: float,
    safety_factor: float = 1.25,
) -> GCIReport:
    """Calculate apparent order of convergence and GCI for three systematically refined grids.

    Index 1 = Fine, Index 2 = Medium, Index 3 = Coarse (h1 < h2 < h3).
    """
    r21 = h_medium / h_fine
    r32 = h_coarse / h_medium

    eps32 = phi_coarse - phi_medium
    eps21 = phi_medium - phi_fine

    # Check for oscillatory or uniform convergence
    ratio = eps32 / eps21 if abs(eps21) > 1e-12 else 1.0
    s = 1.0 if ratio > 0 else -1.0

    # Iterative solve for apparent order p
    # For constant refinement ratio r21 == r32 = r: p = ln|eps32 / eps21| / ln(r)
    p = abs(math.log(abs(ratio))) / math.log(r21) if ratio != 0.0 else 2.0

    # Refine p using fixed-point iteration if r21 != r32
    if abs(r21 - r32) > 0.01:
        for _ in range(20):
            q = math.log((r21**p - s) / (r32**p - s))
            p_new = (1.0 / math.log(r21)) * (abs(math.log(abs(ratio))) + q)
            if abs(p_new - p) < 1e-4:
                p = p_new
                break
            p = 0.5 * (p + p_new)

    # Bound order to realistic physical numerical range [0.5, 4.0]
    p = max(0.5, min(4.0, p))

    # Extrapolated value
    denom21 = (r21**p) - 1.0
    denom32 = (r32**p) - 1.0
    if abs(denom21) < 1e-6:
        denom21 = 1e-6
    if abs(denom32) < 1e-6:
        denom32 = 1e-6

    phi_ext = ((r21**p) * phi_fine - phi_medium) / denom21

    # Relative errors
    ea21 = abs((phi_fine - phi_medium) / phi_fine) if abs(phi_fine) > 1e-12 else 0.0
    ea32 = abs((phi_medium - phi_coarse) / phi_medium) if abs(phi_medium) > 1e-12 else 0.0

    # GCI values in percentage (ASME Eq. 5)
    gci_fine = (safety_factor * ea21 / denom21) * 100.0
    gci_medium = (safety_factor * ea32 / denom32) * 100.0

    # Asymptotic range check (ASME Eq. 6): ratio should be close to 1.0
    denom_asymp = (r21**p) * gci_fine
    asymp_ratio = gci_medium / denom_asymp if abs(denom_asymp) > 1e-12 else 1.0
    is_asymp = 0.85 <= asymp_ratio <= 1.15

    return GCIReport(
        phi_fine=phi_fine,
        phi_medium=phi_medium,
        phi_coarse=phi_coarse,
        h_fine=h_fine,
        h_medium=h_medium,
        h_coarse=h_coarse,
        r21=r21,
        r32=r32,
        apparent_order_p=p,
        phi_extrapolated=phi_ext,
        relative_error_fine_pct=ea21 * 100.0,
        gci_fine_pct=gci_fine,
        gci_medium_pct=gci_medium,
        asymptotic_range_ratio=asymp_ratio,
        is_in_asymptotic_range=is_asymp,
    )
