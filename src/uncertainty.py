"""
Monte Carlo uncertainty propagation for aero parameter estimates.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from src.ode_fit import FitResult

G = 9.81


@dataclass
class MCResult:
    CdA_mean: float
    CdA_std: float
    CdA_p5: float
    CdA_p95: float
    Crr_mean: float
    Crr_std: float


def propagate_uncertainty(
    fit_results: list[FitResult],
    rho: float,
    ClA_mean: float,
    ClA_std: float,
    n_samples: int = 10_000,
    seed: int = 42,
) -> MCResult:
    """
    Draw from the distribution of each fit result's alpha/beta (treating them as
    independent Gaussians) and ClA, then compute the resulting CdA distribution.
    Aggregates across all fit results by pooling.
    Returns an MCResult of NaNs if fit_results is empty.
    """
    if not fit_results:
        nan = float('nan')
        return MCResult(
            CdA_mean=nan, CdA_std=nan, CdA_p5=nan, CdA_p95=nan,
            Crr_mean=nan, Crr_std=nan,
        )

    rng = np.random.default_rng(seed)
    CdA_pool: list[np.ndarray] = []
    Crr_pool: list[np.ndarray] = []

    for fit in fit_results:
        alpha_s = rng.normal(fit.alpha, fit.alpha_std, n_samples).clip(min=1e-6)
        beta_s = rng.normal(fit.beta, fit.beta_std, n_samples).clip(min=1.0)
        ClA_s = rng.normal(ClA_mean, ClA_std, n_samples).clip(min=0.1)

        Crr_s = beta_s / (fit.m * G)
        composite_s = 2.0 * alpha_s / rho
        CdA_s = composite_s - Crr_s * ClA_s

        CdA_pool.append(CdA_s)
        Crr_pool.append(Crr_s)

    CdA_all = np.concatenate(CdA_pool)
    Crr_all = np.concatenate(Crr_pool)

    return MCResult(
        CdA_mean=float(np.mean(CdA_all)),
        CdA_std=float(np.std(CdA_all)),
        CdA_p5=float(np.percentile(CdA_all, 5)),
        CdA_p95=float(np.percentile(CdA_all, 95)),
        Crr_mean=float(np.mean(Crr_all)),
        Crr_std=float(np.std(Crr_all)),
    )


def drs_delta_uncertainty(
    mc_closed: MCResult,
    mc_open: MCResult,
    n_samples: int = 10_000,
    seed: int = 43,
) -> tuple[float, float]:
    """
    Propagate uncertainty in the DRS drag delta.
    Returns (mean_delta, std_delta).
    """
    rng = np.random.default_rng(seed)
    closed_s = rng.normal(mc_closed.CdA_mean, mc_closed.CdA_std, n_samples)
    open_s = rng.normal(mc_open.CdA_mean, mc_open.CdA_std, n_samples)
    delta = open_s - closed_s
    return float(np.mean(delta)), float(np.std(delta))
