"""
ODE integration and curve fitting for coast-down segments.

Model:  m·dv/dt = -α·v² - β
Analytical solution:
    v(t) = sqrt(β/α) · tan(φ₀ - (k/m)·t)
    where  k = sqrt(α·β),  φ₀ = arctan(v₀·sqrt(α/β))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    alpha: float        # N·s²/m²
    beta: float         # N
    alpha_std: float
    beta_std: float
    v0: float           # m/s
    m: float            # kg
    drs_open: bool
    lap_number: int
    r2: float


def v_model(t: np.ndarray, alpha: float, beta: float, v0: float, m: float) -> np.ndarray:
    """Analytical solution to m·dv/dt = -alpha·v² - beta."""
    k = np.sqrt(alpha * beta)
    phi0 = np.arctan(v0 * np.sqrt(alpha / beta))
    arg = phi0 - (k / m) * t
    # Guard: tan blows up approaching ±π/2
    arg = np.clip(arg, -1.5, 1.5)
    return np.sqrt(beta / alpha) * np.tan(arg)


def fit_segment(
    seg: pd.DataFrame,
    m: float,
    rho: float,
    drs_open: bool,
    lap_number: int,
) -> FitResult | None:
    """
    Fit alpha and beta to a single coast-down segment.
    Returns None if the fit fails or the result is physically unreasonable.
    """
    t = seg["t"].values - seg["t"].values[0]
    v = seg["Speed"].values / 3.6      # km/h → m/s
    v0 = float(v[0])

    # Need high entry speed so the α·v² term dominates and can be separated from β.
    # Below ~33 m/s (120 km/h) drag is small relative to rolling resistance and
    # the two parameters trade off freely, producing unphysical results.
    if v0 < 33.0 or len(t) < 5:
        return None

    def model(t_, alpha, beta):
        return v_model(t_, alpha, beta, v0, m)

    try:
        popt, pcov = curve_fit(
            model, t, v,
            p0=[0.5, 160.0],
            bounds=([1e-4, 30.0], [2.5, 1500.0]),
            maxfev=5000,
        )
    except (RuntimeError, ValueError):
        return None

    alpha, beta = popt
    perr = np.sqrt(np.diag(pcov))
    alpha_std, beta_std = perr

    # Reject if uncertainty dominates
    if alpha_std / alpha > 1.0 or beta_std / beta > 1.0:
        return None

    # Physical plausibility gates: Monza low-downforce config
    # α = ½ρ·(CdA + Crr·ClA): expected 0.35–1.2 N·s²/m²
    # β = Crr·m·g: Crr 0.010–0.025, m 850–970 kg → β 83–238 N
    if not (0.2 < alpha < 1.8) or not (50 < beta < 500):
        return None

    v_pred = model(t, alpha, beta)
    ss_res = np.sum((v - v_pred) ** 2)
    ss_tot = np.sum((v - v.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return FitResult(
        alpha=alpha,
        beta=beta,
        alpha_std=alpha_std,
        beta_std=beta_std,
        v0=v0,
        m=m,
        drs_open=drs_open,
        lap_number=lap_number,
        r2=r2,
    )


def fit_all_segments(
    segments: list[pd.DataFrame],
    m: float,
    rho: float,
    drs_states: list[bool],
    lap_number: int,
    min_r2: float = 0.90,
) -> list[FitResult]:
    results = []
    for seg, drs in zip(segments, drs_states):
        result = fit_segment(seg, m, rho, drs, lap_number)
        if result is not None and result.r2 >= min_r2:
            results.append(result)
    return results
