"""
ODE integration and curve fitting for coast-down segments.

Model:  m·dv/dt = -α·v² - β - P_mgu/v

  α      : aerodynamic coefficient  (N·s²/m²)  =  ½ρ·(CdA + Crr·ClA)
  β      : rolling resistance force (N)         =  Crr·m·g
  P_mgu  : MGU-K harvest power      (W)         absorbed as constant-power retarding term

Adding the P_mgu/v term separates genuine rolling resistance (β) from the
speed-dependent energy-recovery braking that contaminates β in a two-parameter fit.
The ODE has no closed-form solution so we integrate numerically.

Note on engine braking: at throttle=0, the drivetrain applies additional retarding
torque (compression, MGU-H harvest). This force is collinear with α in coast-down
data — both α·v² and engine braking terms are speed-dependent — making them
non-separable without external torque measurements. Engine braking is therefore
absorbed into α, inflating the composite 2α/ρ by ~66% relative to expected CdA.
The Durbin-Watson statistic of ~0.66 in the residuals reflects this model
mis-specification and should be treated as a known systematic limitation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.integrate import odeint
from scipy.optimize import curve_fit, least_squares


@dataclass
class FitResult:
    alpha: float        # N·s²/m²
    beta: float         # N
    alpha_std: float
    beta_std: float
    P_mgu: float        # W
    P_mgu_std: float
    v0: float           # m/s
    m: float            # kg
    drs_open: bool
    lap_number: int
    r2: float


def v_model(
    t: np.ndarray,
    alpha: float,
    beta: float,
    P_mgu: float,
    v0: float,
    m: float,
) -> np.ndarray:
    """Numerically integrate m·dv/dt = -α·v² - β - P_mgu/v."""
    def rhs(v, _t):
        v_safe = max(float(v[0]), 1.0)
        return [-(alpha * v_safe**2 + beta + P_mgu / v_safe) / m]

    result = odeint(rhs, v0, t, rtol=1e-4, atol=1e-6, full_output=False)
    return result[:, 0]


def fit_segment(
    seg: pd.DataFrame,
    m: float,
    rho: float,
    drs_open: bool,
    lap_number: int,
    beta_fixed: float | None = None,
    min_speed_drop: float = 25.0,
) -> FitResult | None:
    """
    Fit (alpha, P_mgu) — or (alpha, beta, P_mgu) if beta_fixed is None — to a
    single coast-down segment.

    beta_fixed: if provided, β is held at this value (N) and only α and P_mgu
                are fitted. Recommended: use the median β from a prior free fit
                (e.g. 120 N) to break the α/β/P_mgu degeneracy.

    min_speed_drop: minimum speed decrease over the segment in m/s (default 25.0).
                    Lower values allow shorter segments on tight circuits (e.g. 10.0)
                    at the cost of reduced α/P_mgu separability.

    Returns None if the fit fails or the result is physically unreasonable.
    """
    t = seg["t"].values - seg["t"].values[0]
    v = seg["Speed"].values / 3.6
    v0 = float(v[0])
    v_drop = v0 - float(v[-1])

    # Require high entry speed, enough data points, and a minimum speed drop.
    # The speed-drop filter ensures α·v² and P_mgu/v have genuinely different
    # shapes across the segment — the key condition for reliable separation.
    if v0 < 50.0 or len(t) < 10 or v_drop < min_speed_drop:
        return None

    if beta_fixed is not None:
        beta = float(beta_fixed)

        def model(t_, alpha, P_mgu):
            return v_model(t_, alpha, beta, P_mgu, v0, m)

        try:
            popt, pcov = curve_fit(
                model, t, v,
                p0=[0.5, 60_000.0],
                bounds=([0.10, 0.0], [2.50, 120_000.0]),
                maxfev=10_000,
            )
        except (RuntimeError, ValueError):
            return None

        alpha, P_mgu = popt
        diag = np.diag(pcov)
        if np.any(~np.isfinite(diag)) or np.any(diag < 0):
            return None
        alpha_std, P_mgu_std = np.sqrt(diag)
        beta_std = 0.0

        if alpha_std / alpha > 1.0:
            return None
        if P_mgu > 1_000 and P_mgu_std / P_mgu > 1.0:
            return None

        v_pred = model(t, alpha, P_mgu)

    else:
        def model3(t_, alpha, beta, P_mgu):
            return v_model(t_, alpha, beta, P_mgu, v0, m)

        try:
            popt, pcov = curve_fit(
                model3, t, v,
                p0=[0.5, 150.0, 60_000.0],
                bounds=(
                    [0.10,  80.0,       0.0],
                    [2.50, 220.0, 120_000.0],
                ),
                maxfev=10_000,
            )
        except (RuntimeError, ValueError):
            return None

        alpha, beta, P_mgu = popt
        diag = np.diag(pcov)
        if np.any(~np.isfinite(diag)) or np.any(diag < 0):
            return None
        alpha_std, beta_std, P_mgu_std = np.sqrt(diag)

        if alpha_std / alpha > 1.0 or beta_std / beta > 1.0:
            return None
        if P_mgu > 1_000 and P_mgu_std / P_mgu > 1.0:
            return None

        v_pred = model3(t, alpha, beta, P_mgu)

    if not (0.2 < alpha < 1.8):
        return None

    ss_res = np.sum((v - v_pred) ** 2)
    ss_tot = np.sum((v - v.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return FitResult(
        alpha=alpha,
        beta=beta,
        alpha_std=alpha_std,
        beta_std=beta_std,
        P_mgu=P_mgu,
        P_mgu_std=P_mgu_std,
        v0=v0,
        m=m,
        drs_open=drs_open,
        lap_number=lap_number,
        r2=r2,
    )


def fit_segments_pooled(
    segments: list[pd.DataFrame],
    m_list: list[float],
    drs_states: list[bool],
    lap_numbers: list[int],
    seed_results: list[FitResult] | None = None,
    beta_fixed: float | None = None,
) -> list[FitResult] | None:
    """
    Pooled fitting: α and β are shared across all segments; P_mgu varies per segment.

    Parameters reduce from 3×N (per-segment) to 2 + N (pooled), giving much
    tighter constraints on α and β when N ≥ 3.

    seed_results: per-segment FitResult list used to warm-start the optimiser.
                  Falls back to fixed defaults if None.

    Returns a list of FitResult (one per segment) with the shared α, β,
    or None if the fit fails.
    """
    n = len(segments)
    if n < 2:
        return None

    t_list, v_list, v0_list = [], [], []
    for seg in segments:
        t = seg["t"].values - seg["t"].values[0]
        v = seg["Speed"].values / 3.6
        t_list.append(t)
        v_list.append(v)
        v0_list.append(float(v[0]))

    _beta = float(beta_fixed) if beta_fixed is not None else None

    def _residuals(params: np.ndarray) -> np.ndarray:
        if _beta is not None:
            alpha = params[0]
            P_mgus = params[1:]
            beta = _beta
        else:
            alpha, beta = params[0], params[1]
            P_mgus = params[2:]
        parts = []
        for i in range(n):
            v_pred = v_model(t_list[i], alpha, beta, P_mgus[i], v0_list[i], m_list[i])
            parts.append(v_list[i] - v_pred)
        return np.concatenate(parts)

    # Warm-start from per-segment medians / individual values
    if seed_results and len(seed_results) == n:
        a0 = float(np.median([r.alpha for r in seed_results]))
        b0 = float(np.median([r.beta  for r in seed_results]))
        p0_list = [r.P_mgu for r in seed_results]
    else:
        a0, b0 = 0.5, 150.0
        p0_list = [60_000.0] * n

    if _beta is not None:
        x0 = np.array([a0] + p0_list)
        lower = np.array([0.10] + [0.0]       * n)
        upper = np.array([2.50] + [120_000.0] * n)
    else:
        x0 = np.array([a0, b0] + p0_list)
        lower = np.array([0.10,  80.0] + [0.0]       * n)
        upper = np.array([2.50, 220.0] + [120_000.0] * n)

    try:
        result = least_squares(
            _residuals, x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=100_000,
        )
    except Exception:
        return None

    if not result.success and result.cost > 10.0:
        return None

    if _beta is not None:
        alpha = float(result.x[0])
        beta  = _beta
        P_mgus = result.x[1:]
    else:
        alpha, beta = float(result.x[0]), float(result.x[1])
        P_mgus = result.x[2:]

    # Approximate parameter uncertainties from the Jacobian
    J = result.jac
    n_pts = sum(len(v) for v in v_list)
    dof = max(1, n_pts - len(x0))
    try:
        cov = np.linalg.inv(J.T @ J) * (result.cost / dof)
        perr = np.sqrt(np.abs(np.diag(cov)))
        if _beta is not None:
            alpha_std = float(perr[0])
            beta_std  = 0.0
            P_mgu_stds = perr[1:].tolist()
        else:
            alpha_std, beta_std = float(perr[0]), float(perr[1])
            P_mgu_stds = perr[2:].tolist()
    except np.linalg.LinAlgError:
        alpha_std = beta_std = float("nan")
        P_mgu_stds = [float("nan")] * n

    fit_results = []
    for i in range(n):
        v_pred = v_model(t_list[i], alpha, beta, P_mgus[i], v0_list[i], m_list[i])
        ss_res = np.sum((v_list[i] - v_pred) ** 2)
        ss_tot = np.sum((v_list[i] - v_list[i].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        fit_results.append(FitResult(
            alpha=alpha,
            beta=beta,
            alpha_std=alpha_std,
            beta_std=beta_std,
            P_mgu=float(P_mgus[i]),
            P_mgu_std=float(P_mgu_stds[i]),
            v0=v0_list[i],
            m=m_list[i],
            drs_open=drs_states[i],
            lap_number=lap_numbers[i],
            r2=r2,
        ))

    return fit_results


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
