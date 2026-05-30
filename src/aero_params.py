"""
Derive aerodynamic parameters (CdA, ClA, Crr) from fitted ODE coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.ode_fit import FitResult
from src.segments import extract_corner_samples

G = 9.81


@dataclass
class AeroParams:
    CdA: float
    CdA_std: float
    ClA: float
    ClA_std: float
    Crr: float
    Crr_std: float
    CdA_drs_open: float
    CdA_drs_open_std: float
    delta_CdA: float
    delta_CdA_std: float


def air_density(temp_C: float, pressure_hPa: float) -> float:
    R = 287.05
    return (pressure_hPa * 100.0) / (R * (temp_C + 273.15))


def car_mass(
    lap_number: int,
    M_car: float = 798.0,
    M_driver: float = 80.0,
    M_fuel_0: float = 95.0,
    fuel_burn: float = 1.8,
) -> float:
    fuel = max(0.0, M_fuel_0 - fuel_burn * lap_number)
    return M_car + M_driver + fuel


def extract_crr_and_composite(
    fit: FitResult, rho: float
) -> tuple[float, float]:
    """Return (Crr, CdA + Crr*ClA) for a single fit result."""
    Crr = fit.beta / (fit.m * G)
    composite = 2.0 * fit.alpha / rho
    return Crr, composite


# Known fast-corner geometry for Monza (circuit-distance in metres, radius in metres).
# Only corners where F1 cars operate at/near the tyre friction limit are included.
MONZA_CORNERS = [
    {"name": "Curva Grande",  "dist_lo": 1200, "dist_hi": 1600, "radius": 300},
    {"name": "Lesmo 1",       "dist_lo": 1750, "dist_hi": 1950, "radius": 140},
    {"name": "Lesmo 2",       "dist_lo": 2050, "dist_hi": 2250, "radius": 100},
    {"name": "Parabolica",    "dist_lo": 4900, "dist_hi": 5250, "radius":  80},
]


def estimate_ClA_from_corners(
    lap_telemetry: pd.DataFrame,
    m: float,
    rho: float,
    corners: list[dict] | None = None,
    min_speed_kmh: float = 150.0,
    min_throttle: float = 50.0,
) -> tuple[list[float], list[float]]:
    """
    Estimate ClA using known corner radii rather than GPS-derived lateral g.

    For each corner passage, centripetal acceleration = v²/R (no GPS needed).
    Returns (v2_list, centripetal_a_list) suitable for linear regression in the notebook.

    Tyre friction model:   v²/R  =  μ·g  +  (μ·ρ·ClA)/(2m) · v²
    Fitting v²/R vs v²:
        intercept → μ = b / g
        slope     → ClA = a · 2m / (μ·ρ)
    """
    if corners is None:
        corners = MONZA_CORNERS

    tel = lap_telemetry.copy().reset_index(drop=True)
    if "Distance" not in tel.columns:
        return [], []

    tel["v_ms"] = tel["Speed"] / 3.6
    v2_out, ca_out = [], []

    for corner in corners:
        mask = (
            (tel["Distance"] >= corner["dist_lo"])
            & (tel["Distance"] <= corner["dist_hi"])
            & (tel["Speed"] >= min_speed_kmh)
            & (tel["Throttle"] >= min_throttle)
        )
        seg = tel[mask]
        if seg.empty:
            continue

        v2 = seg["v_ms"].values ** 2
        centripetal_a = v2 / corner["radius"]   # m/s²

        v2_out.extend(v2.tolist())
        ca_out.extend(centripetal_a.tolist())

    return v2_out, ca_out


def fit_ClA_from_corners(
    v2_all: list[float],
    ca_all: list[float],
    m: float,
    rho: float,
) -> tuple[float, float, float, float]:
    """
    Linear regression of centripetal_a vs v² across all corner samples:
        ca = a·v² + b   →   a = μ·ρ·ClA/(2m),  b = μ·g

    Returns (ClA, ClA_std, mu, mu_std).
    """
    v2 = np.array(v2_all)
    ca = np.array(ca_all)

    if len(v2) < 4:
        return np.nan, np.nan, np.nan, np.nan

    # OLS with intercept
    X = np.column_stack([v2, np.ones_like(v2)])
    coeffs, residuals, _, _ = np.linalg.lstsq(X, ca, rcond=None)
    a, b = coeffs

    # Uncertainty from residual variance
    if len(v2) > 2 and a > 0 and b > 0:
        ca_pred = a * v2 + b
        sigma2 = np.sum((ca - ca_pred) ** 2) / (len(v2) - 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        a_std, b_std = se
    else:
        a_std, b_std = np.nan, np.nan

    mu     = b / G
    mu_std = b_std / G if np.isfinite(b_std) else np.nan

    if a <= 0 or mu <= 0:
        return np.nan, np.nan, np.nan, np.nan

    ClA     = a * 2.0 * m / (mu * rho)
    ClA_std = ClA * np.sqrt((a_std / a) ** 2 + (mu_std / mu) ** 2) if np.isfinite(a_std) else np.nan

    return float(ClA), float(ClA_std), float(mu), float(mu_std)


def estimate_ClA(
    lap_telemetry: pd.DataFrame,
    m: float,
    rho: float,
    mu: float = 1.8,
    min_speed_kmh: float = 144.0,
    min_lat_g: float = 3.5,
    min_throttle: float = 80.0,
) -> tuple[float, float]:
    """
    Legacy GPS-based ClA estimator (kept for API compatibility).
    Estimate ClA from high-speed corner samples using tyre friction model.
    Returns (median_ClA, std_ClA).
    """
    samples = extract_corner_samples(
        lap_telemetry,
        min_speed_kmh=min_speed_kmh,
        min_lat_g=min_lat_g,
        min_throttle=min_throttle,
    )
    if samples.empty:
        return np.nan, np.nan

    v = samples["v_ms"].values
    lat_a = samples["lat_g"].values * G

    with np.errstate(divide="ignore", invalid="ignore"):
        ClA_vals = (2.0 * m / rho) * (lat_a / mu - G) / v**2

    ClA_vals = ClA_vals[np.isfinite(ClA_vals) & (ClA_vals > 0)]
    if len(ClA_vals) == 0:
        return np.nan, np.nan

    return float(np.median(ClA_vals)), float(np.std(ClA_vals))


def compute_CdA(composite: float, Crr: float, ClA: float) -> float:
    return composite - Crr * ClA


def aggregate_results(
    fit_results: list[FitResult],
    rho: float,
    ClA: float,
    ClA_std: float,
) -> AeroParams:
    """
    Aggregate fit results into final aero parameters, separated by DRS state.
    """
    closed = [r for r in fit_results if not r.drs_open]
    opened = [r for r in fit_results if r.drs_open]

    def _stats(results: list[FitResult]) -> tuple[float, float, float, float]:
        if not results:
            return np.nan, np.nan, np.nan, np.nan
        Crrs, composites = zip(*[extract_crr_and_composite(r, rho) for r in results])
        return (
            float(np.median(Crrs)), float(np.std(Crrs)),
            float(np.median(composites)), float(np.std(composites)),
        )

    Crr_med, Crr_std, comp_closed, comp_closed_std = _stats(closed)
    _, _, comp_open, comp_open_std = _stats(opened)

    CdA_closed = compute_CdA(comp_closed, Crr_med, ClA)
    CdA_open = compute_CdA(comp_open, Crr_med, ClA)

    # Simple quadrature for std propagation
    CdA_closed_std = np.sqrt(comp_closed_std**2 + (Crr_std * ClA)**2 + (Crr_med * ClA_std)**2)
    CdA_open_std = np.sqrt(comp_open_std**2 + (Crr_std * ClA)**2 + (Crr_med * ClA_std)**2)
    delta_std = np.sqrt(CdA_closed_std**2 + CdA_open_std**2)

    return AeroParams(
        CdA=CdA_closed,
        CdA_std=CdA_closed_std,
        ClA=ClA,
        ClA_std=ClA_std,
        Crr=Crr_med,
        Crr_std=Crr_std,
        CdA_drs_open=CdA_open,
        CdA_drs_open_std=CdA_open_std,
        delta_CdA=CdA_open - CdA_closed,
        delta_CdA_std=delta_std,
    )
