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
    Estimate ClA from high-speed corner samples using tyre friction model:
        m·lat_a = μ·(m·g + ½ρ·ClA·v²)
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

    # ClA = (2m/ρ) * (lat_a/μ - g) / v²
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
