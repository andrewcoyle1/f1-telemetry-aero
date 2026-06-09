"""
Deming (errors-in-variables / orthogonal) regression for the F1 aero polar.

The OLS polar fit regresses CLA on CDA, but CDA carries substantial measurement
error (~0.1-0.2 m² after the 13× bootstrap-scaling correction to the Jacobian
bounds). When the x-axis regressor has non-negligible error relative to its
range, OLS is inconsistent and attenuates the slope toward zero. Deming
regression accounts for errors in both axes and gives an unbiased slope
estimate in this setting.

Reference: Fuller (1987), Measurement Error Models; Linnet (1990).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core Deming regression
# ---------------------------------------------------------------------------

def deming_regression(
    x: np.ndarray,
    y: np.ndarray,
    sigma_x: np.ndarray | float,
    sigma_y: np.ndarray | float,
) -> tuple[float, float, float, float]:
    """
    Deming regression with per-point error weights.

    For each point the local variance ratio is:
        lambda_i = (sigma_x_i / sigma_y_i)^2

    When sigma_x and sigma_y are scalars the classical Deming formula is
    recovered exactly (lambda = constant).  With per-point sigmas the
    estimator uses the mean lambda across points, which is a common practical
    approximation that preserves the closed-form solution while capturing the
    overall scale of x-error relative to y-error.

    Parameters
    ----------
    x, y : 1-D arrays of length n
    sigma_x, sigma_y : per-point standard deviations (arrays or scalars)

    Returns
    -------
    slope : Deming slope estimate
    intercept : Deming intercept estimate
    slope_std_boot : bootstrap standard deviation of the slope (10 000 resamples)
    intercept_std_boot : bootstrap standard deviation of the intercept
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = np.broadcast_to(np.asarray(sigma_x, dtype=float), x.shape).copy()
    sy = np.broadcast_to(np.asarray(sigma_y, dtype=float), y.shape).copy()

    slope, intercept = _deming_fit(x, y, sx, sy)
    slope_std, intercept_std = _deming_bootstrap(x, y, sx, sy, n_boot=10_000, seed=42)

    return slope, intercept, slope_std, intercept_std


def _deming_fit(
    x: np.ndarray,
    y: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
) -> tuple[float, float]:
    """
    Closed-form Deming slope using mean variance ratio lambda = mean(sx^2/sy^2).

    The closed-form solution (Carroll et al. 2006, eq. 4.6) is:
        beta = [S_yy - lambda*S_xx + sqrt((S_yy - lambda*S_xx)^2 + 4*lambda*S_xy^2)]
               / (2 * S_xy)

    where S_xx, S_yy, S_xy are the corrected sum-of-squares/cross-products of
    x and y, and lambda is the (mean) variance ratio.
    """
    n = len(x)
    # Use mean lambda across all points
    lam = float(np.mean((sigma_x / sigma_y) ** 2))

    xm = x.mean()
    ym = y.mean()
    Sxx = np.sum((x - xm) ** 2)
    Syy = np.sum((y - ym) ** 2)
    Sxy = np.sum((x - xm) * (y - ym))

    discriminant = (Syy - lam * Sxx) ** 2 + 4.0 * lam * Sxy ** 2
    slope = (Syy - lam * Sxx + np.sqrt(discriminant)) / (2.0 * Sxy)
    intercept = ym - slope * xm
    return float(slope), float(intercept)


def _deming_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap standard deviation of Deming slope and intercept.
    Resamples (x_i, y_i, sx_i, sy_i) jointly with replacement.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = []
    intercepts = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        sxb = sigma_x[idx]
        syb = sigma_y[idx]
        # Guard against degenerate resamples (all identical x)
        if np.std(xb) < 1e-12:
            continue
        s, c = _deming_fit(xb, yb, sxb, syb)
        slopes.append(s)
        intercepts.append(c)

    return float(np.std(slopes)), float(np.std(intercepts))


# ---------------------------------------------------------------------------
# Main: hard-coded five-circuit polar data from Table 3
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Five-circuit data from Table 3 (Results.tex) ────────────────────────
    # CDA point estimates (m²)
    cda = np.array([1.448, 1.376, 1.487, 1.335, 1.689])

    # CLA point estimates (m²)
    cla = np.array([2.66, 3.06, 3.14, 3.53, 4.33])

    # CLA total sigma (statistical SE of median ⊕ mu-systematic) in m²
    cla_sig = np.array([0.42, 0.48, 0.51, 0.55, 0.67])

    # CDA sigma: bootstrap-corrected (13× Jacobian scaling, recomputed from
    # the recompute_report_numbers.py pipeline values).
    #
    # Jacobian CDA sigma = hypot(2*alpha_std/rho, Crr*cla_sig_total)
    # From Table 3 the dagger values are the Jacobian bounds:
    #   Jeddah:     0.009  => alpha_std=0.004, rho=1.180  => 2*0.004/1.180=0.00678; Crr*sig≈0.014*0.42=0.0059 => hypot≈0.0089
    #   Spa:        0.014  => alpha_std=0.007, rho=1.145  => 2*0.007/1.145=0.01223; Crr*sig≈0.014*0.48=0.0067 => hypot≈0.0140
    #   Silverstone:0.010  => alpha_std=0.004, rho=1.193  => 2*0.004/1.193=0.00671; Crr*sig≈0.014*0.51=0.0071 => hypot≈0.0098
    #   Yas Marina: 0.016  => alpha_std=0.008, rho=1.178  => 2*0.008/1.178=0.01358; Crr*sig≈0.014*0.55=0.0077 => hypot≈0.0157
    #   Suzuka:     0.017  => alpha_std=0.008, rho=1.223  => 2*0.008/1.223=0.01308; Crr*sig≈0.014*0.67=0.0094 => hypot≈0.0161
    #
    # Bootstrap-scaled = Jacobian * 13:
    cda_sig_jac = np.array([0.009, 0.014, 0.010, 0.016, 0.017])
    BOOT_RATIO = 13.0
    cda_sig = cda_sig_jac * BOOT_RATIO

    # ── OLS baseline ────────────────────────────────────────────────────────
    xm, ym = cda.mean(), cla.mean()
    ols_slope = np.sum((cda - xm) * (cla - ym)) / np.sum((cda - xm) ** 2)
    ols_intercept = ym - ols_slope * xm

    # ── Deming regression ───────────────────────────────────────────────────
    d_slope, d_intercept, d_slope_std, d_intercept_std = deming_regression(
        cda, cla, cda_sig, cla_sig
    )

    ci_lo = d_slope - 1.645 * d_slope_std   # 90% CI (5th percentile)
    ci_hi = d_slope + 1.645 * d_slope_std   # 90% CI (95th percentile)

    # Also compute bootstrap percentile CI directly
    rng = np.random.default_rng(42)
    n = len(cda)
    boot_slopes = []
    for _ in range(10_000):
        idx = rng.integers(0, n, size=n)
        xb, yb = cda[idx], cla[idx]
        sxb, syb = cda_sig[idx], cla_sig[idx]
        if np.std(xb) < 1e-12:
            continue
        s, _ = _deming_fit(xb, yb, sxb, syb)
        boot_slopes.append(s)
    boot_slopes = np.array(boot_slopes)
    p5  = float(np.percentile(boot_slopes, 5))
    p95 = float(np.percentile(boot_slopes, 95))

    # ── Print results ────────────────────────────────────────────────────────
    print("=" * 65)
    print("F1 Aero Polar — OLS vs Deming Regression")
    print("=" * 65)
    print(f"\nData (5 circuits):")
    circuits = ["Jeddah", "Spa", "Silverstone", "Yas Marina", "Suzuka"]
    for i, name in enumerate(circuits):
        print(f"  {name:12s}  CDA={cda[i]:.3f}±{cda_sig[i]:.3f}  "
              f"CLA={cla[i]:.2f}±{cla_sig[i]:.2f}")

    print(f"\nVariance ratio lambda (mean) = "
          f"{float(np.mean((cda_sig/cla_sig)**2)):.4f}")
    print(f"  (sigma_x/sigma_y ranges from "
          f"{float(np.min(cda_sig/cla_sig)):.3f} to "
          f"{float(np.max(cda_sig/cla_sig)):.3f})")
    print(f"  CDA sigma range: {cda_sig.min():.3f}–{cda_sig.max():.3f} m²  "
          f"(vs CDA range {cda.max()-cda.min():.3f} m²)")

    print(f"\nOLS slope:             {ols_slope:.3f}  (intercept={ols_intercept:.3f})")
    print(f"Deming slope:          {d_slope:.3f}  (intercept={d_intercept:.3f})")
    print(f"Deming slope std:      {d_slope_std:.3f}  (bootstrap, 10 000 resamples)")
    print(f"Deming 90% CI (Gauss): [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"Deming 90% CI (pctl):  [{p5:.2f}, {p95:.2f}]")
    print(f"\nAttenuation (OLS/Deming - 1): {100*(ols_slope/d_slope - 1):+.1f}%")
    print(f"  (positive → OLS underestimates slope due to x-axis measurement error)")
    print("=" * 65)
