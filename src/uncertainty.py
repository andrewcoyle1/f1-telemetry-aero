"""
Monte Carlo uncertainty propagation for aero parameter estimates.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.ode_fit import FitResult, fit_segments_pooled

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


def bootstrap_alpha_ci(
    segments: list[pd.DataFrame],
    m_list: list[float],
    drs_states: list[bool],
    lap_numbers: list[int],
    beta_fixed: float,
    seed_results: list[FitResult] | None = None,
    n_boot: int = 1000,
    ci: float = 0.90,
    seed: int = 42,
    fit_v0: bool = False,
) -> tuple[float, float]:
    """
    Segment-resampling bootstrap CI for pooled α.

    Resamples N segments with replacement, re-runs fit_segments_pooled on each
    replicate, and returns the (lower, upper) percentile bounds at the requested
    confidence level.  Use this instead of the Jacobian-derived std when
    Durbin-Watson is well below 2, as positive residual autocorrelation makes
    the covariance matrix anti-conservative.

    fit_v0 should match the setting used for the primary fit whose CI is being
    characterised (default False, consistent with recompute_report_numbers.py).

    Returns (ci_lo, ci_hi).  Returns (nan, nan) if fewer than half the bootstrap
    replicates converge successfully.
    """
    rng = np.random.default_rng(seed)
    n = len(segments)
    alpha_boot: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        segs_b = [segments[i] for i in idx]
        ms_b   = [m_list[i]   for i in idx]
        drs_b  = [drs_states[i] for i in idx]
        laps_b = [lap_numbers[i] for i in idx]
        seeds_b = [seed_results[i] for i in idx] if seed_results else None

        result = fit_segments_pooled(
            segs_b, ms_b, drs_b, laps_b,
            seed_results=seeds_b,
            beta_fixed=beta_fixed,
            fit_v0=fit_v0,
        )
        if result is not None:
            alpha_boot.append(result[0].alpha)

    if len(alpha_boot) < n_boot // 2:
        return float("nan"), float("nan")

    lo = (1.0 - ci) / 2.0
    hi = 1.0 - lo
    arr = np.array(alpha_boot)
    return float(np.percentile(arr, lo * 100)), float(np.percentile(arr, hi * 100))


def bootstrap_alpha_ci_session_stratified(
    segments: list[pd.DataFrame],
    m_list: list[float],
    drs_states: list[bool],
    lap_numbers: list[int],
    beta_fixed: float,
    session_ids: list[str],
    seed_results: list[FitResult] | None = None,
    n_boot: int = 1000,
    ci: float = 0.90,
    seed: int = 42,
    fit_v0: bool = False,
) -> tuple[float, float]:
    """
    Session-stratified bootstrap CI for pooled alpha.

    Segments within the same FP session share systematic effects (track state,
    ambient temperature, session-level fuel load), making them non-exchangeable.
    This function resamples whole sessions with replacement rather than
    individual segments, producing a between-session CI that properly accounts
    for within-session correlation.

    On each bootstrap replicate, ``len(unique_sessions)`` session labels are
    drawn with replacement.  All segments belonging to each sampled session are
    then collected into the replicate pool (a session drawn k times contributes
    its segments k times).  ``fit_segments_pooled`` is then run on the pool and
    the shared alpha is recorded.

    Parameters
    ----------
    segments:
        Coast-down segment DataFrames (one per segment).
    m_list:
        Vehicle mass (kg) for each segment, aligned with ``segments``.
    drs_states:
        DRS open/closed flag for each segment.
    lap_numbers:
        Lap number for each segment.
    beta_fixed:
        Rolling-resistance force (N) held fixed across all fits.
    session_ids:
        Session label for each segment (e.g. ``['FP1', 'FP1', 'FP2', ...]``).
        Must be the same length as ``segments``.
    seed_results:
        Per-segment warm-start FitResult list, aligned with ``segments``.
        Passed through to ``fit_segments_pooled`` after resampling.
    n_boot:
        Number of bootstrap replicates.
    ci:
        Confidence level (default 0.90 gives a 90% CI).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (ci_lo, ci_hi)
        Percentile confidence interval for alpha.  Returns ``(nan, nan)`` if
        fewer than two unique sessions are present (bootstrap is degenerate)
        or if fewer than half of the replicates converge.
    """
    if len(segments) != len(session_ids):
        raise ValueError(
            "segments and session_ids must have the same length; "
            f"got {len(segments)} and {len(session_ids)}"
        )

    unique_sessions = list(dict.fromkeys(session_ids))  # preserve encounter order
    n_sessions = len(unique_sessions)

    if n_sessions < 2:
        warnings.warn(
            "bootstrap_alpha_ci_session_stratified: only one unique session found "
            f"({unique_sessions[0]!r}).  A session-level bootstrap is degenerate "
            "with a single session — returning (nan, nan).  "
            "Consider using bootstrap_alpha_ci() for within-session CIs.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan"), float("nan")

    # Build a lookup from session label -> list of integer indices into segments.
    session_index: dict[str, list[int]] = {s: [] for s in unique_sessions}
    for i, sid in enumerate(session_ids):
        session_index[sid].append(i)

    rng = np.random.default_rng(seed)
    alpha_boot: list[float] = []

    # Cap retries to avoid an infinite loop when the data are systematically
    # unfittable.  Allow up to 5x n_boot total attempts.
    max_attempts = n_boot * 5
    attempts = 0

    while len(alpha_boot) < n_boot and attempts < max_attempts:
        attempts += 1

        # Draw n_sessions session labels with replacement.
        sampled_sessions = rng.choice(unique_sessions, size=n_sessions, replace=True)

        # Collect segment indices for the replicate pool.
        idx: list[int] = []
        for sid in sampled_sessions:
            idx.extend(session_index[sid])

        # Require at least 2 segments to attempt a pooled fit.
        if len(idx) < 2:
            continue

        segs_b  = [segments[i]    for i in idx]
        ms_b    = [m_list[i]      for i in idx]
        drs_b   = [drs_states[i]  for i in idx]
        laps_b  = [lap_numbers[i] for i in idx]
        seeds_b = [seed_results[i] for i in idx] if seed_results else None

        result = fit_segments_pooled(
            segs_b, ms_b, drs_b, laps_b,
            seed_results=seeds_b,
            beta_fixed=beta_fixed,
            fit_v0=fit_v0,
        )
        if result is not None:
            alpha_boot.append(result[0].alpha)

    if len(alpha_boot) < n_boot // 2:
        return float("nan"), float("nan")

    lo = (1.0 - ci) / 2.0
    hi = 1.0 - lo
    arr = np.array(alpha_boot)
    return float(np.percentile(arr, lo * 100)), float(np.percentile(arr, hi * 100))


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
