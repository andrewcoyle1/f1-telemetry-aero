"""
Gear-stratified falsification test for Yas Marina 2024.

The report's aero polar has an anomaly at Yas Marina (lowest CdA but
second-highest ClA).  The senior-engineer review flagged that the gear-8
stratification test — used to rule out engine-braking contamination at
Jeddah — was never run at Yas Marina.  This script closes that gap.

Hypothesis under test
---------------------
If engine braking contaminates coast-down segments, it adds a speed-
dependent retarding force that is largest in low gears (high engine rpm
per unit road speed).  Gear-8 segments operate at 15-20× higher aero
drag relative to engine-braking torque, so:

  H0 (null, no contamination): alpha_gear8 ≈ alpha_all
  H1 (contamination present):  alpha_gear8 < alpha_all

The 90% bootstrap CI on alpha_all is used as the decision threshold.
If (alpha_gear8 - alpha_all) lies within the CI half-width, H0 is not
rejected and engine braking is ruled out as the dominant inflating term.

Usage
-----
  python scripts/yas_marina_gear_strat.py

Output is also saved to results/yas_marina_gear_strat.txt.
"""

import sys
import os

# Allow imports from the project root regardless of working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import fastf1

from src.segments import extract_coastdown_segments, segment_drs_state
from src.ode_fit import fit_segment, fit_segments_pooled
from src.aero_params import air_density, car_mass
from src.uncertainty import bootstrap_alpha_ci

# ---------------------------------------------------------------------------
# Constants (match recompute_report_numbers.py)
# ---------------------------------------------------------------------------
BETA_FIXED    = 120.0   # N  — rolling-resistance force held fixed (median value)
MIN_R2        = 0.90    # minimum per-segment R^2 to accept a fit
DRIVER        = "14"    # Fernando Alonso (Aston Martin)
YEAR          = 2024
CIRCUIT       = "Abu Dhabi"   # FastF1 event name for Yas Marina
SESSIONS      = ["FP1", "FP2", "FP3"]

N_BOOT        = 1000
BOOT_CI       = 0.90
BOOT_SEED     = 42

CACHE_DIR     = os.path.join(_REPO_ROOT, "cache")
RESULTS_DIR   = os.path.join(_REPO_ROOT, "results")
OUTPUT_FILE   = os.path.join(RESULTS_DIR, "yas_marina_gear_strat.txt")


# ---------------------------------------------------------------------------
# Cache check
# ---------------------------------------------------------------------------
def _session_cache_path(year: int, event_dir_fragment: str, session_name: str) -> str | None:
    """
    Return the cache subdirectory for a given session if it exists, else None.
    FastF1 stores sessions under cache/<year>/<date>_<event_name>/<date>_<session_name>.
    We search by matching the event and session substrings case-insensitively.
    """
    year_dir = os.path.join(CACHE_DIR, str(year))
    if not os.path.isdir(year_dir):
        return None
    for event_dir in sorted(os.listdir(year_dir)):
        if event_dir_fragment.lower() in event_dir.lower():
            event_path = os.path.join(year_dir, event_dir)
            if not os.path.isdir(event_path):
                continue
            for sess_dir in sorted(os.listdir(event_path)):
                # FastF1 maps "FP1" -> "Practice_1", "FP2" -> "Practice_2", etc.
                practice_map = {
                    "FP1": "Practice_1",
                    "FP2": "Practice_2",
                    "FP3": "Practice_3",
                    "Q":   "Qualifying",
                }
                target = practice_map.get(session_name, session_name)
                if target.lower() in sess_dir.lower():
                    return os.path.join(event_path, sess_dir)
    return None


def check_cache(year: int, circuit_fragment: str, sessions: list[str]) -> dict[str, str | None]:
    """Return {session_name: cache_path_or_None} for each requested session."""
    return {s: _session_cache_path(year, circuit_fragment, s) for s in sessions}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def collect_yas_marina_segments(
    year: int, circuit: str, driver: str, sessions: list[str]
) -> tuple[list[pd.DataFrame], list, list[float], list[bool], list[int], float]:
    """
    Load practice sessions, extract coast-down segments, run per-segment fits,
    and return only those that pass the R^2 threshold.

    Returns
    -------
    segs      : list of segment DataFrames
    results   : list of FitResult
    ms        : per-segment car mass (kg)
    drs       : per-segment DRS state (bool)
    laps_n    : per-segment lap number (int)
    rho_mean  : mean air density across sessions (kg/m^3)
    gears     : median gear for each segment (int)
    """
    from src.ode_fit import FitResult  # local import to keep top-level clean

    segs, results, ms, drs_states, laps_n, gears = [], [], [], [], [], []
    rho_vals = []

    for sname in sessions:
        sess = fastf1.get_session(year, circuit, sname)
        sess.load(telemetry=True, weather=True)

        rho_s = air_density(
            sess.weather_data["AirTemp"].mean(),
            sess.weather_data["Pressure"].mean(),
        )
        rho_vals.append(rho_s)

        dl = sess.laps.pick_drivers(driver)
        dl = dl[dl["LapNumber"] > 1]   # skip formation / out-lap

        for _, lap in dl.iterrows():
            lap_num = int(lap["LapNumber"])
            m = car_mass(lap_num)
            try:
                tel = lap.get_telemetry()
            except Exception:
                continue

            for seg in extract_coastdown_segments(
                tel,
                min_duration=0.5,
                min_speed_kmh=120.0,
                throttle_threshold=5.0,
            ):
                # Entry-speed and Dv filters (match Jeddah criteria)
                v_entry = seg["Speed"].iloc[0]     # km/h
                v_exit  = seg["Speed"].iloc[-1]    # km/h
                dv_ms   = (v_entry - v_exit) / 3.6  # m/s

                if v_entry < 180.0:   # entry speed > 180 km/h
                    continue
                if dv_ms < 25.0:      # Dv > 25 m/s
                    continue

                drs_open = segment_drs_state(seg)
                r = fit_segment(
                    seg, m, rho_s, drs_open, lap_num, beta_fixed=BETA_FIXED
                )
                if r is None or r.r2 < MIN_R2:
                    continue

                gear = int(seg["nGear"].median()) if "nGear" in seg.columns else -1

                segs.append(seg)
                results.append(r)
                ms.append(m)
                drs_states.append(drs_open)
                laps_n.append(lap_num)
                gears.append(gear)

    rho_mean = float(np.mean(rho_vals)) if rho_vals else float("nan")
    return segs, results, ms, drs_states, laps_n, rho_mean, gears


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis() -> str:
    """
    Execute the gear-stratified test and return the full report as a string.
    """
    lines: list[str] = []

    def _p(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    _p("=" * 70)
    _p("Gear-stratified falsification test — Yas Marina 2024")
    _p("Driver: Fernando Alonso (#14, Aston Martin)")
    _p("=" * 70)

    # --- 1. Cache check ---------------------------------------------------
    _p("\nStep 1: Cache check")
    _p("-" * 40)
    available = check_cache(YEAR, "Abu_Dhabi", SESSIONS)
    all_cached = True
    for sname in SESSIONS:
        path = available[sname]
        if path is not None:
            _p(f"  {sname}: FOUND  ({path})")
        else:
            _p(f"  {sname}: NOT FOUND")
            all_cached = False

    if not all_cached:
        _p()
        _p("ERROR: One or more Yas Marina 2024 practice sessions are not cached.")
        _p("To download, run the following in a Python session:")
        _p("  import fastf1")
        _p(f"  fastf1.Cache.enable_cache('{CACHE_DIR}')")
        for sname in SESSIONS:
            if available[sname] is None:
                _p(f"  sess = fastf1.get_session(2024, 'Abu Dhabi', '{sname}')")
                _p("  sess.load(telemetry=True, weather=True)")
        _p()
        _p("Re-run this script after the data has been downloaded.")
        return "\n".join(lines)

    _p("  All three practice sessions are cached — proceeding.")

    # --- 2. Load segments -------------------------------------------------
    _p()
    _p("Step 2: Loading sessions and extracting coast-down segments")
    _p("  Criteria: entry speed > 180 km/h, Dv > 25 m/s, throttle < 5%, brake = 0")
    _p("-" * 40)

    fastf1.Cache.enable_cache(CACHE_DIR)
    segs, results, ms, drs_states, laps_n, rho_mean, gears = collect_yas_marina_segments(
        YEAR, CIRCUIT, DRIVER, SESSIONS
    )

    n_total = len(segs)
    _p(f"  Total segments passing filters: {n_total}")
    _p(f"  Mean air density rho:           {rho_mean:.4f} kg/m^3")

    if n_total < 2:
        _p()
        _p("ERROR: Fewer than 2 segments passed the filter criteria.")
        _p("Cannot run pooled fit.  Check that driver 14 laps are present in the data.")
        return "\n".join(lines)

    # Gear distribution
    from collections import Counter
    gear_counts = Counter(gears)
    _p(f"  Gear distribution: { {g: gear_counts[g] for g in sorted(gear_counts)} }")

    # --- 3. All-segment pooled fit ----------------------------------------
    _p()
    _p("Step 3: Pooled fit — ALL segments")
    _p("-" * 40)
    pooled_all = fit_segments_pooled(
        segs, ms, drs_states, laps_n,
        seed_results=results,
        beta_fixed=BETA_FIXED,
        fit_v0=False,
    )
    if pooled_all is None:
        _p("ERROR: Pooled fit over all segments failed to converge.")
        return "\n".join(lines)

    alpha_all     = pooled_all[0].alpha
    alpha_all_std = pooled_all[0].alpha_std
    _p(f"  alpha_all     = {alpha_all:.4f}  (Jacobian std = +/-{alpha_all_std:.4f})")

    # --- 4. Gear-8-only fit -----------------------------------------------
    _p()
    _p("Step 4: Pooled fit — gear-8 segments only")
    _p("-" * 40)
    g8_idx = [i for i, g in enumerate(gears) if g == 8]
    n_gear8 = len(g8_idx)
    _p(f"  Gear-8 segments: {n_gear8} / {n_total}")

    alpha_gear8 = float("nan")
    pct_diff    = float("nan")
    fit_gear8_ok = False

    if n_gear8 < 2:
        _p("  WARNING: Fewer than 2 gear-8 segments — cannot run gear-8-only fit.")
        _p("  Hypothesis test is INCONCLUSIVE for gear-8 stratification.")
    else:
        pooled_g8 = fit_segments_pooled(
            [segs[i]      for i in g8_idx],
            [ms[i]        for i in g8_idx],
            [drs_states[i] for i in g8_idx],
            [laps_n[i]    for i in g8_idx],
            seed_results=[results[i] for i in g8_idx],
            beta_fixed=BETA_FIXED,
            fit_v0=False,
        )
        if pooled_g8 is None:
            _p("  WARNING: Gear-8-only pooled fit failed to converge.")
        else:
            alpha_gear8  = pooled_g8[0].alpha
            pct_diff     = 100.0 * (alpha_gear8 - alpha_all) / alpha_all
            fit_gear8_ok = True
            _p(f"  alpha_gear8   = {alpha_gear8:.4f}")
            _p(f"  Relative diff = {pct_diff:+.2f}%  "
               f"(alpha_gear8 - alpha_all) / alpha_all")

    # --- 5. Bootstrap CI on alpha_all ------------------------------------
    _p()
    _p(f"Step 5: Segment-resampling bootstrap ({N_BOOT} replicates, {int(BOOT_CI*100)}% CI)")
    _p("-" * 40)
    ci_lo, ci_hi = bootstrap_alpha_ci(
        segs, ms, drs_states, laps_n,
        beta_fixed=BETA_FIXED,
        seed_results=results,
        n_boot=N_BOOT,
        ci=BOOT_CI,
        seed=BOOT_SEED,
    )
    if not np.isfinite(ci_lo):
        _p("  WARNING: Bootstrap failed (too few replicates converged).")
        ci_lo = ci_hi = float("nan")
    else:
        boot_hw = (ci_hi - ci_lo) / 2.0
        _p(f"  Bootstrap {int(BOOT_CI*100)}% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
        _p(f"  Half-width                = +/-{boot_hw:.4f}")
        _p(f"  Jacobian std              = +/-{alpha_all_std:.4f}   "
           f"ratio = {boot_hw/alpha_all_std:.1f}x")

    # --- 6. Decision / summary -------------------------------------------
    _p()
    _p("=" * 70)
    _p("SUMMARY")
    _p("=" * 70)
    _p(f"  Circuit           : Yas Marina 2024 (Abu Dhabi GP weekend)")
    _p(f"  Driver            : Fernando Alonso (#14, Aston Martin)")
    _p(f"  Sessions          : FP1 + FP2 + FP3")
    _p(f"  n_total segments  : {n_total}")
    _p(f"  n_gear8 segments  : {n_gear8}")
    _p(f"  alpha_all         : {alpha_all:.4f}  N.s^2/m^2")
    if fit_gear8_ok:
        _p(f"  alpha_gear8       : {alpha_gear8:.4f}  N.s^2/m^2")
        _p(f"  % difference      : {pct_diff:+.2f}%")
    else:
        _p(f"  alpha_gear8       : n/a (insufficient segments or fit failed)")
        _p(f"  % difference      : n/a")
    if np.isfinite(ci_lo):
        _p(f"  Bootstrap {int(BOOT_CI*100)}% CI  : [{ci_lo:.4f}, {ci_hi:.4f}]")
        boot_hw = (ci_hi - ci_lo) / 2.0
        _p(f"  CI half-width     : +/-{boot_hw:.4f}")
    else:
        _p(f"  Bootstrap {int(BOOT_CI*100)}% CI  : n/a")
        boot_hw = float("nan")
    _p()

    if fit_gear8_ok and np.isfinite(ci_lo):
        diff_abs  = abs(alpha_gear8 - alpha_all)
        within_ci = diff_abs <= boot_hw
        _p("  VERDICT:")
        if within_ci:
            _p(f"    |alpha_gear8 - alpha_all| = {diff_abs:.4f} <= CI half-width {boot_hw:.4f}")
            _p("    The gear-8/all difference is WITHIN the bootstrap 90% CI.")
            _p("    -> Engine braking is NOT a significant inflating term at Yas Marina.")
            _p("    -> H0 (no contamination) is NOT rejected.")
            _p("    -> The Yas Marina anomaly in the aero polar cannot be attributed")
            _p("       to engine-braking contamination of coast-down segments.")
        else:
            _p(f"    |alpha_gear8 - alpha_all| = {diff_abs:.4f} > CI half-width {boot_hw:.4f}")
            _p("    The gear-8/all difference is OUTSIDE the bootstrap 90% CI.")
            _p("    -> There is statistical evidence of engine-braking contamination.")
            _p("    -> H1 (contamination present) cannot be ruled out.")
            _p("    -> Consider excluding the Yas Marina point from the aero polar.")
    elif not fit_gear8_ok:
        _p("  VERDICT: INCONCLUSIVE — gear-8 fit unavailable.")
        _p("    Consider using a lower-gear threshold (e.g. gear >= 7) or")
        _p("    excluding the Yas Marina point from the aero polar.")
    else:
        _p("  VERDICT: INCONCLUSIVE — bootstrap CI unavailable.")

    _p()
    _p("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fastf1.Cache.enable_cache(CACHE_DIR)

    report = run_analysis()

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(report)
        fh.write("\n")

    print(f"\nResults saved to: {OUTPUT_FILE}")
