"""
Beta (rolling resistance) sensitivity analysis for Jeddah 2024.

Addresses review issue #1:
    "The prior beta = 120 N ... if the prior is wrong by 20%, both alpha
     and P_mgu absorb the error."

This script sweeps beta_fixed over a plausible range [80, 100, 120, 140, 160] N,
re-runs the pooled coast-down fit at Jeddah 2024 FP1+FP2+FP3 for each value,
and reports how CD_A changes.

Physical model recap:
    m * dv/dt = -alpha * v^2  -  beta  -  P_mgu / v

    alpha   [N*s^2/m^2]  aerodynamic drag coefficient
    beta    [N]          rolling resistance  =  Crr * m * g
    P_mgu   [W]          MGU-K harvest power (per segment)

From alpha we compute:
    composite  =  2 * alpha / rho          [m^2]   =  CdA + Crr * ClA
    Crr        =  beta / (m * g)                   (rolling resistance coefficient)
    CdA        =  composite  -  Crr * ClA          [m^2]

Fixed constants (Jeddah 2024):
    rho  = 1.180 kg/m^3   (mean over FP1-FP3 weather data; hard-coded here so
                           each beta run is evaluated at the same air density)
    m    = 855 kg         (canonical mass for Crr formula; ~lap-21 equivalent:
                           798 + max(0, 95 - 1.8*21) ≈ 855 kg)
    ClA  = 2.66 m^2       (from qualifying CLA estimate — used in the main report)
    g    = 9.81 m/s^2
"""

from __future__ import annotations

import os
import sys

# Allow imports from the project root (src/ package)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

# ── FastF1 cache must be enabled BEFORE any session is loaded ─────────────────
import fastf1
_CACHE_DIR = os.path.join(_REPO_ROOT, "cache")
fastf1.Cache.enable_cache(_CACHE_DIR)

from src.segments import extract_coastdown_segments, segment_drs_state
from src.ode_fit import fit_segment, fit_segments_pooled
from src.aero_params import air_density, car_mass

# ── Constants ─────────────────────────────────────────────────────────────────
DRIVER        = "14"         # Fernando Alonso / Aston Martin
YEAR          = 2024
CIRCUIT       = "Saudi Arabia"
FP_SESSIONS   = ["FP1", "FP2", "FP3"]

# Jeddah mean air density from FP1+FP2+FP3 weather data (computed below,
# but this fallback is used if weather data is unavailable).
RHO_JEDDAH    = 1.180        # kg/m^3

G             = 9.81         # m/s^2
CLA           = 2.66         # m^2   (from main-report qualifying CLA estimate)
M_REF         = 855.0        # kg    canonical mass for Crr = beta/(m*g), as specified
                              #       (corresponds to ~lap 21: 798 + 95 - 1.8*21 ≈ 855 kg)

# Beta values to sweep (N).  120 N is the baseline used throughout the main report.
BETA_SWEEP    = [80.0, 100.0, 120.0, 140.0, 160.0]
BETA_BASELINE = 120.0

# Segment-quality filter (same as main pipeline)
MIN_R2        = 0.90

# Output directory
RESULTS_DIR   = os.path.join(_REPO_ROOT, "results")
OUTPUT_CSV    = os.path.join(RESULTS_DIR, "beta_sensitivity_jeddah.csv")


# ── Cache presence check ───────────────────────────────────────────────────────
def check_jeddah_cache() -> bool:
    """
    Verify that the FastF1 session cache files for Jeddah 2024 FP1/FP2/FP3
    are present on disk before attempting to load them.

    FastF1 stores cached sessions under:
        cache/<year>/<date>_<event>/<date>_<session_type>/

    We search for the Saudi Arabian Grand Prix directory and confirm each
    practice session sub-folder exists and is non-empty.

    Returns True if all three sessions are found, False otherwise.
    """
    year_dir = os.path.join(_CACHE_DIR, str(YEAR))
    if not os.path.isdir(year_dir):
        print(f"  [cache] Year directory not found: {year_dir}")
        return False

    # Locate the Saudi Arabia event folder (date prefix varies by calendar year)
    saudi_dirs = [
        d for d in os.listdir(year_dir)
        if "saudi" in d.lower() or "jeddah" in d.lower()
    ]
    if not saudi_dirs:
        print(f"  [cache] No Saudi Arabia event folder found under {year_dir}")
        return False

    event_dir = os.path.join(year_dir, saudi_dirs[0])
    print(f"  [cache] Found event directory: {event_dir}")

    # Map FastF1 session names to the folder name fragments used on disk
    session_fragments = {
        "FP1": "practice_1",
        "FP2": "practice_2",
        "FP3": "practice_3",
    }

    all_found = True
    for sname in FP_SESSIONS:
        frag = session_fragments[sname]
        matches = [
            d for d in os.listdir(event_dir)
            if frag in d.lower()
        ]
        if matches:
            sess_dir = os.path.join(event_dir, matches[0])
            n_files  = len(os.listdir(sess_dir))
            print(f"  [cache] {sname} -> {matches[0]}  ({n_files} files)")
        else:
            print(f"  [cache] {sname} -> NOT FOUND  (expected fragment '{frag}' in {event_dir})")
            all_found = False

    return all_found


# ── Data loading ───────────────────────────────────────────────────────────────
def load_jeddah_segments() -> tuple[
    list[pd.DataFrame],  # raw coast-down segment DataFrames
    list[float],         # per-segment car mass
    list[bool],          # per-segment DRS state
    list[int],           # per-segment lap number
    float,               # mean air density across the three sessions
]:
    """
    Load Jeddah 2024 FP1+FP2+FP3 telemetry for driver 14, extract coast-down
    segments, and return the lists needed by fit_segments_pooled().

    Uses the same filtering parameters as recompute_report_numbers.py:
        min_duration=0.5 s, min_speed_kmh=120 km/h, throttle_threshold=5%

    Only segments that pass a per-segment pre-filter (fit_segment with any fixed
    beta, min_r2=MIN_R2) are included so the pooled fit receives the same sample
    as the main pipeline.
    """
    segs, ms, drs_states, lap_nums = [], [], [], []
    rho_vals: list[float] = []

    for sname in FP_SESSIONS:
        print(f"  Loading {YEAR} {CIRCUIT} {sname} ...", flush=True)
        try:
            sess = fastf1.get_session(YEAR, CIRCUIT, sname)
            sess.load(telemetry=True, weather=True)
        except Exception as exc:
            print(f"  WARNING: could not load {sname}: {exc}")
            continue

        # Air density from session weather
        rho_s = air_density(
            sess.weather_data["AirTemp"].mean(),
            sess.weather_data["Pressure"].mean(),
        )
        rho_vals.append(rho_s)
        print(f"    rho={rho_s:.4f} kg/m^3", flush=True)

        laps = sess.laps.pick_drivers(DRIVER)
        laps = laps[laps["LapNumber"] > 1]

        for _, lap in laps.iterrows():
            lap_num = int(lap["LapNumber"])
            m       = car_mass(lap_num)
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
                drs_open = segment_drs_state(seg)
                # Pre-filter at the baseline beta to match main pipeline quality gate
                r = fit_segment(
                    seg, m, rho_s, drs_open, lap_num,
                    beta_fixed=BETA_BASELINE,
                )
                if r is not None and r.r2 >= MIN_R2:
                    segs.append(seg)
                    ms.append(m)
                    drs_states.append(drs_open)
                    lap_nums.append(lap_num)

    if not rho_vals:
        raise RuntimeError("No weather data loaded; cannot compute air density.")

    rho_mean = float(np.mean(rho_vals))
    print(f"  Total qualifying segments: {len(segs)}")
    print(f"  Mean rho (FP1+FP2+FP3):   {rho_mean:.4f} kg/m^3")
    return segs, ms, drs_states, lap_nums, rho_mean


# ── Per-beta fit and CdA computation ──────────────────────────────────────────
def compute_cda_for_beta(
    beta: float,
    segs: list[pd.DataFrame],
    ms: list[float],
    drs_states: list[bool],
    lap_nums: list[int],
    rho: float,
) -> dict:
    """
    Run fit_segments_pooled() with the given beta_fixed, then compute:
        alpha_hat   : pooled fitted alpha  [N*s^2/m^2]
        composite   : 2 * alpha_hat / rho  [m^2]  = CdA + Crr*ClA
        Crr         : beta / (M_REF * g)
        CdA         : composite - Crr * ClA

    Returns a dict with all intermediate and final values.
    """
    pooled = fit_segments_pooled(
        segs, ms, drs_states, lap_nums,
        seed_results=None,   # let the optimiser cold-start from defaults
        beta_fixed=beta,
        fit_v0=False,
    )

    if pooled is None:
        return {
            "beta":      beta,
            "alpha_hat": float("nan"),
            "composite": float("nan"),
            "Crr":       float("nan"),
            "CdA":       float("nan"),
            "rel_change": float("nan"),
        }

    alpha_hat = float(pooled[0].alpha)
    composite = 2.0 * alpha_hat / rho
    Crr       = beta / (M_REF * G)
    CdA       = composite - Crr * CLA

    return {
        "beta":      beta,
        "alpha_hat": alpha_hat,
        "composite": composite,
        "Crr":       Crr,
        "CdA":       CdA,
        "rel_change": float("nan"),  # filled in after baseline is known
    }


# ── Pretty-printing ────────────────────────────────────────────────────────────
def print_table(rows: list[dict]) -> None:
    """Print a formatted sensitivity table to stdout."""
    print()
    print("=" * 82)
    print("  Beta sensitivity: Jeddah 2024 FP coast-down  (driver 14, rho=1.180 kg/m^3)")
    print("  CLA = 2.66 m^2,  m_ref = {:.0f} kg,  g = {:.2f} m/s^2".format(M_REF, G))
    print("=" * 82)
    hdr = (
        f"  {'beta (N)':>8}  {'alpha_hat':>10}  {'composite':>10}  "
        f"{'Crr':>7}  {'CdA (m^2)':>10}  {'delta CdA':>10}  {'rel change':>10}"
    )
    print(hdr)
    print("-" * 82)
    for r in rows:
        baseline_marker = " <-- baseline" if r["beta"] == BETA_BASELINE else ""
        rel = r["rel_change"]
        rel_str = f"{rel:+.1f}%" if not np.isnan(rel) else "    n/a"
        delta = r["delta_cda"]
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "    n/a"
        print(
            f"  {r['beta']:>8.0f}  {r['alpha_hat']:>10.5f}  {r['composite']:>10.5f}  "
            f"{r['Crr']:>7.5f}  {r['CdA']:>10.4f}  {delta_str:>10}  {rel_str:>10}"
            f"{baseline_marker}"
        )
    print("=" * 82)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "=" * 70)
    print("  Beta sensitivity analysis — Jeddah 2024 FP coast-down")
    print("=" * 70)

    # 1. Cache presence check
    print("\n[1/4] Checking FastF1 cache ...")
    cache_ok = check_jeddah_cache()
    if not cache_ok:
        print(
            "\nERROR: One or more required FastF1 session files are missing from the cache.\n"
            "       Run the main pipeline (recompute_report_numbers.py) first so the\n"
            "       Jeddah 2024 FP1/FP2/FP3 sessions are downloaded and cached, then\n"
            "       re-run this script.\n"
            "       Cache directory: " + _CACHE_DIR
        )
        sys.exit(1)

    # 2. Load segments
    print("\n[2/4] Loading Jeddah 2024 FP telemetry and extracting segments ...")
    try:
        segs, ms, drs_states, lap_nums, rho = load_jeddah_segments()
    except Exception as exc:
        print(f"\nERROR: Failed to load telemetry data: {exc}")
        sys.exit(1)

    if len(segs) < 2:
        print(
            f"\nERROR: Only {len(segs)} qualifying segment(s) found.  "
            "Need at least 2 for a pooled fit."
        )
        sys.exit(1)

    # Override rho with the hard-coded Jeddah mean for reproducibility.
    # The computed value is printed above for cross-checking.
    rho_used = RHO_JEDDAH
    print(f"  Using rho = {rho_used:.4f} kg/m^3 (hard-coded Jeddah mean)")
    print(f"  (Computed from session weather: {rho:.4f} kg/m^3)")

    # 3. Sweep beta
    print(f"\n[3/4] Sweeping beta over {BETA_SWEEP} N ...")
    raw_rows: list[dict] = []
    for beta in BETA_SWEEP:
        print(f"  Fitting with beta_fixed = {beta:.0f} N ...", flush=True)
        row = compute_cda_for_beta(beta, segs, ms, drs_states, lap_nums, rho_used)
        raw_rows.append(row)

    # Compute relative changes against the baseline (beta = 120 N)
    baseline_row = next((r for r in raw_rows if r["beta"] == BETA_BASELINE), None)
    if baseline_row is None or np.isnan(baseline_row["CdA"]):
        print("\nERROR: Baseline fit (beta=120 N) failed.  Cannot compute relative changes.")
        sys.exit(1)

    cda_baseline = baseline_row["CdA"]
    for r in raw_rows:
        if not np.isnan(r["CdA"]):
            r["delta_cda"] = r["CdA"] - cda_baseline
            r["rel_change"] = 100.0 * (r["CdA"] - cda_baseline) / abs(cda_baseline)
        else:
            r["delta_cda"] = float("nan")
            r["rel_change"] = float("nan")

    # 4. Report
    print("\n[4/4] Results")
    print_table(raw_rows)

    # Save CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame([
        {
            "beta_N":        r["beta"],
            "alpha_hat":     r["alpha_hat"],
            "composite_m2":  r["composite"],
            "Crr":           r["Crr"],
            "CdA_m2":        r["CdA"],
            "delta_CdA_m2":  r["delta_cda"],
            "rel_change_pct": r["rel_change"],
        }
        for r in raw_rows
    ])
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    print(f"  Saved to: {OUTPUT_CSV}")

    # Brief narrative summary for easy transcription
    lo_row = min((r for r in raw_rows if not np.isnan(r["CdA"])), key=lambda r: r["beta"])
    hi_row = max((r for r in raw_rows if not np.isnan(r["CdA"])), key=lambda r: r["beta"])
    print(
        f"\n  Summary: sweeping beta from {lo_row['beta']:.0f} N to {hi_row['beta']:.0f} N\n"
        f"  changes CdA by {lo_row['rel_change']:+.1f}% to {hi_row['rel_change']:+.1f}% "
        f"relative to the beta=120 N baseline.\n"
        f"  Absolute range: {lo_row['CdA']:.4f} - {hi_row['CdA']:.4f} m^2  "
        f"(baseline = {cda_baseline:.4f} m^2)\n"
    )


if __name__ == "__main__":
    main()
