"""
Compute the bootstrap 90% CI on alpha and the post-v0-fit Durbin-Watson
for the Monza 2024 pooled coast-down analysis.

Outputs the two numbers needed to correct §5.1 of the report.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import fastf1
from statsmodels.stats.stattools import durbin_watson

from src.segments import extract_coastdown_segments, segment_drs_state
from src.ode_fit import fit_segment, fit_segments_pooled, v_model, FitResult
from src.aero_params import air_density, car_mass
from src.uncertainty import bootstrap_alpha_ci

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'cache'), exist_ok=True)
fastf1.Cache.enable_cache(os.path.join(os.path.dirname(__file__), '..', 'cache'))

BETA_FIXED = 120.0
MIN_R2     = 0.90
DRIVER     = '14'

print("Loading Monza 2024 FP sessions...")
all_results: list[FitResult] = []
all_segs: list = []
rho_vals = []

for sname in ['FP1', 'FP2', 'FP3']:
    sess = fastf1.get_session(2024, 'Monza', sname)
    sess.load(telemetry=True, weather=True)
    rho_s = air_density(sess.weather_data['AirTemp'].mean(),
                        sess.weather_data['Pressure'].mean())
    rho_vals.append(rho_s)
    laps = sess.laps.pick_drivers(DRIVER)
    laps = laps[laps['LapNumber'] > 1]
    for _, lap in laps.iterrows():
        lap_num = int(lap['LapNumber'])
        m = car_mass(lap_num)
        try:
            tel = lap.get_telemetry()
        except Exception:
            continue
        for seg in extract_coastdown_segments(
                tel, min_duration=0.5, min_speed_kmh=120.0, throttle_threshold=5.0):
            drs_open = segment_drs_state(seg)
            r = fit_segment(seg, m, rho_s, drs_open, lap_num, beta_fixed=BETA_FIXED)
            if r is not None and r.r2 >= MIN_R2:
                all_results.append(r)
                all_segs.append((seg, r, m, lap_num, drs_open))

rho = float(np.mean(rho_vals))
print(f"  {len(all_results)} segments passed R² filter  |  ρ = {rho:.4f} kg/m³")

# ── Pooled fit WITH v0 fitting ────────────────────────────────────────────────
segs_p  = [s for s, *_ in all_segs]
ms_p    = [m for _, _, m, *_ in all_segs]
drs_p   = [d for *_, d in all_segs]
laps_p  = [l for _, _, _, l, _ in all_segs]

pooled_v0 = fit_segments_pooled(
    segs_p, ms_p, drs_p, laps_p,
    seed_results=all_results,
    beta_fixed=BETA_FIXED,
    fit_v0=True,
)

if pooled_v0 is None:
    print("ERROR: pooled fit with v0 failed")
    sys.exit(1)

alpha_hat = pooled_v0[0].alpha
print(f"\nPooled α (with v0 fitting) = {alpha_hat:.4f} N·s²/m²")

# ── Durbin-Watson on v0-fitted residuals ─────────────────────────────────────
dw_vals = []
for i, (seg, _, m, lap_num, _) in enumerate(all_segs):
    t      = seg['t'].values - seg['t'].values[0]
    v_meas = seg['Speed'].values / 3.6
    v_pred = v_model(t, pooled_v0[i].alpha, pooled_v0[i].beta,
                     pooled_v0[i].P_mgu, pooled_v0[i].v0, pooled_v0[i].m)
    dw_vals.append(durbin_watson(v_meas - v_pred))

mean_dw = float(np.mean(dw_vals))
print(f"Mean Durbin-Watson (after v0 fitting) = {mean_dw:.3f}")
print(f"  per-segment DW: min={min(dw_vals):.2f}  max={max(dw_vals):.2f}  "
      f"median={np.median(dw_vals):.2f}")

# ── Bootstrap CI on alpha ─────────────────────────────────────────────────────
print("\nRunning bootstrap (n=1000)...")
ci_lo, ci_hi = bootstrap_alpha_ci(
    segs_p, ms_p, drs_p, laps_p,
    beta_fixed=BETA_FIXED,
    seed_results=all_results,
    n_boot=1000,
    ci=0.90,
    seed=42,
)

half_width = (ci_hi - ci_lo) / 2.0
print(f"Bootstrap 90% CI on α: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Half-width = ±{half_width:.4f} N·s²/m²")
print(f"  Jacobian-derived std = ±{pooled_v0[0].alpha_std:.4f} N·s²/m²")
print(f"  Ratio bootstrap/Jacobian = {half_width / pooled_v0[0].alpha_std:.1f}×")

print("\n── CURRENT VALUES ───────────────────────────────────────────────────────")
print(f"  Table 4.1  Mean DW:       {mean_dw:.2f}")
print(f"  §5.1 text  DW after v0:   {mean_dw:.2f}")
print(f"  §5.1 text  bootstrap CI:  ±{half_width:.3f} N·s²/m²")
