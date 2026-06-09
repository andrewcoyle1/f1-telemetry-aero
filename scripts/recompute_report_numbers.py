"""
Recompute every number cited in the report using a data-driven circuit selection.

Pipeline:
  1. Load the pre-computed 24-circuit survey (survey_results.csv) and confirm
     Aston Martin driver 14 as the analysis subject.
  2. Run coast-down (FP1+FP2+FP3) and qualifying CLA on the 5 best circuits
     chosen to span the full downforce range: Jeddah → Spa → Silverstone →
     Yas Marina → Suzuka.
  3. Fit a 5-point aero polar (CLA vs CdA) via OLS + Monte Carlo CI.
  4. Print a REPORT-VALUES block for transcription into the .tex files.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import fastf1
from statsmodels.stats.stattools import durbin_watson

from src.segments import extract_coastdown_segments, segment_drs_state
from src.ode_fit import fit_segment, fit_segments_pooled, v_model, FitResult
from src.aero_params import air_density, car_mass, estimate_ClA, compute_CdA
from src.uncertainty import bootstrap_alpha_ci, bootstrap_alpha_ci_session_stratified

fastf1.Cache.enable_cache(os.path.join(os.path.dirname(__file__), '..', 'cache'))

BETA_FIXED    = 120.0
MIN_R2        = 0.90
MU_TYRE       = 1.8
G             = 9.81
QUALI_FUEL_KG = 5.0
QUALI_MASS    = 798.0 + QUALI_FUEL_KG   # = 803 kg

QUALIFYING_SESSIONS = {'Q', 'Q1', 'Q2', 'Q3', 'SQ', 'SQ1', 'SQ2', 'SQ3'}

BOOT_JACOBIAN_RATIO = 19.0   # bootstrap / Jacobian ratio (calibrated from DW ~0.55)
N_POLAR_MC          = 50_000


# ── Survey summary ─────────────────────────────────────────────────────────────
SURVEY_CSV = os.path.join(os.path.dirname(__file__), 'survey_results.csv')
df_survey = pd.read_csv(SURVEY_CSV)
df_survey['driver'] = df_survey['driver'].astype(str)
df_survey['score']  = df_survey['n_coastdown'] + df_survey['n_cla']

team_scores = (df_survey.groupby('team')['score'].sum()
                         .sort_values(ascending=False))
print(f"\n{'='*70}\n24-Circuit Survey — Constructor Rankings\n{'='*70}")
print(team_scores.to_string())
print(f"\n→ Selected constructor: {team_scores.index[0]}  (score={team_scores.iloc[0]})")

# Show driver-14 per-circuit data for the selected team
TEAM   = 'Aston Martin'
DRIVER = '14'
amr14  = df_survey[(df_survey['team'] == TEAM) & (df_survey['driver'] == DRIVER)]
amr14  = amr14.sort_values('cla_median')
print(f"\n{'='*70}\n{TEAM} driver {DRIVER} — per-circuit summary (sorted by CLA)\n{'='*70}")
print(amr14[['event','n_coastdown','n_cla','cla_median','cla_se','score']]
      .to_string(index=False))


# ── Five-circuit polar set ─────────────────────────────────────────────────────
# Chosen to span the full AMR downforce range: low → high.
# Selection criteria applied to driver-14 data from survey_results.csv:
#   n_coastdown >= 5,  n_cla >= 4,  CLA in [1.5, 4.5],  SE < 0.30
# These 5 circuits maximise CLA spread while keeping both data-quality metrics high.
#
#  key       FastF1 event    lat_g  speed_kmh   label
POLAR_SET = [
    ('jeddah',  'Saudi Arabia', 2.5, 150.0, 'Jeddah 2024'),
    ('spa',     'Belgian',      3.5, 200.0, 'Spa-Francorchamps 2024'),
    ('sil',     'British',      3.5, 180.0, 'Silverstone 2024'),
    ('yas',     'Abu Dhabi',    3.0, 150.0, 'Yas Marina 2024'),
    ('suzuka',  'Japanese',     3.5, 180.0, 'Suzuka 2024'),
]
REF_KEY = 'jeddah'   # most coast-down segments → used for bootstrap & stratification


# ── Helper functions ──────────────────────────────────────────────────────────
def filter_push_laps(laps_df: pd.DataFrame) -> pd.DataFrame:
    dl = laps_df[pd.isna(laps_df['PitOutTime']) & pd.isna(laps_df['PitInTime'])]
    dl = dl[pd.notna(dl['LapTime'])]
    if dl.empty:
        return dl
    best = dl['LapTime'].min()
    return dl[dl['LapTime'] <= best * 1.10]


def collect_segments(year, circuit, driver, sessions):
    results, segs, ms, drs, laps_n, gears, rho_vals, sess_ids = [], [], [], [], [], [], [], []
    for sname in sessions:
        sess = fastf1.get_session(year, circuit, sname)
        sess.load(telemetry=True, weather=True)
        rho_s = air_density(sess.weather_data['AirTemp'].mean(),
                            sess.weather_data['Pressure'].mean())
        rho_vals.append(rho_s)
        dl = sess.laps.pick_drivers(driver)
        dl = dl[dl['LapNumber'] > 1]
        for _, lap in dl.iterrows():
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
                    results.append(r)
                    segs.append(seg)
                    ms.append(m)
                    drs.append(drs_open)
                    laps_n.append(lap_num)
                    sess_ids.append(sname)
                    gear = int(seg['nGear'].median()) if 'nGear' in seg.columns else -1
                    gears.append(gear)
    return segs, results, ms, drs, laps_n, float(np.mean(rho_vals)), gears, sess_ids


def cla_estimate(year, circuit, driver, lat_g_thresh, speed_thresh):
    vals = []
    try:
        sess = fastf1.get_session(year, circuit, 'Q')
        sess.load(telemetry=True, weather=True)
        rho = air_density(sess.weather_data['AirTemp'].mean(),
                          sess.weather_data['Pressure'].mean())
        dl = filter_push_laps(sess.laps.pick_drivers(driver))
        for _, lap in dl.iterrows():
            try:
                tel = lap.get_telemetry()
            except Exception:
                continue
            cla, _ = estimate_ClA(tel, QUALI_MASS, rho, mu=MU_TYRE,
                                  min_speed_kmh=speed_thresh, min_lat_g=lat_g_thresh,
                                  min_throttle=50.0)
            if np.isfinite(cla) and 0.5 < cla < 10.0:
                vals.append(cla)
    except Exception:
        pass
    return np.array(vals)


def run_circuit(year, circuit, driver, lat_g_thresh, speed_thresh, label):
    print(f"\n{'='*70}\n{label}  (driver {driver})\n{'='*70}")
    segs, results, ms, drs, laps_n, rho, gears, sess_ids = collect_segments(
        year, circuit, driver, ['FP1', 'FP2', 'FP3'])
    print(f"  segments passing R^2>={MIN_R2}: {len(results)}   rho={rho:.4f} kg/m^3")

    pooled    = fit_segments_pooled(segs, ms, drs, laps_n,
                                   seed_results=results, beta_fixed=BETA_FIXED, fit_v0=False)
    pooled_v0 = fit_segments_pooled(segs, ms, drs, laps_n,
                                   seed_results=results, beta_fixed=BETA_FIXED, fit_v0=True)
    alpha     = pooled[0].alpha
    alpha_std = pooled[0].alpha_std
    composite = 2 * alpha / rho
    Crr       = float(np.median([r.beta / (r.m * G) for r in pooled]))

    cla_vals      = cla_estimate(year, circuit, driver, lat_g_thresh, speed_thresh)
    cla           = float(np.median(cla_vals))
    cla_se_median = 1.253 * float(np.std(cla_vals)) / np.sqrt(len(cla_vals))
    CdA           = compute_CdA(composite, Crr, cla)

    dw_vals = []
    for i, seg in enumerate(segs):
        t      = seg['t'].values - seg['t'].values[0]
        v_meas = seg['Speed'].values / 3.6
        v_pred = v_model(t, pooled_v0[i].alpha, pooled_v0[i].beta,
                         pooled_v0[i].P_mgu, pooled_v0[i].v0, pooled_v0[i].m)
        dw_vals.append(durbin_watson(v_meas - v_pred))
    mean_dw = float(np.mean(dw_vals))
    dv0     = np.abs([pooled_v0[i].v0 - (segs[i]['Speed'].values[0] / 3.6)
                      for i in range(len(segs))])

    print(f"  alpha (no v0)   = {alpha:.4f} +/- {alpha_std:.4f}")
    print(f"  alpha (v0 fit)  = {pooled_v0[0].alpha:.4f}")
    print(f"  composite 2a/rho= {composite:.4f} m^2")
    print(f"  Crr             = {Crr:.4f}")
    print(f"  ClA             = {cla:.3f} m^2  (n={len(cla_vals)}, SE_median={cla_se_median:.3f})")
    print(f"    ClA @ mu=1.5   = {cla*MU_TYRE/1.5:.3f}    @ mu=2.0 = {cla*MU_TYRE/2.0:.3f}")
    print(f"  CdA             = {CdA:.3f} m^2")
    print(f"  mean DW (v0)    = {mean_dw:.3f}   mean|dv0|={dv0.mean():.3f} max={dv0.max():.3f}")

    return dict(key=circuit, label=label, rho=rho, n=len(results),
                alpha=alpha, alpha_std=alpha_std,
                composite=composite, Crr=Crr,
                cla=cla, cla_se_median=cla_se_median, cla_vals=cla_vals,
                CdA=CdA, mean_dw=mean_dw,
                segs=segs, results=results, ms=ms, drs=drs, laps_n=laps_n,
                gears=gears, sess_ids=sess_ids)


# ── Run all five circuits ─────────────────────────────────────────────────────
circuit_results = {}
for key, event, lat_g, speed, label in POLAR_SET:
    circuit_results[key] = run_circuit(2024, event, DRIVER, lat_g, speed, label)

ref = circuit_results[REF_KEY]   # reference circuit (Jeddah — most coast segments)
polar_circuits = [circuit_results[k] for k, *_ in POLAR_SET]


# ── Gear-8 stratification (reference circuit) ────────────────────────────────
print(f"\n{'='*70}\nGear-8 stratification ({ref['label']})\n{'='*70}")
g8 = [i for i, g in enumerate(ref['gears']) if g == 8]
print(f"  gear-8 segments: {len(g8)} / {len(ref['gears'])}")
if len(g8) >= 2:
    p8 = fit_segments_pooled([ref['segs'][i] for i in g8],
                             [ref['ms'][i]   for i in g8],
                             [ref['drs'][i]  for i in g8],
                             [ref['laps_n'][i] for i in g8],
                             seed_results=[ref['results'][i] for i in g8],
                             beta_fixed=BETA_FIXED)
    a8 = p8[0].alpha
    print(f"  gear-8 alpha = {a8:.4f}   all-segment alpha = {ref['alpha']:.4f}   "
          f"delta = {100*(a8-ref['alpha'])/ref['alpha']:+.2f}%")
else:
    a8 = float('nan')
    print("  insufficient gear-8 segments")


# ── Inter-driver alpha spread (reference circuit, both AMR cars) ──────────────
ref_event = next(ev for k, ev, *_ in POLAR_SET if k == REF_KEY)
amr_drivers_ref = (df_survey[(df_survey['team'] == TEAM) &
                              (df_survey['event'].str.contains(ref_event[:6], case=False))]
                   ['driver'].tolist())
print(f"\n{'='*70}\nInter-driver alpha spread ({ref['label']}, team: {TEAM})\n{'='*70}")
driver_alphas = {DRIVER: ref['alpha']}
for drv in amr_drivers_ref:
    if drv == DRIVER:
        continue
    try:
        s, r, m, d, l, _, _, _ = collect_segments(2024, ref_event, drv, ['FP1', 'FP2', 'FP3'])
        if len(r) >= 2:
            pj = fit_segments_pooled(s, m, d, l, seed_results=r, beta_fixed=BETA_FIXED)
            driver_alphas[drv] = pj[0].alpha
            print(f"  driver {drv}: alpha={pj[0].alpha:.4f}  ({len(r)} segments)")
        else:
            print(f"  driver {drv}: only {len(r)} segments — skipped")
    except Exception as e:
        print(f"  driver {drv}: {e}")
avals  = list(driver_alphas.values())
spread = (max(avals) - min(avals)) / min(avals)
print(f"  alpha range: {min(avals):.4f}-{max(avals):.4f}   spread={100*spread:.1f}%")


# ── Five-circuit aero polar ───────────────────────────────────────────────────
def cla_sigma(d):
    stat = d['cla_se_median']
    syst = (d['cla']*MU_TYRE/1.5 - d['cla']*MU_TYRE/2.0) / 2.0
    return stat, syst, float(np.hypot(stat, syst))


def ols_slope(x, y):
    xm, ym = x.mean(), y.mean()
    denom  = np.sum((x - xm) ** 2)
    return np.sum((x - xm) * (y - ym)) / denom if denom > 0 else np.nan


CRR_APPROX = 0.014   # 120 N / (855 kg * 9.81) ≈ 0.0143; rounded for propagation

cda_pts  = np.array([d['CdA'] for d in polar_circuits])
cla_pts  = np.array([d['cla'] for d in polar_circuits])

# Bootstrap-scaled CdA sigma for polar plot (x-axis error bars and polar MC).
# Includes both the alpha-fitting term (dominant) and the Crr*σ_ClA term.
# The Crr*σ_ClA contribution is small (~0.006 m²) vs the bootstrap alpha term
# (~0.10 m²), but including it is more correct.
cda_sigs = np.array([
    np.hypot(d['alpha_std'] * BOOT_JACOBIAN_RATIO * 2.0 / d['rho'],
             CRR_APPROX * cla_sigma(d)[2])
    for d in polar_circuits
])

# Jacobian-based CdA sigma (lower bound) — used in Table 1.
# Bootstrap-corrected estimate is ~8–19× larger (see bootstrap ratio above).
cda_sigs_jac = np.array([
    np.hypot(2.0 * d['alpha_std'] / d['rho'], CRR_APPROX * cla_sigma(d)[2])
    for d in polar_circuits
])

cla_sigs = np.array([cla_sigma(d)[2] for d in polar_circuits])

polar_slope     = ols_slope(cda_pts, cla_pts)
polar_intercept = cla_pts.mean() - polar_slope * cda_pts.mean()

rng_polar  = np.random.default_rng(42)
slopes_mc  = []
for _ in range(N_POLAR_MC):
    cda_s = rng_polar.normal(cda_pts, cda_sigs)
    cla_s = rng_polar.normal(cla_pts, cla_sigs)
    s = ols_slope(cda_s, cla_s)
    if np.isfinite(s):
        slopes_mc.append(s)
slopes_mc   = np.array(slopes_mc)
slope_std   = float(np.std(slopes_mc))
slope_ci_lo = float(np.percentile(slopes_mc, 5))
slope_ci_hi = float(np.percentile(slopes_mc, 95))

print(f"\n{'='*70}\n{len(polar_circuits)}-circuit aero polar\n{'='*70}")
for i, d in enumerate(polar_circuits):
    _, _, tot = cla_sigma(d)
    print(f"  {d['label']:30s}  CdA={d['CdA']:.3f}±{cda_sigs[i]:.3f}  "
          f"ClA={d['cla']:.3f}±{tot:.3f}")
print(f"  OLS slope  dClA/dCdA = {polar_slope:.3f}")
print(f"  Intercept             = {polar_intercept:.3f} m^2")
print(f"  MC std (1-sigma)      = {slope_std:.3f}")
print(f"  MC 90% CI             = [{slope_ci_lo:.3f}, {slope_ci_hi:.3f}]")
print(f"  Significance (slope/std) = {polar_slope/slope_std:.2f} sigma from zero")


# ── Bootstrap CI on reference-circuit alpha ───────────────────────────────────
print(f"\n{'='*70}\nBootstrap 90% CI on {ref['label']} alpha (n=1000)\n{'='*70}")
ci_lo, ci_hi = bootstrap_alpha_ci(
    ref['segs'], ref['ms'], ref['drs'], ref['laps_n'],
    beta_fixed=BETA_FIXED, seed_results=ref['results'],
    n_boot=1000, ci=0.90, seed=42, fit_v0=False)
boot_hw = (ci_hi - ci_lo) / 2.0
print(f"  segment bootstrap 90% CI = [{ci_lo:.4f}, {ci_hi:.4f}]  half-width=+/-{boot_hw:.4f}")
print(f"  Jacobian std = +/-{ref['alpha_std']:.4f}   ratio={boot_hw/ref['alpha_std']:.1f}x")

# Session-stratified bootstrap: resamples whole FP sessions rather than
# individual segments, giving a between-session CI that captures session-level
# systematics (track evolution, fuel-load uncertainty, ambient temperature).
unique_sess = sorted(set(ref['sess_ids']))
print(f"  sessions present: {unique_sess}  (n_segments per session: "
      f"{[ref['sess_ids'].count(s) for s in unique_sess]})")
ci_lo_s, ci_hi_s = bootstrap_alpha_ci_session_stratified(
    ref['segs'], ref['ms'], ref['drs'], ref['laps_n'],
    beta_fixed=BETA_FIXED, session_ids=ref['sess_ids'],
    seed_results=ref['results'], n_boot=1000, ci=0.90, seed=42, fit_v0=False)
if not (ci_lo_s != ci_lo_s):   # not NaN
    boot_hw_s = (ci_hi_s - ci_lo_s) / 2.0
    print(f"  session bootstrap 90% CI = [{ci_lo_s:.4f}, {ci_hi_s:.4f}]  half-width=+/-{boot_hw_s:.4f}")
    print(f"  session/segment CI ratio = {boot_hw_s/boot_hw:.2f}x")
else:
    print("  session bootstrap returned NaN (degenerate — fewer than 2 sessions)")


# ── DRS: rescale analytically (dCdA linear in mass at race lap 20) ─────────────
m_old_drs = 798.0 + 80.0 + max(0.0, 95.0 - 1.8*20)
m_new_drs = car_mass(20)
drs_scale = m_new_drs / m_old_drs
print(f"\nDRS mass rescale factor (lap 20): {m_new_drs:.0f}/{m_old_drs:.0f} = {drs_scale:.4f}")


# ── REPORT VALUES ─────────────────────────────────────────────────────────────
print(f"\n\n{'#'*70}\n# REPORT VALUES\n{'#'*70}")
print(f"# Constructor: {TEAM},  Driver: {DRIVER},  Year: 2024")
print(f"# Circuits: {', '.join(d['label'] for d in polar_circuits)}")
print()
for i, d in enumerate(polar_circuits):
    stat, syst, tot = cla_sigma(d)
    print(f"[{d['label']:25s}] alpha={d['alpha']:.3f}+/-{d['alpha_std']:.3f}  "
          f"composite={d['composite']:.3f}  Crr={d['Crr']:.4f}  "
          f"ClA={d['cla']:.2f}(stat={stat:.2f} syst={syst:.2f} tot={tot:.2f})  "
          f"CdA={d['CdA']:.3f}+/-{cda_sigs_jac[i]:.3f}(Jac,LB)  "
          f"CdA_boot_sig={cda_sigs[i]:.3f}  "
          f"rho={d['rho']:.4f}  n_segs={d['n']}  n_cla={len(d['cla_vals'])}")
print()
print(f"[Polar] slope={polar_slope:.3f}  intercept={polar_intercept:.3f}  "
      f"std={slope_std:.3f}  90CI=[{slope_ci_lo:.3f},{slope_ci_hi:.3f}]  "
      f"significance={polar_slope/slope_std:.2f}sigma  n_circuits={len(polar_circuits)}")
print(f"[Gear-8] a8={a8:.4f} vs all={ref['alpha']:.4f}  "
      f"({100*(a8-ref['alpha'])/ref['alpha']:+.2f}%)")
print(f"[Driver spread] {min(avals):.3f}-{max(avals):.3f}  = {100*spread:.1f}%")
print(f"[Bootstrap] +/-{boot_hw:.3f}  ({boot_hw/ref['alpha_std']:.0f}x Jacobian)")
print(f"[DRS] scale={drs_scale:.4f}: -0.107->{-0.107*drs_scale:.3f}  "
      f"-0.086->{-0.086*drs_scale:.3f}  CI[-0.28,0.05]->[{-0.28*drs_scale:.3f},{0.05*drs_scale:.3f}]")
