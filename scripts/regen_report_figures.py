"""
Regenerate all report figures using the updated 5-circuit analysis.

Figures produced:
  02_fit_quality.pgf        ODE coast-down fit for sample segments (Jeddah FP, driver 14)
  03_ClA_vs_lap.pgf         Per-lap CLA from qualifying push laps (Spa, driver 14)
  06_circuit_comparison.pgf Two-panel aero polar (CLA vs CdA), with Yas Marina sensitivity
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import fastf1
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.preamble': '\n'.join([
        r'\usepackage{mathpazo}',
        r'\usepackage{amsmath}',
        r'\usepackage[T1]{fontenc}',
    ]),
})

from src.segments import extract_coastdown_segments, segment_drs_state
from src.ode_fit import fit_segment, fit_segments_pooled, v_model
from src.aero_params import air_density, car_mass, estimate_ClA, compute_CdA

ROOT = os.path.join(os.path.dirname(__file__), '..')
RPT  = os.path.join(ROOT, 'reports', 'FastF1_Data_Correlation')
fastf1.Cache.enable_cache(os.path.join(ROOT, 'cache'))

BETA   = 120.0
MINR2  = 0.90
MU     = 1.8
QUALI_MASS = 803.0

BOOT_JACOBIAN_RATIO = 13.0   # conservative Monza-calibrated scaling for CdA uncertainty

# Pre-computed 5-circuit results from recompute_report_numbers.py
# (CdA, CLA, alpha_std, cla_se_median, rho, n_segs, n_cla)
# alpha_std: corrected Jacobian 1-sigma (factor-of-2 covariance fix applied)
POLAR_DATA = {
    'Jeddah\n2024':      dict(CdA=1.448, ClA=2.662, alpha_std=0.004, cla_se=0.134, rho=1.1795),
    'Spa\n2024':         dict(CdA=1.376, ClA=3.064, alpha_std=0.007, cla_se=0.129, rho=1.1446),
    'Silverstone\n2024': dict(CdA=1.487, ClA=3.137, alpha_std=0.004, cla_se=0.192, rho=1.1925),
    'Yas Marina\n2024':  dict(CdA=1.335, ClA=3.531, alpha_std=0.008, cla_se=0.155, rho=1.1784),
    'Suzuka\n2024':      dict(CdA=1.689, ClA=4.330, alpha_std=0.008, cla_se=0.160, rho=1.2234),
}


def filter_push_laps(laps_df):
    dl = laps_df[pd.isna(laps_df['PitOutTime']) & pd.isna(laps_df['PitInTime'])]
    dl = dl[pd.notna(dl['LapTime'])]
    if dl.empty:
        return dl
    best = dl['LapTime'].min()
    return dl[dl['LapTime'] <= best * 1.10]


def cla_sigma(cla, cla_se):
    stat = cla_se
    syst = (cla * MU / 1.5 - cla * MU / 2.0) / 2.0
    return float(np.hypot(stat, syst))


def ols_slope(x, y):
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    return np.sum((x - xm) * (y - ym)) / denom if denom > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Figure 02: ODE coast-down fit quality  (Jeddah FP, driver 14)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 02: coast-down fit quality (Jeddah)...")
good_segs, good_results, good_ms, good_drs, good_laps = [], [], [], [], []
rho_vals = []
for sname in ['FP1', 'FP2', 'FP3']:
    sess = fastf1.get_session(2024, 'Saudi Arabia', sname)
    sess.load(telemetry=True, weather=True)
    rho_s = air_density(sess.weather_data['AirTemp'].mean(),
                        sess.weather_data['Pressure'].mean())
    rho_vals.append(rho_s)
    dl = sess.laps.pick_drivers('14')
    dl = dl[dl['LapNumber'] > 1]
    for _, lap in dl.iterrows():
        ln = int(lap['LapNumber']); m = car_mass(ln)
        try: tel = lap.get_telemetry()
        except Exception: continue
        for seg in extract_coastdown_segments(tel, min_duration=0.5,
                                              min_speed_kmh=120.0, throttle_threshold=5.0):
            drs_open = segment_drs_state(seg)
            r = fit_segment(seg, m, rho_s, drs_open, ln, beta_fixed=BETA)
            if r is not None and r.r2 >= MINR2:
                good_segs.append(seg); good_results.append(r)
                good_ms.append(m); good_drs.append(drs_open); good_laps.append(ln)

rho = float(np.mean(rho_vals))
pooled = fit_segments_pooled(good_segs, good_ms, good_drs, good_laps,
                             seed_results=good_results, beta_fixed=BETA, fit_v0=True)

# Pick 4 representative segments spread across the available set
n_segs = len(good_segs)
idxs = np.linspace(0, n_segs - 1, 4, dtype=int)
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2), sharey=False)
axes = axes.flatten()
for ax, i in zip(axes, idxs):
    seg = good_segs[i]; r = pooled[i]
    t = seg['t'].values - seg['t'].values[0]
    v_meas = seg['Speed'].values / 3.6
    v_pred = v_model(t, r.alpha, r.beta, r.P_mgu, r.v0, r.m)
    ax.plot(t, v_meas * 3.6, 'o', ms=3, color='steelblue', label='measured')
    ax.plot(t, v_pred * 3.6, '-', lw=1.8, color='tomato', label='ODE fit')
    ax.set_title(f'$R^2={r.r2:.4f}$', fontsize=9)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (km/h)')
axes[0].legend(fontsize=8)
plt.tight_layout()
out02 = os.path.join(RPT, 'Methodology', 'Figures', '02_fit_quality.pgf')
plt.savefig(out02); plt.close()
print(f"  wrote {out02}  ({n_segs} segments total, showing 4)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 03: Per-lap CLA from qualifying push laps  (Spa, driver 14)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 03: per-lap CLA (Spa qualifying push laps)...")
sess = fastf1.get_session(2024, 'Belgian', 'Q')
sess.load(telemetry=True, weather=True)
rho_q = air_density(sess.weather_data['AirTemp'].mean(),
                    sess.weather_data['Pressure'].mean())
push = filter_push_laps(sess.laps.pick_drivers('14'))

cla_rows = []
for _, lap in push.iterrows():
    try: tel = lap.get_telemetry()
    except Exception: continue
    med, std = estimate_ClA(tel, QUALI_MASS, rho_q, mu=MU,
                            min_speed_kmh=200.0, min_lat_g=3.5, min_throttle=50.0)
    if np.isfinite(med) and 0.5 < med < 10.0:
        cla_rows.append((int(lap['LapNumber']), med, std))

laps_q = [r[0] for r in cla_rows]
vals_q  = [r[1] for r in cla_rows]
errs_q  = [r[2] for r in cla_rows]
med_q   = float(np.median(vals_q))
q1, q3  = np.percentile(vals_q, [25, 75])

fig, ax = plt.subplots(figsize=(6.3, 2.8))
ax.errorbar(range(len(laps_q)), vals_q, yerr=errs_q, fmt='o', capsize=4,
            color='steelblue', label='per-lap estimate')
ax.axhline(med_q, color='r', linestyle='--', lw=1.5,
           label=f'median $= {med_q:.2f}$ m$^2$')
ax.fill_between([-0.5, len(laps_q) - 0.5], q1, q3,
                alpha=0.15, color='r', label='IQR')
ax.set_xticks(range(len(laps_q)))
ax.set_xticklabels([f'Lap {l}' for l in laps_q], rotation=30, ha='right', fontsize=8)
ax.set_ylim(0, 6); ax.set_ylabel(r'$C_{LA}$ (m$^2$)')
ax.legend(); plt.tight_layout()
out03 = os.path.join(RPT, 'Methodology', 'Figures', '03_ClA_vs_lap.pgf')
plt.savefig(out03); plt.close()
print(f"  wrote {out03}  (median={med_q:.3f}, IQR=[{q1:.2f},{q3:.2f}], n={len(laps_q)})")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 06: two-panel aero polar (all 5 circuits + robustness without Yas Marina)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 06: two-panel aero polar...")

ALL_LABELS   = list(POLAR_DATA.keys())
ANOMALOUS    = 'Yas Marina\n2024'
ROB_LABELS   = [l for l in ALL_LABELS if l != ANOMALOUS]
ALL_COLORS   = ['#2166ac', '#74add1', '#fdae61', '#f46d43', '#d73027']
COLOR_MAP    = dict(zip(ALL_LABELS, ALL_COLORS))


def polar_arrays(labels):
    cda = np.array([POLAR_DATA[l]['CdA'] for l in labels])
    cla = np.array([POLAR_DATA[l]['ClA'] for l in labels])
    cda_e = np.array([POLAR_DATA[l]['alpha_std'] * BOOT_JACOBIAN_RATIO * 2.0 / POLAR_DATA[l]['rho']
                      for l in labels])
    cla_e = np.array([cla_sigma(POLAR_DATA[l]['ClA'], POLAR_DATA[l]['cla_se']) for l in labels])
    return cda, cla, cda_e, cla_e


def draw_polar(ax, labels, title, anomalous_label=None):
    cda_pts, cla_pts, cda_err, cla_err = polar_arrays(labels)
    colors = [COLOR_MAP[l] for l in labels]

    slope     = ols_slope(cda_pts, cla_pts)
    intercept = cla_pts.mean() - slope * cda_pts.mean()

    rng = np.random.default_rng(42)
    slopes_mc = []
    for _ in range(50_000):
        xs = rng.normal(cda_pts, cda_err)
        ys = rng.normal(cla_pts, cla_err)
        s = ols_slope(xs, ys)
        if np.isfinite(s):
            slopes_mc.append(s)
    slopes_mc = np.array(slopes_mc)
    slope_lo  = float(np.percentile(slopes_mc, 5))
    slope_hi  = float(np.percentile(slopes_mc, 95))
    slope_sd  = float(np.std(slopes_mc))
    n_sigma   = abs(slope) / slope_sd if slope_sd > 0 else np.nan

    cda_range = np.linspace(1.27, 1.83, 200)
    cla_fit   = slope * cda_range + intercept
    mc_ints   = cla_pts.mean() - slopes_mc * cda_pts.mean()
    cla_mc    = slopes_mc[:, None] * cda_range[None, :] + mc_ints[:, None]
    cla_lo    = np.percentile(cla_mc, 5, axis=0)
    cla_hi    = np.percentile(cla_mc, 95, axis=0)

    ax.fill_between(cda_range, cla_lo, cla_hi, alpha=0.15, color='grey',
                    label=r'MC 90\% CI on slope')
    ax.plot(cda_range, cla_lo, '--', color='grey', lw=0.8, alpha=0.6)
    ax.plot(cda_range, cla_hi, '--', color='grey', lw=0.8, alpha=0.6)
    ax.plot(cda_range, cla_fit, '-', color='black', lw=1.5,
            label='OLS fit')

    for i, label in enumerate(labels):
        lname = label.replace('\n', ' ')
        if label == anomalous_label:
            ax.errorbar(cda_pts[i], cla_pts[i],
                        xerr=cda_err[i], yerr=cla_err[i],
                        fmt='D', color=colors[i], ms=8, capsize=5, lw=1.5,
                        mfc='none', mew=2.0,
                        label=lname + r' $\dagger$')
        else:
            ax.errorbar(cda_pts[i], cla_pts[i],
                        xerr=cda_err[i], yerr=cla_err[i],
                        fmt='o', color=colors[i], ms=8, capsize=5, lw=1.5,
                        label=lname)

    ax.text(0.97, 0.05,
            f'slope $= {slope:.2f}$\n'
            f'90\\% CI $= [{slope_lo:.1f},\\,{slope_hi:.1f}]$\n'
            f'${n_sigma:.2f}\\,\\sigma$ from zero',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$C_{DA}$ (m$^2$)')
    ax.set_ylabel(r'$C_{LA}$ (m$^2$)')
    ax.set_xlim(1.27, 1.83)
    ax.set_ylim(1.9, 5.3)

    return slope, slope_lo, slope_hi, n_sigma


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 3.8))

slope5, lo5, hi5, sig5 = draw_polar(
    ax1, ALL_LABELS,
    title='(a)',
    anomalous_label=ANOMALOUS)

slope4, lo4, hi4, sig4 = draw_polar(
    ax2, ROB_LABELS,
    title='(b)')

# Single shared legend beneath both panels
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7,
           bbox_to_anchor=(0.5, 0.0), frameon=True,
           borderpad=0.5, handlelength=1.5, handletextpad=0.4, columnspacing=1.0)
plt.tight_layout(rect=[0, 0.14, 1, 1])

out06 = os.path.join(RPT, 'Results', 'Figures', '06_circuit_comparison.pgf')
plt.savefig(out06, bbox_inches='tight'); plt.close()
print(f"  wrote {out06}")
print(f"  5-circuit: slope={slope5:.3f}  90% CI=[{lo5:.2f},{hi5:.2f}]  {sig5:.2f}σ")
print(f"  4-circuit: slope={slope4:.3f}  90% CI=[{lo4:.2f},{hi4:.2f}]  {sig4:.2f}σ")

print("\nDone — all three figures written.")
