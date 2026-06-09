"""
Survey all 2024 F1 circuits for coast-down and qualifying CLA data quality.

For each circuit and every driver:
  - n_coastdown : FP1+FP2+FP3 coast-down segments passing R²≥0.90
  - n_cla       : qualifying push laps with a finite CLA estimate above threshold
  - cla_median  : median CLA from those laps (m²)
  - cla_se      : standard error of the median

Saves a row to survey_results.csv after every circuit so the run is
resumable; already-processed circuits are skipped on re-run.

Final output: per-circuit top-5 table + per-constructor cross-circuit scores.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import fastf1
import traceback

from src.segments import extract_coastdown_segments, segment_drs_state
from src.ode_fit import fit_segment, FitResult
from src.aero_params import air_density, car_mass, estimate_ClA

fastf1.Cache.enable_cache(os.path.join(os.path.dirname(__file__), '..', 'cache'))

# ── Constants ─────────────────────────────────────────────────────────────────
BETA_FIXED  = 120.0
MIN_R2      = 0.90
MU_TYRE     = 1.8
QUALI_MASS  = 803.0   # 798 + 5 kg fuel for qualifying
FP_SESSIONS = ['FP1', 'FP2', 'FP3']
RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'survey_results.csv')

QUALIFYING_SESSIONS = {'Q', 'Q1', 'Q2', 'Q3'}


def filter_push_laps(laps_df):
    dl = laps_df[pd.isna(laps_df['PitOutTime']) & pd.isna(laps_df['PitInTime'])]
    dl = dl[pd.notna(dl['LapTime'])]
    if dl.empty:
        return dl
    best = dl['LapTime'].min()
    return dl[dl['LapTime'] <= best * 1.10]


# ── 2024 calendar ─────────────────────────────────────────────────────────────
# (event_name, lat_g_thresh, speed_thresh_kmh)
# lat_g / speed thresholds chosen per circuit for qualifying CLA estimation.
CALENDAR_2024 = [
    ('Bahrain',        3.5, 170.0),
    ('Saudi Arabia',   2.5, 150.0),   # Jeddah: fast but chicane-heavy
    ('Australian',     3.0, 150.0),
    ('Japanese',       3.5, 180.0),   # 130R / Spoon
    ('Chinese',        3.0, 150.0),
    ('Miami',          3.0, 150.0),
    ('Emilia Romagna', 3.0, 150.0),
    ('Monaco',         2.0, 100.0),   # very slow circuit
    ('Canadian',       3.0, 150.0),
    ('Spanish',        3.5, 180.0),   # T3 / T9
    ('Austrian',       3.5, 170.0),   # T3 / T6 / T9
    ('British',        3.5, 180.0),   # Maggots/Becketts
    ('Hungarian',      3.5, 170.0),
    ('Belgian',        3.5, 200.0),   # Pouhon
    ('Dutch',          3.5, 180.0),   # banked T3
    ('Italian',        2.5, 150.0),   # Monza: low lateral g
    ('Azerbaijan',     2.5, 150.0),
    ('Singapore',      2.5, 130.0),
    ('United States',  3.5, 170.0),   # T1 / T9-T12
    ('Mexico City',    3.0, 150.0),
    ('São Paulo',      3.0, 150.0),
    ('Las Vegas',      2.5, 150.0),
    ('Qatar',          3.5, 180.0),   # Lusail
    ('Abu Dhabi',      3.0, 150.0),
]


def count_coastdown(year, event, driver):
    """Return number of FP coast-down segments passing R²≥MIN_R2."""
    n = 0
    for sname in FP_SESSIONS:
        try:
            sess = fastf1.get_session(year, event, sname)
            sess.load(telemetry=True, weather=True)
            rho = air_density(sess.weather_data['AirTemp'].mean(),
                              sess.weather_data['Pressure'].mean())
            dl = sess.laps.pick_drivers(driver)
            dl = dl[dl['LapNumber'] > 1]
            for _, lap in dl.iterrows():
                m = car_mass(int(lap['LapNumber']))
                try:
                    tel = lap.get_telemetry()
                except Exception:
                    continue
                for seg in extract_coastdown_segments(
                        tel, min_duration=0.5, min_speed_kmh=120.0,
                        throttle_threshold=5.0):
                    r = fit_segment(seg, m, rho, False, int(lap['LapNumber']),
                                    beta_fixed=BETA_FIXED)
                    if r is not None and r.r2 >= MIN_R2:
                        n += 1
        except Exception:
            pass
    return n


def count_cla(year, event, driver, lat_g, speed):
    """Return (n, median_cla, se_median) from qualifying push laps."""
    vals = []
    try:
        sess = fastf1.get_session(year, event, 'Q')
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
                                  min_speed_kmh=speed, min_lat_g=lat_g,
                                  min_throttle=50.0)
            if np.isfinite(cla) and 0.5 < cla < 10.0:
                vals.append(cla)
    except Exception:
        pass

    if not vals:
        return 0, np.nan, np.nan
    n = len(vals)
    med = float(np.median(vals))
    se  = float(1.253 * np.std(vals) / np.sqrt(n)) if n > 1 else np.nan
    return n, med, se


def survey_event(year, event, lat_g, speed):
    """Survey all drivers at one event. Returns list of row dicts."""
    print(f"  Surveying {event} ...", flush=True)

    # Get driver + team list from the first available FP session
    drivers, teams = [], {}
    for sname in FP_SESSIONS:
        try:
            sess0 = fastf1.get_session(year, event, sname)
            sess0.load(telemetry=False, weather=False)
            drivers = list(sess0.drivers)
            for drv in drivers:
                try:
                    teams[drv] = sess0.get_driver(drv)['TeamName']
                except Exception:
                    teams[drv] = 'Unknown'
            break
        except Exception:
            continue

    if not drivers:
        print(f"    WARNING: could not load any FP session for {event} — skipping")
        return []

    rows = []
    for drv in drivers:
        n_coast = count_coastdown(year, event, drv)
        n_cla, cla_med, cla_se = count_cla(year, event, drv, lat_g, speed)
        rows.append(dict(
            year=year, event=event, driver=drv,
            team=teams.get(drv, 'Unknown'),
            n_coastdown=n_coast, n_cla=n_cla,
            cla_median=round(cla_med, 3) if np.isfinite(cla_med) else np.nan,
            cla_se=round(cla_se, 3) if np.isfinite(cla_se) else np.nan,
        ))
        print(f"    {drv:>3s} {teams.get(drv,'?'):30s}  "
              f"coast={n_coast:2d}  cla_n={n_cla:2d}  "
              f"cla={cla_med:.3f}" if np.isfinite(cla_med) else
              f"    {drv:>3s} {teams.get(drv,'?'):30s}  "
              f"coast={n_coast:2d}  cla_n={n_cla:2d}  cla=  nan",
              flush=True)

    return rows


# ── Main survey loop ──────────────────────────────────────────────────────────
# Load existing results so we can resume interrupted runs
if os.path.exists(RESULTS_CSV):
    existing = pd.read_csv(RESULTS_CSV)
    done_events = set(existing['event'].unique())
    all_rows = existing.to_dict('records')
    print(f"Resuming — {len(done_events)} events already done: {done_events}")
else:
    done_events = set()
    all_rows = []

for event, lat_g, speed in CALENDAR_2024:
    if event in done_events:
        print(f"  Skipping {event} (already in results)")
        continue
    print(f"\n{'='*70}\n{event} 2024\n{'='*70}")
    try:
        rows = survey_event(2024, event, lat_g, speed)
        all_rows.extend(rows)
        # Save after every event so we can resume
        pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
        print(f"  Saved {len(rows)} rows → {RESULTS_CSV}")
    except Exception:
        print(f"  ERROR surveying {event}:")
        traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)

print(f"\n\n{'#'*70}")
print("# CALENDAR SURVEY SUMMARY")
print(f"{'#'*70}")

# Per-event: top 3 drivers by combined score
print("\nTop drivers per event (sorted by n_coastdown + n_cla):")
for event in [e for e, *_ in CALENDAR_2024]:
    ev = df[df['event'] == event].copy()
    if ev.empty:
        continue
    ev['score'] = ev['n_coastdown'] + ev['n_cla']
    top = ev.nlargest(3, 'score')[['driver','team','n_coastdown','n_cla','cla_median','cla_se','score']]
    print(f"\n  {event}:")
    print(top.to_string(index=False, justify='left'))

# Per-constructor: total score across all events
print(f"\n\n{'='*70}")
print("Constructor scores (sum n_coastdown + n_cla across ALL events)")
print(f"{'='*70}")
team_scores = (df.assign(score=df['n_coastdown'] + df['n_cla'])
                 .groupby('team')['score'].sum()
                 .sort_values(ascending=False))
print(team_scores.to_string())

# Best circuits: events where top driver has n_coastdown>=3 AND n_cla>=4
print(f"\n\n{'='*70}")
print("Recommended circuits (any driver: n_coastdown>=3 AND n_cla>=4)")
print(f"{'='*70}")
viable = (df[(df['n_coastdown'] >= 3) & (df['n_cla'] >= 4)]
            .sort_values(['event','n_coastdown'], ascending=[True,False])
            .drop_duplicates('event'))
print(viable[['event','driver','team','n_coastdown','n_cla','cla_median','cla_se']].to_string(index=False))
