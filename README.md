# F1 Telemetry Aerodynamics

Characterise aerodynamic parameters — CdA, ClA, Crr — from public F1 telemetry using [FastF1](https://github.com/theOehrly/Fast-F1). All analysis is based on timing and GPS data from the F1 live timing API; no proprietary team data is used.

Primary dataset: **Monza 2024** (low-drag package). Secondary dataset: **Silverstone 2024** (medium-high downforce), same car (Aston Martin AMR24).

---

## Key Results

| Parameter | Monza 2024 | Silverstone 2024 | Ratio S/M |
|---|---|---|---|
| Pooled α (N·s²/m²) | 0.800 ± 0.002 | 1.076 ± 0.005 | **1.35×** |
| Composite 2α/ρ (m²) | 1.410 | 1.805 | 1.28× |
| ClA (m²) | 3.36 ± 0.65 | 3.88 ± 0.56 | 1.16× |
| CdA (m²)* | 1.37 ± 0.01 | 1.75 ± 0.01 | 1.28× |
| Crr | 0.0131 | 0.0131 | **1.00** |
| ΔClA / ΔCdA efficiency | — | — | **1.4** |

*CdA is inflated ~40% by engine braking torque absorbed into the drag coefficient — see [Limitations](#limitations).

The Crr equality across circuits is a useful internal consistency check: rolling resistance is a car property, not a setup variable, and it is identical to three decimal places.

---

## Methods

### 1. Coast-down ODE fit (`nb02`, `nb06`)

Free-deceleration segments (throttle < 5%, brake = 0) are extracted from FP telemetry and fitted to:

$$m\frac{dv}{dt} = -\alpha v^2 - \beta - \frac{P_\text{MGU}}{v}$$

where α = ½ρ(CdA + Crr·ClA) is the aerodynamic coefficient, β = Crr·mg is rolling resistance, and P_MGU is MGU-K harvest power modelled as a constant-power retarding term. β is fixed at a prior median (120 N) to break the α/β/P_MGU degeneracy. Segments across FP1/FP2/FP3 are pooled into a shared-α fit to tighten parameter uncertainty.

### 2. ClA from GPS lateral g (`nb03`, `nb06`)

At high-speed corners the tyre friction limit gives:

$$a_\text{lat} = \mu\left(g + \frac{\rho C_L A}{2m}v^2\right)$$

GPS position (10 Hz) is differentiated to compute lateral acceleration. Samples above a circuit-specific threshold (2.5g at Monza; 3.5g at Silverstone's Maggotts-Becketts) select near-limit apices. ClA is the median over all qualifying lap estimates with μ = 1.8 assumed. Sensitivity to μ is reported in `nb03`.

### 3. DRS drag delta (`nb05`)

Race trap speeds on the Monza main straight are compared between DRS-open and DRS-closed laps, converted to ΔCDA via an energy balance over the activation zone. A slipstream-stratification analysis (gap to car ahead from lap-end timing) confirms the two effects cannot be separated in race data: every DRS-eligible lap is also a slipstream lap.

### 4. Circuit comparison (`nb06`)

The same pipeline runs on Silverstone 2024 FP data (driver 18, Aston Martin) and is compared against Monza. The ΔClA/ΔCdA ratio of **1.4** quantifies the aerodynamic efficiency of the higher-wing Silverstone setup: 1.4 m² of downforce gained per m² of drag added.

---

## Limitations

**Engine braking inflates CdA.** The ODE has no explicit engine-braking term. At throttle = 0 the drivetrain applies additional retarding torque through compression and MGU-H harvest, which is absorbed into α. The Durbin-Watson statistic for ODE residuals is 0.66–0.71 across both circuits (target ≈ 2.0), confirming systematic positive autocorrelation. CdA absolute values are upper bounds; composite 2α/ρ values and cross-circuit *ratios* are reliable.

**DRS delta inseparable from slipstream in race data.** In 2024 DRS requires a gap ≤ 1 s — the same condition that guarantees wake exposure. The pooled ΔCDA = −0.107 m² (90% CI: −0.28 to +0.05) captures DRS + slipstream combined and cannot be decomposed without controlling for following distance.

**ClA depends on assumed μ.** At μ = 1.5 the Monza estimate becomes 3.93 m²; at μ = 2.0 it is 3.02 m². Sensitivity curves are in `nb03`.

**2026 active aero channel absent from F1 API.** Inspecting the raw timing stream for Canada 2026 (documented in `nb05` on the `2026-regs` branch) shows that channel 45 — DRS in 2024 — is absent from every 2026 car data message. Mode-split ΔCdA measurement requires this channel to be added to the API.

---

## Repository Structure

```
notebooks/
  01_data_exploration.ipynb      # channel verification, DRS encoding, GPS quality
  02_coastdown_fit.ipynb         # ODE fit, pooled α, Spa cross-validation
  03_ClA_estimation.ipynb        # GPS lat-g ClA, μ sensitivity
  04_full_characterisation.ipynb # CdA, Crr, MC uncertainty
  05_drs_delta.ipynb             # DRS trap speed analysis, slipstream stratification
  06_circuit_comparison.ipynb    # Monza vs Silverstone setup comparison (self-contained)

src/
  segments.py     # coast-down and corner sample extraction
  ode_fit.py      # ODE integration, per-segment and pooled fitting
  aero_params.py  # CdA/ClA/Crr derivation, car mass model
  uncertainty.py  # Monte Carlo uncertainty propagation

results/figures/  # saved plots
```

The `2026-regs` branch extends the analysis to Canada 2026 and documents the active aero API investigation.

---

## Running

```bash
pip install -r requirements.txt
jupyter lab
```

Run notebooks in order (01 → 06). Each notebook caches session data on first run via FastF1; subsequent runs load from `cache/`. `nb06` is self-contained and does not depend on pkl files from earlier notebooks.

---

## Data

All telemetry is fetched from the [F1 Live Timing API](https://livetiming.formula1.com) via FastF1. No proprietary or commercially licensed data is used.
