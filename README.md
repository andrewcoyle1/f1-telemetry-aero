# F1 Telemetry Aero Characterisation

Extract aerodynamic coefficients from real F1 telemetry using coast-down ODE fitting and high-speed corner analysis.

Dataset: Monza 2024 (FP1/FP2/FP3), with cross-validation against Belgium (Spa) 2024.

## Results

| Quantity | Result | Expected | Status |
|---|---|---|---|
| ClA (downforce area) | 3.36 ± 0.65 m² | 2.5–3.5 m² | ✓ |
| Crr (rolling resistance) | 0.0131 | 0.015–0.020 | Fixed from data |
| CdA (drag area, coast-down) | 1.37 ± 0.01 m² | 0.8–1.0 m² | See note |
| ΔCDA (DRS delta) | n/a | −0.5 to −0.8 m² | No DRS-open data |

**Note on CdA:** The composite drag (2α/ρ = 1.41 m²) is ~40% above the expected race-speed value. This is a known limitation of the coast-down method applied to ground-effect cars: at the 40–80 m/s deceleration range the car rides higher than at race speed, increasing the effective CdA. The Spa/Monza composite ratio (1.29×) is physically consistent with the higher-downforce Spa setup, confirming the relative values are correct. An absolute correction would require suspension travel data not available in FastF1.

## Setup

```bash
pip install -r requirements.txt
```

## Notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `01_data_exploration` | Verify telemetry channels, DRS encoding, GPS availability | ✓ |
| `02_coastdown_fit` | Driver survey across FP sessions/circuits, ODE fitting, pooled fit, Spa validation | ✓ |
| `03_ClA_estimation` | GPS lateral-g computation, downforce area from high-speed corners | ✓ |
| `04_full_characterisation` | Combine CdA, Crr, ClA → final parameters with Monte Carlo uncertainty | ✓ |

## Physical model

Coast-down ODE with MGU-K energy recovery term:

```
m·dv/dt = -α·v² - β - P_mgu/v
```

| Parameter | Physical meaning | How estimated |
|---|---|---|
| α | ½ρ·(CdA + Crr·ClA) | Pooled fit across all segments (shared) |
| β | Crr·m·g | Fixed at 120 N from per-segment median |
| P_mgu | MGU-K harvest power | Per-segment free parameter (0–120 kW) |

ClA from high-speed corner lateral-g via tyre friction model:

```
m·v²/R = μ·(m·g + ½ρ·ClA·v²)
```

## Pipeline

```
FastF1 (Monza + Spa FP sessions)
        │
        ▼
  segments.py        ← extract coast-down & corner slices
        │
        ▼
   ode_fit.py        ← 3-param ODE fit per segment → pooled fit (shared α)
        │
  aero_params.py     ← GPS lateral-g → ClA from corner samples
        │
        └──────────────────────┐
                               ▼
                   aggregate + uncertainty.py
                               │
                               ▼
                   CdA, ClA, Crr ± MC bounds
```

## Key implementation notes

- **GPS lat-g**: FastF1 merges car data (~240 Hz) with GPS (~4 Hz). Computing lateral acceleration requires filtering to `Source=='pos'` rows, removing close-together GPS fixes (dt < 0.1 s), and converting from decimeter to meter coordinates before differentiating.
- **MGU-K term**: Two-parameter fits (α, β only) absorb energy recovery braking into β, inflating both parameters. Adding P_mgu/v gives β = 120 N (Crr = 0.013) and P_mgu = 0–53 kW, physically consistent with MGU-K harvest rates.
- **Pooled fitting**: α and β are physical constants of the car setup. Fitting them as shared parameters across all segments (with only P_mgu free per segment) reduces parameter count from 3N to N+2 and tightens the α estimate.
- **Speed-drop filter**: Segments require a minimum 25 m/s speed drop to ensure the three ODE terms have distinguishable shapes across the data.
- **DRS delta**: DRS closes at lift-off, so coast-down segments are always DRS-closed. The delta requires a different approach (e.g. comparing acceleration traces on DRS-eligible straights).

## Known limitations

1. CdA is ~40% above race-speed value due to ride-height dependence of ground-effect aerodynamics
2. DRS drag delta cannot be extracted from free-practice coast-down data
3. Crr is fixed rather than fitted due to degeneracy in the 3-parameter model with available segment lengths
