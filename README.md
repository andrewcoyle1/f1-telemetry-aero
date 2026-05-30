# F1 Telemetry Aero Characterisation

Extract **CdA** (drag area), **ClA** (downforce area), and the **DRS drag delta** from real F1 telemetry using coast-down ODE fitting and high-speed corner analysis.

Dataset: Monza 2024 Race (FastF1).

## Setup

```bash
pip install -r requirements.txt
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_exploration` | Verify telemetry channels, DRS encoding, lateral-g availability |
| `02_coastdown_fit` | Extract coast-down segments, fit α/β per segment, scatter by DRS state |
| `03_ClA_estimation` | Estimate downforce area from high-speed corner lateral g |
| `04_full_characterisation` | Combine everything → CdA, Crr, ΔCDA(DRS) with Monte Carlo uncertainty |

## Physical model

```
m·dv/dt = -α·v² - β
```

Analytical solution used for fitting (avoids differentiating noisy telemetry):

```
v(t) = sqrt(β/α) · tan(φ₀ − (k/m)·t)
k = sqrt(α·β),  φ₀ = arctan(v₀·sqrt(α/β))
```

From fitted coefficients:
- `Crr = β / (m·g)`
- `CdA + Crr·ClA = 2α/ρ`
- `CdA = (2α/ρ) − Crr·ClA`

## Expected results (Monza low-downforce config)

| Quantity | Expected |
|---|---|
| Crr | 0.015 – 0.020 |
| CdA (DRS closed) | 0.8 – 1.0 m² |
| ClA | 2.5 – 3.5 m² |
| ΔCDA (DRS open) | −0.5 to −0.8 m² |
