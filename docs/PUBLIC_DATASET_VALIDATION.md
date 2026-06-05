# Validation on a real, public industrial dataset

A synthetic-data project invites one fair objection: *"what does your baseline do on
data you didn't make?"* This is the answer — the WattsWorth ISO 50001 baseline engine,
unchanged, run on a real third-party steel-plant dataset.

## Dataset

**UCI / Mendeley — "Steel Industry Energy Consumption"** (DAEWOO Steel Co. Ltd,
Gwangyang, South Korea, 2018). 35,040 records at 15-minute resolution; energy use,
reactive power, power factor, load class (Light/Medium/Maximum), and calendar fields.
Public, peer-cited, and genuinely industrial — exactly the steel-process context
WattsWorth targets.

## Method

The engine's `fit_baseline` accepts arbitrary driver columns, so no engine code
changed. The raw 15-minute series was aggregated to **365 daily observations**, and
daily energy was regressed on two **independent, non-collinear** ISO 50001 relevant
variables:

- `load_index` — mean daily load class (production-intensity proxy)
- `weekend` — calendar effect

Reactive power was excluded (it is co-determined with active energy, not an
independent driver), and `active_hours` was excluded (collinear with `load_index` —
the VIF diagnostic flagged this with an infinite VIF when both were included, which is
itself evidence the diagnostic works).

Reproduce: `python scripts/validate_public_dataset.py`

## Results (real data)

| Metric | Value | Reading |
|---|---|---|
| Daily observations | 365 | one year |
| Drivers retained | `load_index`, `weekend` | both significant; sensible signs |
| Coefficient: load_index | **+2366** | more production → more energy ✓ |
| Coefficient: weekend | **−1271** | weekends use less ✓ |
| **R²** | **0.60** | modest — proxies, not a production meter |
| **CV(RMSE) in-sample** | **35.6%** | |
| **CV(RMSE) out-of-sample** | **27.8%** | honest hold-out error |
| Durbin–Watson | **1.01** | strong residual autocorrelation, correctly flagged |
| Max VIF | **1.3** | clean — no multicollinearity with these drivers |
| Breusch–Pagan p | **0.013** | heteroskedastic, correctly flagged |

## Interpretation — this is the valuable part

The headline is **not** a high R². It is that the tool behaves like a competent M&V
practitioner on real data:

1. **It generalises.** The same engine that fits the synthetic plant fits a real
   Korean steel plant with no code changes and economically sensible coefficients.
2. **Its diagnostics tell the truth.** Durbin–Watson 1.01 correctly flags the
   autocorrelation that pervades real industrial energy data; the VIF diagnostic
   correctly blows up (∞) when collinear drivers are forced in.
3. **It correctly refuses to over-certify.** CV(RMSE) of ~36% **exceeds the IPMVP /
   ASHRAE Guideline 14 acceptance threshold of 20%**, so this baseline is **not
   certifiable for M&V**. That is the *right* answer: with only load-class and
   calendar proxies — and **no independent production-volume or ambient-temperature
   meter** — you should not sign off on savings. A tool that returned a confident
   green number here would be wrong.

The contrast makes the point precisely:

| | Synthetic clean data | Real steel data (proxy drivers) |
|---|---|---|
| R² | 0.92 | 0.60 |
| CV(RMSE) | 2.6% | 35.6% |
| IPMVP gate (<20%) | ✅ passes | ❌ fails — correctly flagged |

**Conclusion.** WattsWorth's value is not that it always produces a passing baseline —
it is that it produces an *honest* one, and tells you, via standard ASHRAE/IPMVP
diagnostics, whether the baseline is good enough to certify savings against. On this
public dataset the verdict is "not yet — get a production and weather signal first,"
which is exactly what a real energy manager would conclude.
