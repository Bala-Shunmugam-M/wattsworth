# Methods & Limitations

This document states, plainly, what WattsWorth does, the standards it follows, and
where it is simplified or limited. Naming limitations up front is deliberate: a
defensible M&V tool is one whose author knows exactly where the bodies are buried.

---

## Methods implemented

| Area | Method | Where |
|------|--------|-------|
| Energy baseline | OLS regression of energy on production, ambient temperature, operating hours, with backward elimination of insignificant drivers (p > 0.05) | `engine/baseline.py` |
| Fit acceptance | ASHRAE Guideline 14 CV(RMSE), reported **in-sample and out-of-sample** (chronological 20% hold-out) | `engine/baseline.py` |
| Regression validity | Durbin–Watson (autocorrelation), VIF (multicollinearity), Breusch–Pagan (heteroskedasticity) | `engine/baseline.py` |
| Savings | IPMVP Option C — baseline-expected minus actual under reporting-period conditions | `engine/mv.py` |
| Significance | One-sided t-test on daily avoided energy, **corrected for serial correlation** via an effective sample size `n_eff = n·(1−ρ)/(1+ρ)` | `engine/mv.py` |
| Drift | Cumulative sum (CUSUM) of (actual − expected) | `engine/mv.py` |
| Anomaly | Baseline-residual robust z-score (MAD), rolling-median detrended so a regime change is not flagged; plus a multivariate IsolationForest on specific energy consumption | `engine/anomaly.py` |
| Forecast | Holt–Winters exponential smoothing with a 95% band; naive-mean fallback | `engine/forecast.py` |
| Opportunities | Motor right-sizing (efficiency-vs-load), time-of-use load shifting, power-factor correction — ranked by ₹ payback | `engine/optimize.py` |

---

## Limitations (and how each is mitigated or would be addressed)

### 1. Validation data is synthetic
The data is generated, not metered. **Mitigation:** two modes ship — a *clean* mode
(exactly linear, for teaching) and a *realistic* hard mode (`generate_realistic_plant_data`)
with non-linearity, autocorrelated/heteroskedastic noise, collinear drivers, weekend
effects, and missing days. On realistic data the baseline R² falls to ~0.57 and
Durbin–Watson to ~1.1 — the engine is shown degrading honestly rather than recovering a
model identical to the one that generated the data. **Done:** the engine has now also been
run on a real public dataset (UCI Steel Industry Energy Consumption, DAEWOO Steel, Korea) —
R²=0.60, CV(RMSE) 36%, Durbin–Watson 1.01; it generalises with sensible coefficients, the
diagnostics flag the real autocorrelation, and it correctly reports the baseline as *not*
IPMVP-certifiable on proxy drivers. See [PUBLIC_DATASET_VALIDATION.md](PUBLIC_DATASET_VALIDATION.md).

### 2. In-sample fit is optimistic
R² and in-sample CV(RMSE) overstate predictive accuracy. **Mitigation:** an out-of-sample
CV(RMSE) (chronological hold-out) is computed and displayed alongside the in-sample number,
so the gap is visible. **Next:** k-fold / rolling-origin cross-validation.

### 3. Daily energy is autocorrelated
A naive i.i.d. significance test overstates confidence. **Mitigation:** the t-test deflates
the sample size by the lag-1 autocorrelation (effective N) and reports ρ and `n_eff`.
**Next:** Newey–West (HAC) standard errors, and the full IPMVP fractional-savings-uncertainty
band rather than a significance flag alone.

### 4. "Optimization" is heuristic, not constrained optimization
The optimization page ranks three independent engineering heuristics by ₹; there is no
objective function or constraint solver. It is an *opportunity register*. **Next:** a
capital-budget knapsack (maximize ₹ or tCO₂ subject to a capex limit) would make the term
literal.

### 5. Single site, single meter, daily resolution, batch
Real M&V runs on 15-minute interval data across a meter hierarchy and many sites; this models
one plant's daily site-level kWh — the easy case. The hard parts (interval data quality,
sub-metering, cross-site normalization) are out of scope. The `engine/` is UI- and
storage-free, so adding a time-series store and per-site dimension is plumbing, not an
analytics rewrite.

### 6. Economics are India-defaulted constants
Grid emission factor (0.71 kg CO₂/kWh), tariffs, demand charge, and capex rates are
configurable defaults in `engine/config.py`, not sourced per utility/region. The grid factor
in particular is regional and time-varying. **Next:** per-site economic settings.

### 7. Motor model is a simplified curve
`motor_right_sizing` uses an efficiency-vs-load parabola and assumes a constant-shaft load.
Real motor swaps must respect NEMA frame, starting torque, and process turndown, and the
dominant benefit of right-sizing is often power factor, not efficiency. The asset register
carries a measured `efficiency_pct` that a future version should prefer over the curve.

---

## How to talk about this

The honest framing is the strong one: *"I built a labelled ground-truth harness — a known
6% efficiency project and 5 injected anomalies — then blinded the engine to the labels to
measure detection accuracy, and added a realistic hard-mode dataset to confirm the methods
degrade gracefully and the diagnostics catch it."* That is how you validate a detector
before trusting it on a real plant.
