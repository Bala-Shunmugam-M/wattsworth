# ⚡ WattsWorth — Industrial Energy Performance & Decarbonization Engine

[![CI](https://github.com/Bala-Shunmugam-M/wattsworth/actions/workflows/ci.yml/badge.svg)](https://github.com/Bala-Shunmugam-M/wattsworth/actions/workflows/ci.yml)
&nbsp;**🔗 Live demo:** https://wattsworth-4em4jgncpi8gh2yt2rkayk.streamlit.app/

An end-to-end energy-analytics application for a heavy-industry plant (modelled on a
100 TPD coal-based DRI / sponge-iron unit). It implements the methods real energy
managers use — **ISO 50001** baselining and **IPMVP Option C** measurement &
verification — and turns plant data into **₹ and tonnes of CO₂** decisions.

Built by an electrical engineer to bridge the plant floor and the P&L: every number is
expressed in money and carbon, not just kWh.

> **Try the demo as a detective story:** the sample data hides a real **6% efficiency
> project on 2025-09-01** and **5 abnormal energy days** — none of them labelled. See if
> the engine finds them.

---

## What it does

| Page | Method | Output |
|------|--------|--------|
| **📊 Energy Baseline** | ISO 50001 OLS regression of energy on production, ambient temperature, and operating hours, with backward elimination of insignificant drivers | R², CV(RMSE) (IPMVP ≤ 20% gate), fitted coefficients, actual-vs-expected |
| **✅ Savings M&V** | IPMVP Option C: actual vs. baseline-expected energy, with a one-sided significance test and CUSUM | Verified saving in MWh, **₹**, **tCO₂**, annualised, with statistical significance |
| **🔮 Forecast & Anomaly** | Holt-Winters forecast (95% band) + residual-spike and IsolationForest anomaly detection | Forecast outlook + flagged abnormal days |
| **⚙️ Optimization** | Motor right-sizing (efficiency-vs-load), time-of-use load shifting, power-factor correction | Ranked actions with ₹/yr, tCO₂/yr, and payback |

## Architecture

```
WattsWorth/
├── Home.py                 # landing dashboard
├── pages/                  # 4 Streamlit module pages (thin UI only)
├── engine/                 # pure-Python analytics — NO Streamlit imports, fully tested
│   ├── data.py             # synthetic plant data + ground-truth harness
│   ├── baseline.py         # ISO 50001 OLS baseline
│   ├── mv.py               # IPMVP Option C savings + CUSUM + significance
│   ├── anomaly.py          # residual-spike + IsolationForest detectors
│   ├── forecast.py         # Holt-Winters forecast
│   ├── optimize.py         # motor / ToU / power-factor optimisers
│   ├── economics.py        # ₹ conversions      carbon.py  # tCO₂ conversions
│   └── config.py           # constants (tariffs, emission factor, thresholds)
├── data/                   # generated sample CSVs
└── tests/                  # 61 tests (one per engine module)
```

The `engine/` package is UI-free and independently testable; the Streamlit pages only
orchestrate and render. This separation is what makes the engine portable to an API,
a database, or a multi-site deployment without touching the analytics.

## Run it

```bash
# Windows (PowerShell)
cd D:\Amrita\WattsWorth
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit run Home.py
```

Opens at http://localhost:8501. On first launch, click **Generate sample data** (or it
loads the CSVs in `data/`).

## Test it

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## About the data

The dataset is **synthetic by design** — a labelled validation harness. The generator
injects a known 6% energy-intensity step change and 5 anomaly spikes, records the ground
truth in `DataFrame.attrs`, and deliberately does **not** persist those labels to CSV, so
the analytics must rediscover the events from the data alone. This is how you'd validate a
detector before trusting it on a real plant.

The same engine is **also validated on a real public dataset** (UCI Steel Industry Energy
Consumption — DAEWOO Steel, Korea): it generalises with sensible coefficients, its
diagnostics correctly flag the autocorrelation in real data, and it honestly reports the
baseline as *not* IPMVP-certifiable on proxy drivers. See
**[docs/PUBLIC_DATASET_VALIDATION.md](docs/PUBLIC_DATASET_VALIDATION.md)**.

## Standards implemented

ISO 50001 (EnPI / energy baseline) · IPMVP Option C (M&V) · ASHRAE Guideline 14 (CV(RMSE)
acceptance) · CUSUM · specific energy consumption (SEC) benchmarking · power-factor /
demand-charge economics.

See **[docs/METHODS_AND_LIMITATIONS.md](docs/METHODS_AND_LIMITATIONS.md)** for a candid
account of the methods, their assumptions, and where the tool is simplified — including the
clean-vs-realistic data modes, in-sample vs out-of-sample validation, and the
autocorrelation-corrected significance test.

## Tech stack

Python 3.13 · Streamlit · pandas · numpy · statsmodels · scikit-learn · plotly.

## Status & roadmap

**MVP complete** — 5 working pages, tested engine, runs locally. Planned next: PDF/Excel
M&V report export, multi-site fleet benchmarking, SQLite persistence (versioned baselines),
driver-based scenario forecasting, and public deployment.
