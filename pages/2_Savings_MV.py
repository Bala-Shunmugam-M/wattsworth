"""Savings M&V (IPMVP Option C) — verify and value energy savings.

Fits a baseline on a pre-intervention window, then measures how much less energy
the plant used afterwards, expresses it in ₹ and tonnes of CO2, projects it to a
year, and states whether the saving is statistically significant.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import baseline, config, data, mv
from reporting import build_mv_report_pdf, build_mv_report_xlsx
from reporting.meta import ReportMeta

st.set_page_config(page_title="Savings M&V", page_icon="✅", layout="wide")

st.title("✅ Savings M&V — IPMVP Option C")
st.caption(
    "How much energy did an efficiency project actually save? Measured against the "
    "baseline's expectation, valued in ₹ and CO₂, and tested for statistical significance."
)


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame | None:
    return data.load_plant_energy()


plant = _load()
if plant is None:
    st.warning("No dataset found. Go to **Home** and click *Generate sample data* first.")
    st.stop()

plant = plant.copy()
plant["date"] = pd.to_datetime(plant["date"])
min_date, max_date = plant["date"].min().date(), plant["date"].max().date()

# --- Controls ----------------------------------------------------------------
st.subheader("1 · Define baseline and intervention")
c1, c2, c3 = st.columns(3)
baseline_start = c1.date_input("Baseline start", value=min_date, min_value=min_date, max_value=max_date)
intervention = c2.date_input(
    "Intervention date", value=min(pd.Timestamp("2025-09-01").date(), max_date),
    min_value=min_date, max_value=max_date,
    help="Reporting period starts here. Baseline is fit on everything before it.",
)
tariff = c3.number_input("Tariff (₹/kWh)", value=float(config.TARIFF_FLAT_INR_PER_KWH), step=0.5, min_value=0.0)

emission = st.slider(
    "Grid emission factor (kg CO₂/kWh)", 0.0, 1.5,
    float(config.GRID_EMISSION_FACTOR_KG_PER_KWH), 0.01,
)

if baseline_start >= intervention:
    st.error("Baseline start must be before the intervention date.")
    st.stop()

baseline_end = (pd.Timestamp(intervention) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# --- Fit + measure (baseline fit cached; tariff/CO2 sliders don't refit) ------
@st.cache_resource(show_spinner=False)
def _fit_cached(fp: tuple, start_iso: str, end_iso: str) -> baseline.BaselineModel:
    return baseline.fit_baseline(plant, baseline_period=(start_iso, end_iso))


try:
    _fp = (len(plant), float(plant["energy_kwh"].sum()))
    model = _fit_cached(_fp, str(baseline_start), baseline_end)
    avoided = mv.compute_savings(plant, model, str(intervention))
    summary = mv.summarize_savings(avoided, tariff_inr_per_kwh=tariff, emission_factor=emission)
except ValueError as exc:
    st.error(f"Could not compute savings: {exc}")
    st.stop()

st.caption(
    f"Baseline fit on {model.n_obs} days · R²={model.r_squared:.3f} · "
    f"CV(RMSE)={model.cv_rmse * 100:.1f}% · reporting period = {summary.days} days."
)

# --- Headline savings --------------------------------------------------------
st.subheader("2 · Verified savings (reporting period)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Energy avoided", f"{summary.avoided_kwh / 1000:,.1f} MWh", f"{summary.pct_saving * 100:.1f}%")
k2.metric("Cost saved", f"{config.CURRENCY_SYMBOL}{summary.inr_saved / 1e5:,.2f} L")
k3.metric("CO₂ avoided", f"{summary.tonnes_co2_avoided:,.0f} t")
k4.metric("Avg daily saving", f"{summary.avoided_kwh / max(summary.days, 1):,.0f} kWh")

_sig_detail = (
    f"autocorrelation-corrected one-sided t-test: t={summary.t_stat:.1f}, "
    f"p={summary.p_value:.1e}, ρ={summary.lag1_autocorr:.2f}, "
    f"effective N={summary.n_effective:.0f} of {summary.days} days"
)
if summary.is_significant:
    st.success(
        f"✅ **Statistically significant saving** ({_sig_detail}). A real reduction, not noise — "
        "and the test accounts for serial correlation, which most M&V tools ignore."
    )
else:
    st.warning(
        f"⚠️ Saving is **not** statistically significant ({_sig_detail}). "
        "Treat with caution — it may be within baseline noise."
    )

# --- Annualised --------------------------------------------------------------
st.subheader("3 · Projected to a full year")
a1, a2, a3 = st.columns(3)
a1.metric("Annual energy", f"{summary.annual_kwh / 1000:,.0f} MWh/yr")
a2.metric("Annual cost", f"{config.CURRENCY_SYMBOL}{summary.annual_inr / 1e5:,.1f} L/yr")
a3.metric("Annual CO₂", f"{summary.annual_tco2:,.0f} t/yr")

# --- CUSUM chart -------------------------------------------------------------
st.subheader("4 · Cumulative avoided energy (CUSUM)")
st.caption(
    "Each day adds (expected − actual) to a running total. A steady climb means "
    "persistent savings; the slope is the saving rate."
)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=avoided["date"], y=avoided["cumulative_avoided_kwh"] / 1000.0,
        name="Cumulative avoided", fill="tozeroy",
        line=dict(color="#2E6F40", width=2),
    )
)
fig.update_layout(
    height=340, margin=dict(l=10, r=10, t=20, b=10),
    yaxis_title="Cumulative avoided (MWh)", xaxis_title=None,
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# --- Actual vs expected in reporting period ----------------------------------
with st.expander("Reporting-period detail — actual vs. expected"):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=avoided["date"], y=avoided["actual_kwh"], name="Actual",
                              line=dict(color="#6B2737", width=1.2)))
    fig2.add_trace(go.Scatter(x=avoided["date"], y=avoided["expected_kwh"], name="Expected (baseline)",
                              line=dict(color="#B58A3C", width=1.6, dash="dash")))
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                       yaxis_title="Energy (kWh/day)", plot_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(avoided.tail(20), use_container_width=True, hide_index=True)

# --- Download audit report ---------------------------------------------------
st.subheader("5 · Download audit report")
st.caption(
    "Board-ready, audit-grade: baseline equation + diagnostics, the ASHRAE G14 acceptance "
    "verdict, the autocorrelation-corrected significance, the CUSUM chart, and a methodology "
    "& limitations appendix. This is the deliverable a pilot is judged on."
)
report_meta = ReportMeta(
    facility_name="Demo Plant",
    report_id=f"WW-MV-{intervention:%Y-%m}",
    prepared_by="WattsWorth",
    baseline_start=str(baseline_start),
    baseline_end=baseline_end,
    reporting_start=str(avoided["date"].min().date()),
    reporting_end=str(avoided["date"].max().date()),
    tariff_inr_per_kwh=tariff,
    emission_factor=emission,
    generated_at=datetime.now().isoformat(timespec="seconds"),
)
try:
    pdf_bytes = build_mv_report_pdf(summary, model, avoided, report_meta)
    xlsx_bytes = build_mv_report_xlsx(summary, model, avoided, report_meta)
    rc1, rc2 = st.columns(2)
    rc1.download_button(
        "⬇️ M&V report (PDF)", pdf_bytes, file_name="wattsworth_mv_report.pdf",
        mime="application/pdf", use_container_width=True,
    )
    rc2.download_button(
        "⬇️ M&V workbook (Excel)", xlsx_bytes, file_name="wattsworth_mv_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
except ValueError as exc:
    st.warning(f"Report unavailable: {exc}")
