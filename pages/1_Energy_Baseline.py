"""Energy Baseline (ISO 50001) — interactive baseline regression.

Lets the user choose a baseline period, fits the OLS energy model on it, and
shows fit quality (R², CV(RMSE)), the fitted driver coefficients, and an
actual-vs-expected chart over the full series. The gap that opens after the
baseline window is the energy saving that build step 3 (M&V) will quantify in
₹ and tonnes of CO₂.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import baseline, data

st.set_page_config(page_title="Energy Baseline", page_icon="📊", layout="wide")

st.title("📊 Energy Baseline — ISO 50001")
st.caption(
    "Model expected energy from operational drivers (production, ambient temperature, "
    "operating hours). Everything downstream is a deviation from this expectation."
)


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame | None:
    """Load the plant-energy series (cached)."""
    return data.load_plant_energy()


plant = _load()
if plant is None:
    st.warning("No dataset found. Go to the **Home** page and click *Generate sample data* first.")
    st.stop()

plant = plant.copy()
plant["date"] = pd.to_datetime(plant["date"])
min_date = plant["date"].min().date()
max_date = plant["date"].max().date()

# Default baseline window: pre-intervention (data ships with a 2025-09-01 step).
default_end = min(pd.Timestamp("2025-08-31").date(), max_date)

st.subheader("1 · Choose the baseline period")
st.caption(
    "Pick a clean, representative window of *normal* operation — ideally before any "
    "efficiency project. A baseline mixing two regimes (or full of anomalies) fits poorly."
)
col_a, col_b = st.columns(2)
start = col_a.date_input("Baseline start", value=min_date, min_value=min_date, max_value=max_date)
end = col_b.date_input("Baseline end", value=default_end, min_value=min_date, max_value=max_date)

if start >= end:
    st.error("Baseline start must be before baseline end.")
    st.stop()

# --- Fit (cached so re-runs that don't change the window don't refit) --------
@st.cache_resource(show_spinner=False)
def _fit_cached(fp: tuple, start_iso: str, end_iso: str) -> baseline.BaselineModel:
    return baseline.fit_baseline(plant, baseline_period=(start_iso, end_iso))


_FP = (len(plant), float(plant["energy_kwh"].sum()))
try:
    model = _fit_cached(_FP, str(start), str(end))
except ValueError as exc:
    st.error(f"Could not fit baseline: {exc}")
    st.stop()

expected_full = baseline.predict_expected(model, plant)

st.subheader("2 · Baseline fit quality")
k1, k2, k3, k4 = st.columns(4)
k1.metric("R²", f"{model.r_squared:.3f}", help="Share of energy variation explained. Higher is better.")
k2.metric("CV(RMSE)", f"{model.cv_rmse * 100:.1f}%", help="IPMVP wants this below 20%.")
k3.metric("Observations", f"{model.n_obs}", help="Days used to fit the baseline.")
k4.metric("Drivers kept", f"{len(model.drivers)}/{len(model.coefficients) - 1 + len(model.dropped)}")

if model.cv_rmse < 0.20 and model.r_squared > 0.75:
    st.success("✅ This is a statistically usable ISO 50001 baseline (CV(RMSE) < 20%).")
else:
    st.warning(
        "⚠️ Marginal fit. Try a cleaner / single-regime window — e.g. end the baseline "
        "before the 2025-09-01 efficiency project."
    )

with st.expander("Regression diagnostics — ISO 50001 / ASHRAE G14 rigor"):
    d1, d2, d3, d4 = st.columns(4)
    oos = "—" if model.cv_rmse_oos is None else f"{model.cv_rmse_oos * 100:.1f}%"
    d1.metric("Out-of-sample CV(RMSE)", oos,
              help="Hold-out predictive error (last 20% of the window). The honest metric — "
                   "in-sample CV(RMSE) is optimistic.")
    d2.metric("Durbin–Watson", f"{model.durbin_watson:.2f}",
              help="≈2 = no residual autocorrelation; <1.5 inflates significance.")
    d3.metric("Max VIF", "—" if np.isnan(model.max_vif) else f"{model.max_vif:.1f}",
              help=">10 indicates problematic multicollinearity among drivers.")
    d4.metric("Breusch–Pagan p", "—" if np.isnan(model.bp_pvalue) else f"{model.bp_pvalue:.2f}",
              help="<0.05 = heteroskedastic (non-constant error variance).")
    st.caption(
        "These check the regression's *assumptions*, not just its fit — exactly what an "
        "M&V auditor scrutinises before accepting a baseline."
    )

with st.expander("Model details — coefficients & dropped drivers"):
    coef_df = pd.DataFrame(
        {"term": list(model.coefficients), "coefficient": list(model.coefficients.values())}
    )
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.markdown(
        "Each coefficient is the marginal kWh per unit of that driver "
        "(e.g. kWh per extra tonne of production)."
    )
    if model.dropped:
        st.info(f"Dropped as statistically insignificant (p > 0.05): {', '.join(model.dropped)}")
    else:
        st.caption("No drivers dropped — all were statistically significant.")

# --- Actual vs expected ------------------------------------------------------
st.subheader("3 · Actual vs. expected energy")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=plant["date"], y=plant["energy_kwh"], name="Actual",
        line=dict(color="#6B2737", width=1.4),
    )
)
fig.add_trace(
    go.Scatter(
        x=plant["date"], y=expected_full, name="Expected (baseline)",
        line=dict(color="#B58A3C", width=1.8, dash="dash"),
    )
)
fig.add_vrect(
    x0=str(start), x1=str(end),
    fillcolor="#6B2737", opacity=0.06, line_width=0,
    annotation_text="baseline window", annotation_position="top left",
)
fig.update_layout(
    height=380, margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="Energy (kWh/day)", xaxis_title=None,
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# --- Post-baseline gap (teaser for M&V) --------------------------------------
post = plant[plant["date"] > pd.Timestamp(end)]
if not post.empty:
    exp_post = baseline.predict_expected(model, post)
    gap = 1.0 - post["energy_kwh"].sum() / exp_post.sum()
    st.subheader("4 · What happened after the baseline window?")
    m1, m2 = st.columns(2)
    m1.metric(
        "Energy vs. baseline expectation",
        f"{gap * 100:+.1f}%",
        delta=f"{'saving' if gap > 0 else 'excess'}",
        delta_color="inverse",
    )
    m2.metric("Reporting-period days", f"{len(post)}")
    st.caption(
        "A negative gap means the plant used **less** energy than the baseline predicted — "
        "i.e. a real saving. Build step 3 (Savings M&V) turns this into ₹ and tonnes of CO₂ "
        "with statistical confidence."
    )
