"""Forecast & Anomaly — operational outlook and abnormal-day detection.

Two tools an operator uses day-to-day: spot the abnormal energy days that just
happened, and see where consumption is heading next.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import anomaly, baseline, data, forecast as fc

st.set_page_config(page_title="Forecast & Anomaly", page_icon="🔮", layout="wide")

st.title("🔮 Forecast & Anomaly Detection")
st.caption("Catch abnormal energy days, and forecast where consumption is heading.")


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame | None:
    return data.load_plant_energy()


plant = _load()
if plant is None:
    st.warning("No dataset found. Go to **Home** and click *Generate sample data* first.")
    st.stop()

plant = plant.copy()
plant["date"] = pd.to_datetime(plant["date"])
_FP = (len(plant), float(plant["energy_kwh"].sum()))


@st.cache_resource(show_spinner=False)
def _fit_full(fp: tuple) -> baseline.BaselineModel:
    return baseline.fit_baseline(plant)


@st.cache_resource(show_spinner=False)
def _multivariate(fp: tuple, contamination: float) -> pd.DataFrame:
    return anomaly.detect_multivariate_anomalies(plant, contamination=contamination)


@st.cache_resource(show_spinner=False)
def _forecast_cached(fp: tuple, horizon: int):
    return fc.forecast(plant, horizon_days=horizon)


# ============================================================ ANOMALY DETECTION
st.subheader("1 · Anomaly detection")
ac1, ac2 = st.columns(2)
sigma = ac1.slider("Sensitivity (σ control limit)", 2.0, 5.0, 3.0, 0.5,
                   help="Lower = more sensitive (more flags).")
method = ac2.radio("Method", ["Baseline residual (spikes)", "Multivariate (IsolationForest)"],
                   horizontal=False)

try:
    if method.startswith("Baseline"):
        model = _fit_full(_FP)
        result = anomaly.detect_residual_anomalies(plant, model, sigma=sigma)
        flagged = result[result["is_anomaly"]]
    else:
        result = _multivariate(_FP, 0.02)
        flagged = result[result["is_anomaly"]]
except ValueError as exc:
    st.error(f"Detection failed: {exc}")
    st.stop()

m1, m2 = st.columns(2)
m1.metric("Days analysed", f"{len(result):,}")
m2.metric("Anomalies flagged", f"{len(flagged)}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=plant["date"], y=plant["energy_kwh"], name="Energy",
                         line=dict(color="#6B2737", width=1.1)))
if not flagged.empty and "date" in flagged.columns:
    marks = flagged[["date"]].merge(plant[["date", "energy_kwh"]], on="date", how="left")
    fig.add_trace(go.Scatter(
        x=marks["date"], y=marks["energy_kwh"], name="Anomaly", mode="markers",
        marker=dict(color="#C8102E", size=9, symbol="x"),
    ))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                  yaxis_title="Energy (kWh/day)", plot_bgcolor="rgba(0,0,0,0)",
                  legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
st.plotly_chart(fig, use_container_width=True)

with st.expander(f"Flagged days ({len(flagged)})"):
    if flagged.empty:
        st.caption("No anomalies at this sensitivity.")
    else:
        cols = [c for c in ("date", "actual_kwh", "expected_kwh", "robust_z", "direction",
                            "anomaly_score") if c in flagged.columns]
        st.dataframe(flagged[cols].sort_values(cols[0]), use_container_width=True, hide_index=True)

# ===================================================================== FORECAST
st.subheader("2 · Energy forecast")
horizon = st.slider("Forecast horizon (days)", 7, 60, 21, 1)

try:
    result_fc = _forecast_cached(_FP, horizon)
except ValueError as exc:
    st.error(f"Forecast failed: {exc}")
    st.stop()

st.caption(f"Method: **{result_fc.method}** · 95% confidence band from in-sample residual scatter.")

history = plant.tail(120)
ff = go.Figure()
ff.add_trace(go.Scatter(x=history["date"], y=history["energy_kwh"], name="History",
                        line=dict(color="#6B2737", width=1.2)))
ff.add_trace(go.Scatter(x=result_fc.upper.index, y=result_fc.upper.to_numpy(),
                        name="Upper 95%", line=dict(width=0), showlegend=False))
ff.add_trace(go.Scatter(x=result_fc.lower.index, y=result_fc.lower.to_numpy(),
                        name="95% band", fill="tonexty", fillcolor="rgba(181,138,60,0.20)",
                        line=dict(width=0)))
ff.add_trace(go.Scatter(x=result_fc.forecast.index, y=result_fc.forecast.to_numpy(),
                        name="Forecast", line=dict(color="#B58A3C", width=2.2, dash="dash")))
ff.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                 yaxis_title="Energy (kWh/day)", plot_bgcolor="rgba(0,0,0,0)",
                 legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
st.plotly_chart(ff, use_container_width=True)

f1, f2 = st.columns(2)
f1.metric("Mean forecast", f"{result_fc.forecast.mean():,.0f} kWh/day")
f2.metric("Horizon total", f"{result_fc.forecast.sum() / 1000:,.1f} MWh")
