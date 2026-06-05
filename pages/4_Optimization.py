"""Optimization — ranked energy-saving actions with ₹, CO2, and payback.

Runs the three optimisers (motor right-sizing, time-of-use shifting, power-factor
correction) against the plant data and asset register, with what-if controls for
tariff and shiftable load.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import config, data, optimize

st.set_page_config(page_title="Optimization", page_icon="⚙️", layout="wide")

st.title("⚙️ Optimization — Ranked Actions")
st.caption("Where the money is: ranked savings with rupees, CO₂, and payback for each action.")


@st.cache_data(show_spinner=False)
def _load() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return data.load_plant_energy(), data.load_motors()


plant, motors = _load()
if plant is None or motors is None:
    st.warning("No dataset found. Go to **Home** and click *Generate sample data* first.")
    st.stop()

# --- What-if controls --------------------------------------------------------
st.subheader("1 · Assumptions (what-if)")
c1, c2, c3 = st.columns(3)
tariff = c1.number_input("Tariff (₹/kWh)", value=float(config.TARIFF_FLAT_INR_PER_KWH), step=0.5, min_value=0.0)
shiftable = c2.slider("Shiftable load (%)", 0, 40, int(config.DEFAULT_SHIFTABLE_FRACTION * 100), 1) / 100.0
target_pf = c3.slider("Target power factor", 0.90, 0.99, float(config.TARGET_POWER_FACTOR), 0.01)

actions = optimize.recommend_all(
    motors, plant,
    tariff_inr_per_kwh=tariff,
    shiftable_fraction=shiftable,
    target_pf=target_pf,
)

# --- Portfolio totals --------------------------------------------------------
total_inr = sum(a.inr_saved for a in actions)
total_kwh = sum(a.kwh_saved for a in actions)
total_co2 = sum(a.tonnes_co2_avoided for a in actions)
total_capex = sum(a.capex_inr for a in actions)

st.subheader("2 · Total opportunity")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Annual saving", f"{config.CURRENCY_SYMBOL}{total_inr / 1e5:,.1f} L/yr")
t2.metric("Energy saved", f"{total_kwh / 1000:,.1f} MWh/yr")
t3.metric("CO₂ avoided", f"{total_co2:,.1f} t/yr")
t4.metric("Total capex", f"{config.CURRENCY_SYMBOL}{total_capex / 1e5:,.1f} L")

# --- Ranked action cards -----------------------------------------------------
st.subheader("3 · Ranked actions")
_ICON = {"motor": "🔌", "load_shift": "⏱️", "power_factor": "🔋"}
for i, a in enumerate(actions, start=1):
    with st.container(border=True):
        head, kpi = st.columns([3, 2])
        with head:
            st.markdown(f"**{i}. {_ICON.get(a.category, '•')} {a.action}**")
            payback = "Immediate" if a.payback_months == 0 else (
                "—" if a.payback_months == float("inf") else f"{a.payback_months:.1f} months"
            )
            st.caption(
                f"Category: {a.category} · Capex: {config.CURRENCY_SYMBOL}{a.capex_inr:,.0f} · "
                f"Payback: {payback}"
            )
        with kpi:
            k1, k2, k3 = st.columns(3)
            k1.metric("₹/yr", f"{a.inr_saved / 1e5:,.2f} L")
            k2.metric("MWh/yr", f"{a.kwh_saved / 1000:,.1f}")
            k3.metric("tCO₂/yr", f"{a.tonnes_co2_avoided:,.1f}")

# --- Motor right-size table --------------------------------------------------
st.subheader("4 · Motor right-sizing detail")
rows = []
for _, m in motors.iterrows():
    lf = float(m["load_factor"])
    underloaded = lf < config.UNDERLOADED_LF
    rows.append(
        {
            "Asset": m["asset_id"],
            "Name": m["name"],
            "Rated kW": m["rated_kw"],
            "Measured kW": m["measured_kw"],
            "Load factor": f"{lf:.0%}",
            "Status": "⚠️ Under-loaded" if underloaded else ("🔴 Overloaded" if lf > config.OVERLOADED_LF else "✅ OK"),
            "Recommended kW": round(float(m["measured_kw"]) / config.MOTOR_TARGET_LOAD) if underloaded else "—",
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "Under-loaded motors (load factor < 40%) waste energy at poor efficiency. "
    "Right-sizing to ~75% load recovers it. Overloaded motors are a reliability risk, not an energy one."
)
