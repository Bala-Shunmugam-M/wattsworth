"""WattsWorth — Home / Overview dashboard.

Streamlit entry point. Loads data if present (offers to generate sample data
otherwise) and renders a placeholder KPI dashboard. All heavy lifting lives in
the pure-Python ``engine`` package; this file only orchestrates and renders.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import baseline, carbon, config, data, economics

st.set_page_config(
    page_title="WattsWorth — Industrial Energy Intelligence",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load plant-energy and motor data (cached)."""
    return data.load_plant_energy(), data.load_motors()


def _kpi_row(plant: pd.DataFrame) -> None:
    """Render the headline KPI tiles from available data."""
    total_kwh = float(plant["energy_kwh"].sum())
    sec = float(baseline.specific_energy_consumption(plant).mean())
    inr = economics.energy_cost(total_kwh)
    tco2 = carbon.co2_emissions_tonnes(total_kwh)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Specific Energy", f"{sec:,.0f} kWh/t")
    c2.metric("Energy (period)", f"{total_kwh / 1000:,.0f} MWh")
    c3.metric("Energy cost", f"{config.CURRENCY_SYMBOL}{inr / 1e5:,.1f} L")
    c4.metric("CO₂ (period)", f"{tco2:,.0f} t")


def main() -> None:
    """Render the Home page."""
    st.title("⚡ WattsWorth")
    st.caption(
        "Industrial Energy Performance & Decarbonization Engine — "
        "ISO 50001 · IPMVP M&V · Forecast · Optimization"
    )

    plant, motors = _load()

    if plant is None:
        st.warning(
            "No dataset found. Generate credible synthetic plant data to explore the app."
        )
        if st.button("⚙️ Generate sample data", type="primary"):
            with st.spinner("Generating synthetic plant + motor data…"):
                data.save_synthetic_data()
            st.cache_data.clear()
            st.rerun()
        st.stop()

    n_motors = 0 if motors is None else len(motors)
    st.success(f"Loaded {len(plant):,} days of plant data and {n_motors} motors.")
    _kpi_row(plant)

    st.divider()
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Energy trend")
        st.line_chart(plant.set_index("date")["energy_kwh"], height=280)
    with right:
        st.subheader("Modules")
        st.markdown(
            "- **Energy Baseline** — ISO 50001 regression *(build step 2)*\n"
            "- **Savings M&V** — IPMVP Option C *(build step 3)*\n"
            "- **Forecast & Anomaly** *(build step 4)*\n"
            "- **Optimization** *(build step 5)*"
        )
        st.info("Engine stubs are wired and import-clean. Each page is a placeholder until its step lands.")

    with st.expander("Preview raw data"):
        st.dataframe(plant.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
