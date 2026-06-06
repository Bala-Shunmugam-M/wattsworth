"""WattsWorth — Home / Overview dashboard.

Streamlit entry point. Loads data if present (offers to generate sample data
otherwise) and renders a placeholder KPI dashboard. All heavy lifting lives in
the pure-Python ``engine`` package; this file only orchestrates and renders.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import baseline, carbon, config, data, economics, ingestion
from readiness import assess_readiness

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
        st.warning("No dataset found. Generate plant data to explore the app.")
        mode = st.radio(
            "Data mode",
            ["Clean demo", "Realistic (hard-mode)"],
            horizontal=True,
            help="Clean = exactly-linear demo (every method looks perfect). "
            "Realistic = non-linear, autocorrelated, collinear drivers + missing days, "
            "so you can see the engine degrade honestly (lower R², Durbin-Watson < 2).",
        )
        if st.button("⚙️ Generate sample data", type="primary"):
            with st.spinner("Generating plant + motor data…"):
                data.save_synthetic_data(mode="realistic" if mode.startswith("Realistic") else "clean")
            st.cache_data.clear()
            st.cache_resource.clear()
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
            "- **📊 Energy Baseline** — ISO 50001 regression that learns the plant's normal energy\n"
            "- **✅ Savings M&V** — IPMVP Option C: verified savings in ₹ and CO₂\n"
            "- **🔮 Forecast & Anomaly** — load forecast + abnormal-day detection\n"
            "- **⚙️ Optimization** — ranked actions with payback"
        )
        st.info(
            "💡 The demo data hides a real **6% efficiency project on 2025-09-01** and "
            "**5 anomaly days** — explore the sidebar pages and see if the engine finds them."
        )

    with st.expander("Preview raw data"):
        st.dataframe(plant.head(50), use_container_width=True)

    with st.expander("⚙️ Regenerate / switch data mode"):
        regen_mode = st.radio(
            "Data mode", ["Clean demo", "Realistic (hard-mode)"], horizontal=True, key="regen_mode",
            help="Realistic mode adds non-linearity, autocorrelated noise, collinear drivers, "
            "and missing days — the engine then reports a modest R² and a Durbin-Watson < 2.",
        )
        if st.button("Regenerate data"):
            with st.spinner("Regenerating…"):
                data.save_synthetic_data(mode="realistic" if regen_mode.startswith("Realistic") else "clean")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    with st.expander("📤 Upload your own plant data (CSV)"):
        st.caption(
            "Columns: date, energy_kwh, production_tonnes, ambient_temp_c, operating_hours, "
            "grid_pf, tariff_period. Messy data (commas, blanks, duplicates, out-of-range) is "
            "cleaned and validated before use."
        )
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read CSV: {exc}")
            else:
                cleaned, report = ingestion.validate_and_clean_plant_energy(raw)
                r1, r2, r3 = st.columns(3)
                r1.metric("Rows received", report.rows_in)
                r2.metric("Rows accepted", report.rows_out)
                r3.metric("Rows dropped", report.dropped_invalid_rows)
                if any([report.coerced_cells, report.duplicate_rows_removed,
                        report.clipped_cells, report.date_gaps]):
                    st.caption(
                        f"Cleaned — {report.coerced_cells} cells coerced, "
                        f"{report.duplicate_rows_removed} duplicate dates removed, "
                        f"{report.clipped_cells} values clipped, {report.date_gaps} date gaps."
                    )
                for msg in report.messages:
                    st.caption("• " + msg)
                if report.ok:
                    rr = assess_readiness(cleaned)
                    _icon = {"READY": "✅", "MARGINAL": "⚠️", "NOT_READY": "⛔"}.get(rr.verdict, "•")
                    st.markdown(f"### {_icon} M&V Readiness: **{rr.verdict}** &nbsp; ({rr.score:.0f}/100)")
                    st.caption(rr.headline)
                    st.caption(rr.summary)
                    with st.expander("Readiness checks (what each means + what to do)"):
                        st.dataframe(
                            pd.DataFrame([
                                {"Check": c.name, "Status": c.status,
                                 "What it means": c.detail, "Action": c.action}
                                for c in rr.checks
                            ]),
                            use_container_width=True, hide_index=True,
                        )
                    if st.button("Use this dataset", type="primary"):
                        cleaned.to_csv(config.PLANT_ENERGY_CSV, index=False)
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.success("Dataset loaded.")
                        st.rerun()
                else:
                    st.error("Dataset unusable: " + "; ".join(report.messages))


if __name__ == "__main__":
    main()
