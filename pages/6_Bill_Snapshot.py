"""Bill Snapshot — first-look from a prospect's monthly electricity bills.

The on-ramp before a full ISO 50001 baseline: a real prospect rarely has a year of
daily driver data, but they do have a few monthly bills. Type or upload 3–12 of
them and get the exploratory **Energy Snapshot** — the same document the Discovery
Kit offers in outreach — generated from their real numbers, nothing invented.

Self-contained and additive: imports only ``engine.bills`` and writes nothing.
"""
from __future__ import annotations

from datetime import date as _date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import bills, config

st.set_page_config(page_title="Bill Snapshot", page_icon="🧾", layout="wide")

MAROON = "#6B2737"
GOLD = "#B58A3C"

st.title("🧾 Bill Snapshot")
st.caption(
    "A first-look read from a prospect's monthly electricity bills — the exploratory "
    "Energy Snapshot you can send after a discovery call. **Not an audit, not a guarantee.** "
    "Every figure is computed from the bills you enter; nothing is invented."
)

# --- Who is this for ---------------------------------------------------------
st.subheader("1 · Who is this for?")
c1, c2, c3 = st.columns(3)
company = c1.text_input("Company / facility", value=st.session_state.get("facility_name", ""),
                        placeholder="e.g. Shrivik Industries")
prepared_by = c2.text_input("Prepared by", value="WattsWorth")
snap_date = c3.date_input("Date", value=_date.today())
if company:
    st.session_state["facility_name"] = company  # the M&V report picks this up too

# --- Bills input -------------------------------------------------------------
st.subheader("2 · Enter the bills")
st.caption(
    "Minimum: **month**, **energy_kwh**, **amount_inr**. Optional but valuable: "
    "**max_demand_kva**, **power_factor** (power factor is the fastest, most confirmable saving to check)."
)

_TEMPLATE = pd.DataFrame({
    "month": ["2025-01", "2025-02", "2025-03"],
    "energy_kwh": [None, None, None],
    "amount_inr": [None, None, None],
    "max_demand_kva": [None, None, None],
    "power_factor": [None, None, None],
})

mode = st.radio("Input method", ["Type the bills", "Upload a CSV"], horizontal=True)

raw: pd.DataFrame | None = None
if mode == "Type the bills":
    seed = pd.DataFrame({
        "month": ["", "", "", "", "", ""],
        "energy_kwh": [None] * 6, "amount_inr": [None] * 6,
        "max_demand_kva": [None] * 6, "power_factor": [None] * 6,
    })
    edited = st.data_editor(
        seed, num_rows="dynamic", use_container_width=True, hide_index=True, key="bills_editor",
        column_config={
            "month": st.column_config.TextColumn(help="Billing month — e.g. 2025-01 or Jan 2025"),
            "energy_kwh": st.column_config.NumberColumn(help="Units consumed (kWh)", min_value=0.0),
            "amount_inr": st.column_config.NumberColumn(help="Total bill (₹)", min_value=0.0),
            "max_demand_kva": st.column_config.NumberColumn(help="Recorded max demand (kVA) — optional", min_value=0.0),
            "power_factor": st.column_config.NumberColumn(help="Average power factor 0–1 — optional", min_value=0.0, max_value=1.0),
        },
    )
    raw = edited[edited["month"].astype(str).str.strip() != ""]
    st.download_button("⬇️ Blank CSV template", _TEMPLATE.to_csv(index=False).encode("utf-8"),
                       file_name="bill_template.csv", mime="text/csv")
else:
    up = st.file_uploader("CSV with columns: month, energy_kwh, amount_inr [, max_demand_kva, power_factor]", type=["csv"])
    st.download_button("⬇️ Blank CSV template", _TEMPLATE.to_csv(index=False).encode("utf-8"),
                       file_name="bill_template.csv", mime="text/csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            raw = None

if raw is None or raw.empty:
    st.info("Enter at least one bill (3+ months gives a more reliable snapshot).")
    st.stop()

# --- Validate ----------------------------------------------------------------
clean, report = bills.validate_and_clean_bills(raw)
if report.coerced_cells or report.dropped_invalid_rows or report.duplicate_rows_removed:
    st.caption(
        f"Cleaned — {report.coerced_cells} cells coerced, "
        f"{report.dropped_invalid_rows} invalid row(s) dropped, "
        f"{report.duplicate_rows_removed} duplicate month(s) removed."
    )
for m in report.messages:
    st.caption("• " + m)
if not report.ok:
    st.error("No usable bills yet — fill month, energy_kwh and amount_inr for at least one row.")
    st.stop()

snap = bills.compute_snapshot(clean)

# --- Summary -----------------------------------------------------------------
st.subheader("3 · Snapshot")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total spend", f"{config.CURRENCY_SYMBOL}{snap.total_inr / 1e5:,.2f} L")
k2.metric("Avg / month", f"{config.CURRENCY_SYMBOL}{snap.avg_monthly_inr / 1e5:,.2f} L")
k3.metric("Effective rate", f"{config.CURRENCY_SYMBOL}{snap.effective_rate_inr_per_kwh:,.2f}/kWh")
k4.metric("Energy", f"{snap.total_kwh / 1000:,.1f} MWh")
st.caption(
    f"{snap.n_bills} bill(s) · {snap.period_start} → {snap.period_end} · "
    f"highest **{snap.highest_month}** ({config.CURRENCY_SYMBOL}{snap.highest_month_inr:,.0f}) · "
    f"lowest **{snap.lowest_month}** ({config.CURRENCY_SYMBOL}{snap.lowest_month_inr:,.0f})"
)

# --- Charts ------------------------------------------------------------------
chart = clean.copy()
chart["rate"] = chart["amount_inr"] / chart["energy_kwh"]
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown("##### Monthly spend (₹)")
    fig = go.Figure(go.Bar(x=chart["month"], y=chart["amount_inr"], marker_color=MAROON))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", yaxis_title="₹")
    st.plotly_chart(fig, use_container_width=True)
with cc2:
    st.markdown("##### Effective ₹/kWh")
    fig2 = go.Figure(go.Scatter(x=chart["month"], y=chart["rate"], mode="lines+markers",
                                line=dict(color=GOLD, width=2)))
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       plot_bgcolor="rgba(0,0,0,0)", yaxis_title="₹/kWh")
    st.plotly_chart(fig2, use_container_width=True)

# --- Observations ------------------------------------------------------------
st.subheader("4 · Observations")
st.caption("Only what is visible in the bills. Confidence is stated; the one indicative ₹ figure is an assumption to confirm against the tariff.")
_BADGE = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "⚪ Low"}
for i, o in enumerate(snap.observations, start=1):
    with st.container(border=True):
        st.markdown(f"**{i}. {o.title}**  &nbsp; {_BADGE.get(o.confidence, o.confidence)} confidence")
        st.write(o.detail)
        if o.indicative_inr_per_year is not None:
            st.caption(f"💡 Indicative impact (confirm against tariff): ~{config.CURRENCY_SYMBOL}{o.indicative_inr_per_year:,.0f}/year")

st.markdown("##### Questions worth investigating")
for q in snap.questions:
    st.markdown(f"- {q}")

# --- Download the Energy Snapshot --------------------------------------------
st.subheader("5 · Send it")
st.caption("Download the exploratory Energy Snapshot (Markdown) — the same template the Discovery Kit uses, filled from these bills.")
md = bills.render_snapshot_markdown(snap, company=company, date=str(snap_date), prepared_by=prepared_by or "WattsWorth")
d1, d2 = st.columns(2)
d1.download_button("⬇️ Energy Snapshot (Markdown)", md.encode("utf-8"),
                   file_name=f"energy_snapshot_{(company or 'prospect').replace(' ', '_')}.md",
                   mime="text/markdown", type="primary", use_container_width=True)
d2.download_button("⬇️ Cleaned bills (CSV)", clean[bills.BILL_COLUMNS].to_csv(index=False).encode("utf-8"),
                   file_name="bills_cleaned.csv", mime="text/csv", use_container_width=True)
with st.expander("Preview the Energy Snapshot"):
    st.markdown(md)
