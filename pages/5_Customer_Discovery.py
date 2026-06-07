"""Customer Discovery & Pilots — GTM tracker.

Operationalises the WattsWorth Discovery & Pilot Kit (master plan Vol IV.5) inside
the app: log discovery conversations with a live Fit Score, and track pilots with
Verified Value Delivered (VVD) plus a small dashboard.

Self-contained and additive — it reads/writes two CSVs under ``data/`` and touches
no engine module. All editing happens in ``st.data_editor``; nothing is saved until
you click the Save button.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Customer Discovery", page_icon="🧭", layout="wide")

# --- House palette (matches the rest of the app) -----------------------------
MAROON = "#6B2737"
GOLD = "#B58A3C"

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
DISCOVERY_CSV: Path = DATA_DIR / "discovery_tracker.csv"
PILOT_CSV: Path = DATA_DIR / "pilot_tracker.csv"

# Fit Score weights (master plan Vol IV.5): factors are 1-5, score scales to 20-100.
FIT_WEIGHTS = {
    "Pain_Severity_1to5": 0.30,
    "Urgency_1to5": 0.20,
    "Budget_Signal_1to5": 0.20,
    "Frequency_1to5": 0.15,
    "Authority_1to5": 0.15,
}

DISCOVERY_COLUMNS = [
    "ID", "Date", "Contact", "Company", "Segment", "Stakeholder_Type", "Source",
    "Pain_Points", "Pain_Severity_1to5", "Frequency_1to5", "Urgency_1to5",
    "Budget_Signal_1to5", "Authority_1to5", "Current_Solution",
    "Status", "Next_Step", "FollowUp_Date", "Notes",
]
PILOT_COLUMNS = [
    "Pilot_ID", "Customer", "Segment", "Start_Date", "End_Date", "Success_Metric",
    "Baseline_Value", "Current_Value", "Energy_Reduction_Pct",
    "Estimated_Savings_Per_Year", "ROI_Pct", "Users_Active", "Adoption_Pct",
    "Time_Saved_Hrs_Per_Week", "VVD_Verified", "VVD_Value_Per_Year",
    "Status", "Reference_YN", "Next_Step", "Notes",
]

SEGMENTS = ["Manufacturing", "Municipality", "Commercial Building",
            "Industrial Facility", "Smart City", "Other"]
STAKEHOLDERS = ["Decision Maker", "Buyer", "Influencer", "User"]
DISC_STATUS = ["New", "Reached out", "Scheduled", "Interviewed", "Validated",
               "Pilot-candidate", "Disqualified", "Nurture"]
PILOT_STATUS = ["Active", "Converting", "Won", "Lost"]
VVD_STATES = ["N", "Pending", "Y"]
YN = ["Y", "N", "Pending"]


def _load(path: Path, columns: list[str]) -> pd.DataFrame:
    """Load a tracker CSV, or return an empty frame with the right columns."""
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:  # noqa: BLE001 — empty/corrupt file -> start fresh
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)
    # Ensure every expected column exists and is ordered.
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def _fit_score(row: pd.Series) -> float:
    """Fit Score (20-100) from the 1-5 factors. Blank factors count as 0."""
    total = 0.0
    for col, weight in FIT_WEIGHTS.items():
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(val):
            total += weight * float(val)
    return round(20.0 * total, 0)


def _band(score: float) -> str:
    if score >= 80:
        return "🟢 Pursue"
    if score >= 60:
        return "🟡 Nurture"
    return "⚪ Park"


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# =============================================================================
st.title("🧭 Customer Discovery & Pilots")
st.caption(
    "GTM tracker for WattsWorth Energy (master plan Vol IV.5). Log conversations and "
    "pilots here; edit inline, then **Save**. Fit Score and VVD are computed for you."
)

tab_disc, tab_pilot = st.tabs(["🧭 Discovery", "🚀 Pilots"])

# --- Discovery ---------------------------------------------------------------
with tab_disc:
    st.subheader("Discovery conversations")
    st.caption(
        "Score each factor 1 (low) – 5 (high). **Fit Score = "
        "20 × (0.30·Severity + 0.20·Urgency + 0.20·Budget + 0.15·Frequency + 0.15·Authority)** "
        "→ ≥80 pursue · 60–79 nurture · <60 park."
    )

    disc = _load(DISCOVERY_CSV, DISCOVERY_COLUMNS)

    edited_disc = st.data_editor(
        disc,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="disc_editor",
        column_config={
            "Date": st.column_config.TextColumn(help="YYYY-MM-DD"),
            "FollowUp_Date": st.column_config.TextColumn(help="YYYY-MM-DD"),
            "Segment": st.column_config.SelectboxColumn(options=SEGMENTS),
            "Stakeholder_Type": st.column_config.SelectboxColumn(options=STAKEHOLDERS),
            "Status": st.column_config.SelectboxColumn(options=DISC_STATUS),
            "Pain_Severity_1to5": st.column_config.NumberColumn(min_value=1, max_value=5, step=1, format="%d"),
            "Frequency_1to5": st.column_config.NumberColumn(min_value=1, max_value=5, step=1, format="%d"),
            "Urgency_1to5": st.column_config.NumberColumn(min_value=1, max_value=5, step=1, format="%d"),
            "Budget_Signal_1to5": st.column_config.NumberColumn(min_value=1, max_value=5, step=1, format="%d"),
            "Authority_1to5": st.column_config.NumberColumn(min_value=1, max_value=5, step=1, format="%d"),
        },
    )

    scored = edited_disc.copy()
    if len(scored):
        scored["Fit_Score"] = scored.apply(_fit_score, axis=1)
        scored["Band"] = scored["Fit_Score"].apply(_band)
    else:
        scored["Fit_Score"] = pd.Series(dtype=float)
        scored["Band"] = pd.Series(dtype=object)

    total = len(scored)
    high = int((scored["Fit_Score"] >= 80).sum()) if total else 0
    avg = float(scored["Fit_Score"].mean()) if total else 0.0
    interviewed = int(edited_disc["Status"].isin(
        ["Interviewed", "Validated", "Pilot-candidate"]).sum()) if total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Conversations", f"{total}", help="Goal: 100 (Vol IV.5).")
    m2.metric("Interviewed+", f"{interviewed}")
    m3.metric("High-fit (≥80)", f"{high}")
    m4.metric("Avg Fit Score", f"{avg:.0f}")

    csave, cdl = st.columns([1, 1])
    if csave.button("💾 Save discovery", type="primary", key="save_disc"):
        try:
            scored[DISCOVERY_COLUMNS + ["Fit_Score", "Band"]].to_csv(DISCOVERY_CSV, index=False)
            st.success(f"Saved {total} rows → {DISCOVERY_CSV.name}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save: {exc}")
    cdl.download_button(
        "⬇️ Download CSV",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="discovery_tracker.csv",
        mime="text/csv",
        key="dl_disc",
    )

    if total:
        st.markdown("##### Pipeline by fit (highest first)")
        st.dataframe(
            scored.sort_values("Fit_Score", ascending=False)[
                ["Company", "Contact", "Segment", "Status", "Fit_Score", "Band", "Next_Step"]
            ],
            use_container_width=True, hide_index=True,
        )

# --- Pilots ------------------------------------------------------------------
with tab_pilot:
    st.subheader("Pilots")
    st.caption(
        "A pilot is a **win** only when VVD is verified. "
        "**VVD = customer-confirmed value the pilot produced (₹/yr), agreed in writing.**"
    )

    pilots = _load(PILOT_CSV, PILOT_COLUMNS)

    edited_pilot = st.data_editor(
        pilots,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="pilot_editor",
        column_config={
            "Start_Date": st.column_config.TextColumn(help="YYYY-MM-DD"),
            "End_Date": st.column_config.TextColumn(help="YYYY-MM-DD"),
            "Segment": st.column_config.SelectboxColumn(options=SEGMENTS),
            "Status": st.column_config.SelectboxColumn(options=PILOT_STATUS),
            "VVD_Verified": st.column_config.SelectboxColumn(options=VVD_STATES),
            "Reference_YN": st.column_config.SelectboxColumn(options=YN),
            "Energy_Reduction_Pct": st.column_config.NumberColumn(format="%.1f%%"),
            "ROI_Pct": st.column_config.NumberColumn(format="%.0f%%"),
            "Adoption_Pct": st.column_config.NumberColumn(format="%.0f%%"),
            "Estimated_Savings_Per_Year": st.column_config.NumberColumn(format="₹%.0f"),
            "VVD_Value_Per_Year": st.column_config.NumberColumn(format="₹%.0f"),
        },
    )

    n = len(edited_pilot)
    active = int((edited_pilot["Status"] == "Active").sum()) if n else 0
    won = int((edited_pilot["Status"] == "Won").sum()) if n else 0
    lost = int((edited_pilot["Status"] == "Lost").sum()) if n else 0
    verified = edited_pilot["VVD_Verified"].astype(str).str.upper() == "Y" if n else pd.Series(dtype=bool)
    vvd_total = float(_num(edited_pilot.loc[verified, "VVD_Value_Per_Year"]).sum()) if n else 0.0
    avg_red = float(_num(edited_pilot["Energy_Reduction_Pct"]).mean()) if n else 0.0
    conv = (won / (won + lost) * 100) if (won + lost) else 0.0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Active pilots", f"{active}")
    p2.metric("Verified VVD", f"₹{vvd_total:,.0f}/yr")
    p3.metric("Avg energy ↓", f"{avg_red:.1f}%")
    p4.metric("Pilot→Won", f"{conv:.0f}%", help="Target ≥ 40% (Vol IV gate).")

    psave, pdl = st.columns([1, 1])
    if psave.button("💾 Save pilots", type="primary", key="save_pilot"):
        try:
            edited_pilot[PILOT_COLUMNS].to_csv(PILOT_CSV, index=False)
            st.success(f"Saved {n} pilots → {PILOT_CSV.name}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save: {exc}")
    pdl.download_button(
        "⬇️ Download CSV",
        edited_pilot.to_csv(index=False).encode("utf-8"),
        file_name="pilot_tracker.csv",
        mime="text/csv",
        key="dl_pilot",
    )

    if n:
        left, right = st.columns(2)
        with left:
            st.markdown("##### Pilots by status")
            counts = edited_pilot["Status"].value_counts().reindex(PILOT_STATUS).fillna(0)
            fig = go.Figure(go.Bar(
                x=list(counts.index), y=list(counts.values),
                marker_color=[GOLD if s in ("Active", "Converting") else MAROON for s in counts.index],
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                              plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Pilots")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("##### Verified VVD by customer (₹/yr)")
            vdf = edited_pilot.loc[verified, ["Customer", "VVD_Value_Per_Year"]].copy()
            vdf["VVD_Value_Per_Year"] = _num(vdf["VVD_Value_Per_Year"])
            if len(vdf):
                fig2 = go.Figure(go.Bar(
                    x=vdf["VVD_Value_Per_Year"], y=vdf["Customer"], orientation="h",
                    marker_color=GOLD,
                ))
                fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                                   plot_bgcolor="rgba(0,0,0,0)", xaxis_title="₹ / year")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No verified VVD yet. A pilot becomes a case study only once VVD is confirmed.")

st.divider()
st.caption(
    "Templates & method: see the master plan **Discovery & Pilot Kit** "
    "(`discovery-kit/`) and **Vol IV.5**. Saved files live in `data/`."
)
