"""Bill Snapshot — a first-look read from a handful of monthly electricity bills.

Where the rest of the engine needs ~a year of clean DAILY driver data, a real
prospect arrives with three-to-twelve MONTHLY bills (kWh + ₹, sometimes max-demand
and power factor). This module turns those into the exploratory **Energy Snapshot**
that opens a discovery conversation — the on-ramp before a full ISO 50001 baseline.

It is deliberately conservative and honest. Every figure is computed from the bills
provided; it never invents a saving. Observations are flagged Low / Medium / High
confidence, and the single indicative ₹ figure (a power-factor penalty estimate) is
labelled as an assumption to confirm against the customer's actual tariff — not a
measured saving. The output is an exploratory discussion document, not an audit.

Pure Python — no Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import carbon, config

# Columns a monthly electricity bill realistically carries.
ESSENTIAL: list[str] = ["month", "energy_kwh", "amount_inr"]
OPTIONAL: list[str] = ["max_demand_kva", "power_factor"]
BILL_COLUMNS: list[str] = ESSENTIAL + OPTIONAL
_NUMERIC: list[str] = ["energy_kwh", "amount_inr", "max_demand_kva", "power_factor"]
_NULL_TOKENS = {"", "nan", "na", "n/a", "null", "none", "-"}

# Indicative power-factor penalty assumption (clearly surfaced; confirm vs. tariff).
# Many Indian utilities levy ~0.5% of the energy bill per 0.01 of PF below target.
PF_PENALTY_FRAC_PER_POINT: float = 0.005
PF_PENALTY_CAP_FRAC: float = 0.15  # never claim more than 15% of the bill


@dataclass
class BillReport:
    """What ingestion had to do to make a set of monthly bills analysable."""

    rows_in: int = 0
    rows_out: int = 0
    missing_columns: list[str] = field(default_factory=list)
    dropped_invalid_rows: int = 0
    duplicate_rows_removed: int = 0
    coerced_cells: int = 0
    messages: list[str] = field(default_factory=list)
    ok: bool = True


@dataclass
class Observation:
    """One data-driven observation from the bills.

    ``indicative_inr_per_year`` is populated only for the power-factor penalty, and
    is an assumption-based estimate (not a measured saving) — always shown labelled.
    """

    code: str
    title: str
    detail: str
    confidence: str  # "Low" | "Medium" | "High"
    indicative_inr_per_year: float | None = None


@dataclass
class BillSnapshot:
    """Computed first-look summary of a set of monthly bills."""

    n_bills: int = 0
    period_start: str = ""
    period_end: str = ""
    total_kwh: float = 0.0
    total_inr: float = 0.0
    avg_monthly_inr: float = 0.0
    effective_rate_inr_per_kwh: float = float("nan")
    rate_cv: float = float("nan")
    highest_month: str = ""
    highest_month_inr: float = 0.0
    lowest_month: str = ""
    lowest_month_inr: float = 0.0
    avg_power_factor: float = float("nan")
    peak_demand_kva: float = float("nan")
    total_tco2: float = 0.0
    observations: list[Observation] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, tolerating thousands separators and null tokens."""
    if not pd.api.types.is_numeric_dtype(series):
        cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
        cleaned = cleaned.where(~cleaned.str.lower().isin(_NULL_TOKENS), other=np.nan)
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def validate_and_clean_bills(df: pd.DataFrame) -> tuple[pd.DataFrame, BillReport]:
    """Validate and clean a raw monthly-bills frame. Never raises on bad data.

    Required columns: ``month`` (any parseable month/date), ``energy_kwh``,
    ``amount_inr``. Optional: ``max_demand_kva``, ``power_factor``.

    Returns ``(clean_df, report)``. ``clean_df`` has columns ``BILL_COLUMNS`` plus a
    parsed ``period`` (Timestamp, month start), sorted ascending and de-duplicated
    by month. It is empty when the data is unusable (see ``report.ok``).
    """
    report = BillReport(rows_in=len(df))
    work = df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]

    missing = [c for c in ESSENTIAL if c not in work.columns]
    if missing:
        report.missing_columns = missing
        report.ok = False
        report.messages.append(f"Missing essential column(s): {missing}.")
        return pd.DataFrame(columns=[*BILL_COLUMNS, "period"]), report

    for col in OPTIONAL:
        if col not in work.columns:
            work[col] = np.nan
            report.messages.append(f"Optional column '{col}' was absent; left blank.")

    # Parse the month label into a sortable period (kept alongside the original label).
    before_na = work["month"].isna().sum()
    work["period"] = pd.to_datetime(work["month"], errors="coerce")
    report.coerced_cells += int(work["period"].isna().sum() - before_na)
    work["month"] = work["month"].astype(str).str.strip()

    for col in _NUMERIC:
        before = work[col].isna().sum()
        work[col] = _coerce_numeric(work[col])
        report.coerced_cells += int(work[col].isna().sum() - before)

    # Range handling.
    work.loc[work["energy_kwh"] <= 0, "energy_kwh"] = np.nan
    work.loc[work["amount_inr"] < 0, "amount_inr"] = np.nan
    pf_outside = ((work["power_factor"] < 0.0) | (work["power_factor"] > 1.0)) & work["power_factor"].notna()
    work.loc[pf_outside, "power_factor"] = np.nan
    work.loc[work["max_demand_kva"] < 0, "max_demand_kva"] = np.nan

    # Drop rows missing any essential value (incl. an unparseable month).
    before_rows = len(work)
    work = work.dropna(subset=["period", "energy_kwh", "amount_inr"])
    report.dropped_invalid_rows = int(before_rows - len(work))

    # De-duplicate on month (keep last), sort chronologically.
    dup = int(work["period"].duplicated().sum())
    report.duplicate_rows_removed = dup
    work = work.drop_duplicates(subset="period", keep="last").sort_values("period").reset_index(drop=True)

    report.rows_out = len(work)
    report.ok = report.rows_out >= 1
    if not report.ok:
        report.messages.append("No usable bills remained after cleaning.")
    elif report.rows_out < 3:
        report.messages.append(
            f"Only {report.rows_out} usable bill(s) — a snapshot is more reliable with 3+ months."
        )

    return work[[*BILL_COLUMNS, "period"]], report


def compute_snapshot(
    df: pd.DataFrame,
    *,
    pf_target: float = config.TARGET_POWER_FACTOR,
    grid_emission_factor: float = config.GRID_EMISSION_FACTOR_KG_PER_KWH,
    pf_penalty_frac_per_point: float = PF_PENALTY_FRAC_PER_POINT,
) -> BillSnapshot:
    """Compute the Bill Snapshot from cleaned monthly bills.

    Args:
        df: Output of :func:`validate_and_clean_bills` (non-empty).
        pf_target: Power-factor threshold below which a penalty is likely.
        grid_emission_factor: kg CO2 per kWh for the CO2 estimate.
        pf_penalty_frac_per_point: Indicative penalty as a fraction of the bill per
            0.01 of PF below ``pf_target`` (an assumption, surfaced to the user).

    Returns:
        A populated :class:`BillSnapshot`. Raises ``ValueError`` if ``df`` is empty.
    """
    if df.empty:
        raise ValueError("Cannot compute a snapshot from an empty bills frame.")

    work = df.sort_values("period").reset_index(drop=True)
    n = len(work)
    kwh = work["energy_kwh"].astype(float)
    inr = work["amount_inr"].astype(float)

    total_kwh = float(kwh.sum())
    total_inr = float(inr.sum())
    avg_monthly_inr = total_inr / n
    eff_rate = total_inr / total_kwh if total_kwh > 0 else float("nan")

    hi = int(inr.values.argmax())
    lo = int(inr.values.argmin())

    per_rate = (inr / kwh.replace(0, np.nan)).to_numpy(dtype=float)
    rate_cv = float(np.nanstd(per_rate) / np.nanmean(per_rate)) if np.nanmean(per_rate) else float("nan")

    avg_pf = float(work["power_factor"].astype(float).mean()) if work["power_factor"].notna().any() else float("nan")
    peak_demand = float(work["max_demand_kva"].astype(float).max()) if work["max_demand_kva"].notna().any() else float("nan")

    snap = BillSnapshot(
        n_bills=n,
        period_start=str(work["month"].iloc[0]),
        period_end=str(work["month"].iloc[-1]),
        total_kwh=total_kwh,
        total_inr=total_inr,
        avg_monthly_inr=avg_monthly_inr,
        effective_rate_inr_per_kwh=eff_rate,
        rate_cv=rate_cv,
        highest_month=str(work["month"].iloc[hi]),
        highest_month_inr=float(inr.iloc[hi]),
        lowest_month=str(work["month"].iloc[lo]),
        lowest_month_inr=float(inr.iloc[lo]),
        avg_power_factor=avg_pf,
        peak_demand_kva=peak_demand,
        total_tco2=float(carbon.co2_emissions_tonnes(total_kwh, grid_emission_factor)),
    )

    obs: list[Observation] = []

    # 1 · Spend baseline (always; pure fact).
    obs.append(Observation(
        code="SPEND",
        title="Electricity spend",
        detail=(
            f"Across {n} bill(s) the facility spent ₹{total_inr:,.0f} "
            f"(≈ ₹{avg_monthly_inr:,.0f}/month) for {total_kwh:,.0f} kWh — "
            f"an effective rate of ₹{eff_rate:,.2f}/kWh."
        ),
        confidence="High",
    ))

    # 2 · Effective-rate volatility.
    if n >= 3 and not np.isnan(rate_cv) and rate_cv > 0.08:
        obs.append(Observation(
            code="RATE_VOL",
            title="Effective ₹/kWh swings month to month",
            detail=(
                f"Your effective rate varies about ±{rate_cv * 100:,.0f}% across the period. "
                "That can point to demand charges, a wrong tariff category, seasonal slabs, "
                "or billing errors — worth a line-by-line check of the bill components."
            ),
            confidence="Medium",
        ))

    # 3 · Consumption spike.
    if n >= 3:
        mean_kwh = float(kwh.mean())
        spike_idx = int(kwh.values.argmax())
        spike_kwh = float(kwh.iloc[spike_idx])
        if mean_kwh > 0 and spike_kwh > mean_kwh * 1.25:
            obs.append(Observation(
                code="SPIKE",
                title="A month stands out for consumption",
                detail=(
                    f"{work['month'].iloc[spike_idx]} used {spike_kwh:,.0f} kWh — about "
                    f"{(spike_kwh / mean_kwh - 1) * 100:,.0f}% above the period average. "
                    "Worth understanding why (a process change, a hot month, or a meter issue)."
                ),
                confidence="Medium",
            ))

    # 4 · Power factor (the most common, most confirmable quick win) — indicative ₹.
    if not np.isnan(avg_pf) and avg_pf < pf_target:
        points_below = (pf_target - avg_pf) * 100.0
        penalty_frac = min(points_below * pf_penalty_frac_per_point, PF_PENALTY_CAP_FRAC)
        indicative = avg_monthly_inr * penalty_frac * 12.0
        obs.append(Observation(
            code="PF",
            title="Low power factor — likely a penalty you can remove",
            detail=(
                f"Power factor averaged {avg_pf:.2f}, below {pf_target:.2f}. Most Indian utilities "
                "levy a PF penalty here, and capacitor correction is a common fast-payback fix. "
                f"Indicative only (assumes ~{pf_penalty_frac_per_point * 100:.1f}% of the bill per 0.01 "
                "below target) — confirm against the actual tariff."
            ),
            confidence="Medium",
            indicative_inr_per_year=indicative,
        ))
    elif np.isnan(avg_pf):
        obs.append(Observation(
            code="PF_UNKNOWN",
            title="Power factor not in the bills provided",
            detail=(
                "PF and any penalty/incentive aren't in this data. It's one of the fastest, "
                "most confirmable savings to check — ask for a bill that shows PF."
            ),
            confidence="Low",
        ))

    # 5 · Peak demand (qualitative — no invented number).
    if not np.isnan(peak_demand) and work["max_demand_kva"].notna().sum() >= 2:
        avg_demand = float(work["max_demand_kva"].astype(float).mean())
        if avg_demand > 0 and peak_demand > avg_demand * 1.3:
            obs.append(Observation(
                code="DEMAND",
                title="Peak demand spikes above the norm",
                detail=(
                    f"Recorded demand peaks at {peak_demand:,.0f} kVA vs an average of "
                    f"{avg_demand:,.0f} kVA. Demand charges are billed on the peak, so flattening "
                    "it (staggering loads, demand control) can cut the fixed part of the bill."
                ),
                confidence="Low",
            ))

    snap.observations = obs
    snap.questions = _questions(snap)
    return snap


def _questions(snap: BillSnapshot) -> list[str]:
    """Data-aware 'questions worth investigating' for the snapshot."""
    qs = [
        "Is the sanctioned demand and tariff category right for this load profile?",
        "What are the 2–3 biggest energy end-uses (motors, heating, compressed air, HVAC)?",
        "Is consumption metered only at the main meter, or are key areas sub-metered?",
    ]
    codes = {o.code for o in snap.observations}
    if "PF" in codes or "PF_UNKNOWN" in codes:
        qs.append("Is power-factor correction installed, sized correctly, and maintained?")
    if "SPIKE" in codes:
        qs.append(f"What changed in {snap.highest_month} or the highest-consumption month?")
    if "RATE_VOL" in codes:
        qs.append("Do the bill components (energy, demand, PF, taxes) reconcile month to month?")
    return qs[:5]


def render_snapshot_markdown(
    snap: BillSnapshot,
    *,
    company: str = "",
    date: str = "",
    prepared_by: str = "WattsWorth",
) -> str:
    """Render the snapshot as the filled WattsWorth Energy Snapshot (Markdown).

    Mirrors ``discovery-kit/energy-snapshot-template.md`` so the founder can send it
    directly. Disclaimers are preserved; no figure is invented.
    """
    lines: list[str] = []
    a = lines.append
    a("# WattsWorth Energy Snapshot")
    a("")
    a("**CONFIDENTIAL**")
    a("")
    a(f"- **Prepared for:** {company or '[Company Name]'}")
    a(f"- **Prepared by:** {prepared_by}")
    a(f"- **Date:** {date or '[Date]'}")
    a("")
    a("---")
    a("")
    a("## Purpose")
    a("")
    a("This is a high-level observational review of recent electricity consumption and billing patterns.")
    a("")
    a("- This is **not** an audit.")
    a("- This is **not** a guarantee of savings.")
    a("- The purpose is to identify areas that **may warrant further investigation.**")
    a("")
    a("---")
    a("")
    a("## Inputs Reviewed")
    a("")
    a(f"- Electricity bills reviewed: {snap.n_bills}")
    a(f"- Billing period: {snap.period_start} → {snap.period_end}")
    pf = "—" if np.isnan(snap.avg_power_factor) else f"{snap.avg_power_factor:.2f}"
    dem = "—" if np.isnan(snap.peak_demand_kva) else f"{snap.peak_demand_kva:,.0f} kVA"
    a(f"- Avg power factor: {pf}  ·  Peak demand: {dem}")
    a("")
    a("---")
    a("")
    a("## Snapshot Summary")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| **Total electricity spend** | ₹{snap.total_inr:,.0f} |")
    a(f"| **Average monthly spend** | ₹{snap.avg_monthly_inr:,.0f} |")
    rate = "—" if np.isnan(snap.effective_rate_inr_per_kwh) else f"₹{snap.effective_rate_inr_per_kwh:,.2f}/kWh"
    a(f"| **Effective rate** | {rate} |")
    a(f"| **Highest billing month** | {snap.highest_month} (₹{snap.highest_month_inr:,.0f}) |")
    a(f"| **Lowest billing month** | {snap.lowest_month} (₹{snap.lowest_month_inr:,.0f}) |")
    a("")
    a("---")
    a("")
    a("## Observations")
    a("")
    for i, o in enumerate(snap.observations, start=1):
        a(f"### Observation {i} — {o.title}")
        a(f"- **Description:** {o.detail}")
        if o.indicative_inr_per_year is not None:
            a(f"- **Indicative impact (confirm):** ~₹{o.indicative_inr_per_year:,.0f}/year")
        a(f"- **Confidence:** {o.confidence}")
        a("")
    a("---")
    a("")
    a("## Questions Worth Investigating")
    a("")
    for i, q in enumerate(snap.questions, start=1):
        a(f"{i}. {q}")
    a("")
    a("---")
    a("")
    a("## Suggested Next Step")
    a("")
    a("If useful, WattsWorth can conduct a deeper discussion to understand:")
    a("- Operational context")
    a("- Energy-intensive processes")
    a("- Existing monitoring methods")
    a("- Reporting workflows")
    a("")
    a("*No commitment required.*")
    a("")
    a("---")
    a("")
    a("## Important Note")
    a("")
    a("This snapshot is based **solely on the information provided** and should be treated as an "
      "**exploratory discussion document, not an engineering assessment.**")
    a("")
    return "\n".join(lines)
