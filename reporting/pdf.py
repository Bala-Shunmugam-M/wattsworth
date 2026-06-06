"""PDF M&V report (board-ready, audit-grade) via reportlab.

Currency is rendered as 'INR' (ASCII) in the PDF body because reportlab's default
Helvetica has no rupee glyph; the ₹ symbol is used in the Excel report and the UI.
"""
from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from reporting import charts, format as fmt, verdicts
from reporting.meta import ReportMeta

if TYPE_CHECKING:
    from engine.baseline import BaselineModel
    from engine.mv import SavingsSummary

_TABLE_STYLE = TableStyle(
    [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2EFE9")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
)


def build_mv_report_pdf(
    summary: "SavingsSummary",
    model: "BaselineModel",
    avoided_df: pd.DataFrame,
    meta: ReportMeta,
) -> bytes:
    """Build the PDF M&V report and return the file bytes.

    Raises:
        ValueError: If ``avoided_df`` is empty.
    """
    if avoided_df.empty:
        raise ValueError("Cannot build report: avoided_df is empty.")

    styles = getSampleStyleSheet()
    body = styles["Normal"]
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, title="WattsWorth M&V Savings Report",
        author=meta.prepared_by, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
    )
    s: list = []

    # 1. Cover / metadata
    s.append(Paragraph("WattsWorth &mdash; M&amp;V Savings Report", styles["Title"]))
    s.append(Paragraph(f"Facility: <b>{meta.facility_name}</b> &middot; Report {meta.report_id}", body))
    s.append(Paragraph(
        f"IPMVP {meta.ipmvp_option} &middot; ISO 50001 baseline &middot; {meta.software_version}", body))
    s.append(Paragraph(f"Prepared by {meta.prepared_by} &middot; Generated {meta.generated_at}", body))
    s.append(Spacer(1, 10))

    # 2. Baseline model & fit
    passed, verdict_text = verdicts.ashrae_verdict(model)
    s.append(Paragraph("1. Baseline model &amp; fit quality", styles["Heading2"]))
    s.append(Paragraph(f"Period: {model.baseline_period} &middot; {model.n_obs} observations", body))
    s.append(Paragraph(fmt.coefficient_equation(model.coefficients), styles["Code"]))
    s.append(Spacer(1, 4))
    fit_rows = [
        ["R-squared", f"{model.r_squared:.3f}"],
        ["CV(RMSE) in-sample", fmt.pct(model.cv_rmse)],
        ["CV(RMSE) out-of-sample", "n/a" if model.cv_rmse_oos is None else fmt.pct(model.cv_rmse_oos)],
        ["Durbin-Watson", f"{model.durbin_watson:.2f}"],
        ["Max VIF", fmt.na_or(model.max_vif, "{:.1f}")],
        ["Breusch-Pagan p", fmt.na_or(model.bp_pvalue, "{:.3f}")],
    ]
    t = Table(fit_rows, colWidths=[60 * mm, 40 * mm])
    t.setStyle(_TABLE_STYLE)
    s.append(t)
    s.append(Spacer(1, 4))
    s.append(Paragraph(f"<b>Acceptance (ASHRAE Guideline 14):</b> {verdict_text}", body))
    s.append(Spacer(1, 10))

    # 3. Verified savings
    s.append(Paragraph("2. Verified savings (reporting period)", styles["Heading2"]))
    s.append(Paragraph(
        f"Avoided energy: <b>{fmt.mwh(summary.avoided_kwh)}</b> ({fmt.pct(summary.pct_saving)}) "
        f"over {summary.days} days.", body))
    s.append(Paragraph(
        f"Cost saved: <b>{fmt.inr_lakhs(summary.inr_saved, ascii_only=True)}</b> &middot; "
        f"CO2 avoided: <b>{fmt.tonnes(summary.tonnes_co2_avoided)}</b>.", body))
    if summary.avoided_kwh_ci_low == summary.avoided_kwh_ci_low:  # not NaN
        s.append(Paragraph(
            f"{summary.ci_confidence * 100:.0f}% confidence interval on avoided energy: "
            f"[{summary.avoided_kwh_ci_low:,.0f} &ndash; {summary.avoided_kwh_ci_high:,.0f}] kWh.", body))
    s.append(Paragraph(f"Annualised: {fmt.mwh(summary.annual_kwh)}/yr &middot; "
                       f"{fmt.inr_lakhs(summary.annual_inr, ascii_only=True)}/yr &middot; "
                       f"{fmt.tonnes(summary.annual_tco2)}/yr.", body))
    s.append(Spacer(1, 4))
    s.append(Paragraph(f"<b>Significance:</b> {verdicts.significance_sentence(summary)}", body))
    s.append(Spacer(1, 10))

    # 4. CUSUM chart
    s.append(Paragraph("3. Cumulative avoided energy (CUSUM)", styles["Heading2"]))
    s.append(Image(BytesIO(charts.cusum_png(avoided_df)), width=165 * mm, height=70 * mm))
    s.append(Spacer(1, 8))

    # 5. Methodology
    s.append(Paragraph("4. Methodology &amp; assumptions", styles["Heading2"]))
    s.append(Paragraph(
        f"Tariff INR {meta.tariff_inr_per_kwh}/kWh; grid emission factor {meta.emission_factor} "
        "kgCO2/kWh. Baseline: OLS with backward elimination. M&amp;V: IPMVP Option C "
        "(baseline-expected minus actual). Significance corrected for serial correlation "
        "via an effective sample size. Annualisation: daily mean &times; 365.", body))
    s.append(Spacer(1, 8))

    # 6. Limitations
    s.append(Paragraph("5. Limitations", styles["Heading2"]))
    lim = ("This is a statistical M&amp;V baseline, not a metered sub-account. Figures depend on "
           "the chosen drivers and baseline window.")
    if not passed:
        lim += (" <b>NOTE: the baseline does not meet the ASHRAE Guideline 14 CV(RMSE) threshold &mdash; "
                "the savings figure is indicative, not certifiable, until a better-fitting baseline is established.</b>")
    s.append(Paragraph(lim, body))

    doc.build(s)
    return buf.getvalue()
