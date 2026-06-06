"""Audit verdict logic (pure): ASHRAE-14 acceptance and significance wording."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from engine import config

if TYPE_CHECKING:
    from engine.baseline import BaselineModel
    from engine.mv import SavingsSummary


def ashrae_cvrmse_used(model: "BaselineModel") -> tuple[float, str]:
    """Return the CV(RMSE) to judge against and which one it is.

    Prefers the out-of-sample value (the honest predictive metric); falls back to
    in-sample when no hold-out was computed.
    """
    if model.cv_rmse_oos is not None:
        return float(model.cv_rmse_oos), "out-of-sample"
    return float(model.cv_rmse), "in-sample"


def ashrae_verdict(model: "BaselineModel", threshold: float = config.CVRMSE_MAX) -> tuple[bool, str]:
    """Return (passed, human text) for the ASHRAE Guideline 14 CV(RMSE) gate."""
    value, which = ashrae_cvrmse_used(model)
    passed = value <= threshold
    label = "PASS" if passed else "FAIL"
    return passed, (
        f"{label} — {which} CV(RMSE) {value * 100:.1f}% vs the {threshold * 100:.0f}% "
        f"ASHRAE Guideline 14 acceptance threshold."
    )


def significance_sentence(summary: "SavingsSummary") -> str:
    """Plain-English significance verdict reflecting ``is_significant``."""
    if summary.p_value is None or (isinstance(summary.p_value, float) and math.isnan(summary.p_value)):
        return (
            "Statistical significance could not be tested (insufficient reporting-period "
            "data). Treat the saving with caution."
        )
    detail = (
        f"autocorrelation-corrected one-sided t-test: t={summary.t_stat:.1f}, "
        f"p={summary.p_value:.1e}, rho={summary.lag1_autocorr:.2f}, "
        f"effective N={summary.n_effective:.0f} of {summary.days} days"
    )
    if summary.is_significant:
        return (
            f"The saving is statistically significant ({detail}). It is a real reduction, "
            "not noise — and the test corrects for serial correlation, which most M&V tools ignore."
        )
    return (
        f"The saving is NOT statistically significant ({detail}). Treat with caution — "
        "it may be within baseline noise."
    )
