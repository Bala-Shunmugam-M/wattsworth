"""Baseline feasibility — the decisive check.

We do not *predict* whether a certifiable baseline is possible; we trial-fit one
with the real engine and measure the CV(RMSE) against the IPMVP/ASHRAE gate. This
is what makes the readiness verdict trustworthy rather than a surface profile.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from engine import baseline as _baseline
from readiness import config
from readiness.models import FAIL, PASS, WARN, CheckResult


def trial_fit(
    df: pd.DataFrame,
    target: str,
    drivers: Sequence[str],
) -> tuple[Optional["_baseline.BaselineModel"], CheckResult]:
    """Fit a trial baseline and judge certifiability.

    Returns ``(model_or_None, feasibility_CheckResult)``. CV(RMSE) is OOS-preferred
    (the honest predictive metric); in-sample is used only when no hold-out exists.
    """
    try:
        # Degenerate inputs (a constant driver -> singular design) make statsmodels
        # emit benign divide warnings while still returning a usable fit; silence them.
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")
            model = _baseline.fit_baseline(df, target=target, drivers=list(drivers), validate_oos=True)
    except Exception as exc:  # noqa: BLE001 — any fit failure means "not feasible"
        return None, CheckResult(
            "feasibility", "Baseline feasibility", FAIL, None, config.CVRMSE_GATE,
            f"A baseline could not be fitted ({type(exc).__name__}).",
            "Without a baseline there is nothing to verify savings against.",
            "Fix the upstream data issues (columns, sample size, variance) and retry.",
        )

    cv = model.cv_rmse_oos if model.cv_rmse_oos is not None else model.cv_rmse
    which = "out-of-sample" if model.cv_rmse_oos is not None else "in-sample"
    if cv <= config.CVRMSE_GATE:
        st = PASS
    elif cv <= config.CVRMSE_WARN:
        st = WARN
    else:
        st = FAIL
    detail = (
        f"A trial baseline fits with R²={model.r_squared:.2f} and {which} "
        f"CV(RMSE)={cv * 100:.0f}% (IPMVP/ASHRAE gate is {config.CVRMSE_GATE * 100:.0f}%)."
    )
    if st == PASS:
        impact = "A reliable baseline can be built and savings can be certified with confidence."
        action = "Proceed to baseline and M&V."
    elif st == WARN:
        impact = "A baseline fits but is borderline — savings will carry higher uncertainty and may not be auditor-certifiable."
        action = "Add stronger drivers (production grade, weather) or a cleaner baseline window to tighten the fit."
    else:
        impact = "The drivers do not explain energy well enough to certify any savings number."
        action = "Add explanatory drivers that actually drive consumption, or collect better data, before claiming savings."
    return model, CheckResult("feasibility", "Baseline feasibility", st, float(cv), config.CVRMSE_GATE,
                              detail, impact, action)
