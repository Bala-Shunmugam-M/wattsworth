"""Aggregate checks + feasibility into the overall readiness verdict."""
from __future__ import annotations

from typing import Optional

from readiness import config, narrative
from readiness.models import FAIL, MARGINAL, NOT_READY, PASS, READY, WARN, CheckResult, ReadinessReport


def aggregate(
    checks: list[CheckResult],
    model: Optional[object],
    feasibility: CheckResult,
) -> ReadinessReport:
    """Combine all checks into a deterministic verdict, score, and narrative."""
    n_fail = sum(1 for c in checks if c.status == FAIL)
    n_warn = sum(1 for c in checks if c.status == WARN)

    if n_fail > 0:
        verdict = NOT_READY
    elif n_warn > 0:
        verdict = MARGINAL
    else:
        verdict = READY

    score = max(0.0, min(100.0, 100.0 - config.SCORE_FAIL_PENALTY * n_fail - config.SCORE_WARN_PENALTY * n_warn))
    likely_certifiable = feasibility.status == PASS

    expected_r2 = float(getattr(model, "r_squared")) if model is not None else None
    expected_cv_rmse = float(feasibility.value) if feasibility.value is not None else None

    return ReadinessReport(
        verdict=verdict,
        score=score,
        likely_certifiable=likely_certifiable,
        expected_r2=expected_r2,
        expected_cv_rmse=expected_cv_rmse,
        headline=narrative.headline(verdict),
        summary=narrative.summary(verdict, checks),
        checks=checks,
    )
