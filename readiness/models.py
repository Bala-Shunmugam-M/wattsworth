"""Dataclasses for the M&V readiness assessment (no logic, no deps)."""
from __future__ import annotations

from dataclasses import dataclass, field

# Per-check statuses and overall verdicts (plain string constants for portability).
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

READY = "READY"
MARGINAL = "MARGINAL"
NOT_READY = "NOT_READY"


@dataclass
class CheckResult:
    """One readiness check, expressed for a non-statistician.

    Attributes:
        id: Stable machine id (e.g. ``"feasibility"``).
        name: Human-readable check name.
        status: ``PASS`` / ``WARN`` / ``FAIL``.
        value: The measured value (or None if not applicable).
        threshold: The threshold it was judged against (or None).
        detail: Plain-English statement of what was found.
        business_impact: Why it matters, in business terms.
        action: The corrective action to take.
    """

    id: str
    name: str
    status: str
    value: float | None
    threshold: float | None
    detail: str
    business_impact: str
    action: str


@dataclass
class ReadinessReport:
    """The overall M&V readiness verdict for a dataset.

    Attributes:
        verdict: ``READY`` / ``MARGINAL`` / ``NOT_READY``.
        score: 0-100 readiness score.
        likely_certifiable: Whether a savings claim would likely pass the
            IPMVP/ASHRAE CV(RMSE) gate.
        expected_r2: Trial-fit R² (or None if a baseline could not be fit).
        expected_cv_rmse: Trial-fit CV(RMSE), OOS-preferred (or None).
        headline: One-line executive verdict, jargon-free.
        summary: Short paragraph naming the top issues and next steps.
        checks: All individual :class:`CheckResult` items.
    """

    verdict: str
    score: float
    likely_certifiable: bool
    expected_r2: float | None
    expected_cv_rmse: float | None
    headline: str
    summary: str
    checks: list[CheckResult] = field(default_factory=list)
