"""Plain-English narrative for the readiness verdict (jargon-free)."""
from __future__ import annotations

from readiness.models import FAIL, MARGINAL, NOT_READY, READY, WARN, CheckResult

_HEADLINES = {
    READY: "READY — your data can support a defensible, certifiable M&V baseline.",
    MARGINAL: "MARGINAL — usable, but savings estimates will carry higher uncertainty and may not be auditor-certifiable.",
    NOT_READY: "NOT READY — fix the flagged issues before relying on any savings number.",
}


def headline(verdict: str) -> str:
    return _HEADLINES.get(verdict, verdict)


def summary(verdict: str, checks: list[CheckResult]) -> str:
    """One short paragraph: the verdict plus the top issues and their actions."""
    problems = [c for c in checks if c.status in (FAIL, WARN)]
    problems.sort(key=lambda c: 0 if c.status == FAIL else 1)
    if not problems:
        return ("All readiness checks passed. The dataset is suitable for ISO 50001 baselining "
                "and IPMVP Option C savings verification.")
    top = problems[:3]
    bullets = " ".join(f"{c.name}: {c.action}" for c in top)
    lead = {
        READY: "Minor items to note, but the data is ready.",
        MARGINAL: "The data is usable but borderline.",
        NOT_READY: "The data is not yet suitable for defensible M&V.",
    }.get(verdict, "")
    return f"{lead} Priority actions — {bullets}"
