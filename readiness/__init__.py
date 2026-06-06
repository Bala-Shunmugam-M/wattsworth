"""M&V Readiness Assessment — is this data fit for defensible baselining & M&V?

Pure-Python, UI-free, deterministic. Tells a non-statistician whether uploaded
data can support an ISO 50001 baseline and IPMVP Option C savings verification,
with a READY / MARGINAL / NOT_READY verdict, plain-English explanations, business
impact, corrective actions, and a measured certifiability estimate.
"""
from __future__ import annotations

from readiness.core import assess_readiness
from readiness.models import CheckResult, ReadinessReport

__all__ = ["assess_readiness", "ReadinessReport", "CheckResult"]
