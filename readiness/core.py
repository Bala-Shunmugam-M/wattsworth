"""Orchestrator: assess_readiness() runs all checks and aggregates the verdict."""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from readiness import checks as _checks
from readiness import config, score
from readiness.models import ReadinessReport


def assess_readiness(
    df: pd.DataFrame,
    target: str = config.TARGET,
    drivers: Optional[Sequence[str]] = None,
    date_col: str = "date",
) -> ReadinessReport:
    """Assess whether ``df`` is suitable for defensible energy baselining and M&V.

    Runs the data-quality and energy-specific checks, then trial-fits a baseline to
    measure certifiability, and returns a :class:`~readiness.models.ReadinessReport`
    with a READY / MARGINAL / NOT_READY verdict understandable without statistics.

    Args:
        df: The uploaded plant-energy frame.
        target: Energy column name.
        drivers: Explanatory variables; defaults to the engine's default drivers.
        date_col: Timestamp column name.

    Returns:
        A populated :class:`ReadinessReport`.
    """
    drivers = list(drivers) if drivers is not None else list(config.DEFAULT_DRIVERS)

    results = [
        _checks.required_columns(df, target, drivers, date_col),
        _checks.sample_size(df),
        _checks.missing_data(df, target, drivers),
        _checks.duplicate_dates(df, date_col),
        _checks.date_gaps(df, date_col),
        _checks.physical_sanity(df),
        _checks.outliers(df, target),
        _checks.driver_variance(df, drivers),
        _checks.multicollinearity(df, drivers),
    ]

    from readiness import feasibility as _feasibility  # local import keeps module load light

    model, feasibility_result = _feasibility.trial_fit(df, target, drivers)
    results.append(feasibility_result)

    return score.aggregate(results, model, feasibility_result)
