"""ISO 50001 energy baseline — the regression spine of WattsWorth.

Fits an OLS model of energy on operational drivers, yielding an expected-energy
predictor that M&V, anomaly detection, and optimisation all build on.

The baseline answers the core ISO 50001 question: *given how hard the plant was
working (production, weather, runtime), how much energy should it have used?*
Everything downstream is a deviation from that expectation:

* **M&V** — actual minus expected, after an efficiency project, is the saving.
* **Anomaly detection** — a day far from expected is an abnormal energy signature.
* **Optimisation** — changing the drivers changes the expectation.

Pure Python — no Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from engine import config


@dataclass
class BaselineModel:
    """Result of fitting an ISO 50001 baseline regression.

    Attributes:
        drivers: Driver columns retained after significance filtering.
        coefficients: Mapping of term -> fitted coefficient (includes ``const``).
        r_squared: Coefficient of determination on the baseline period.
        cv_rmse: Coefficient of variation of RMSE — ``RMSE / mean(actual)``. IPMVP
            generally expects this below ~0.20 for a usable baseline.
        n_obs: Number of observations used to fit.
        dropped: Candidate drivers removed for being statistically insignificant.
        model: Underlying fitted ``statsmodels`` results object (opaque to callers;
            used by :func:`predict_expected`).
    """

    drivers: list[str] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    cv_rmse: float = 0.0
    n_obs: int = 0
    dropped: list[str] = field(default_factory=list)
    model: object | None = None


def _design_matrix(df: pd.DataFrame, drivers: Sequence[str]) -> pd.DataFrame:
    """Build the OLS design matrix: a constant column followed by the drivers.

    Column order is fixed (``const`` first) so that fitting and prediction always
    agree on layout.

    Args:
        df: Source frame containing every column in ``drivers``.
        drivers: Driver column names, in order.

    Returns:
        DataFrame with columns ``["const", *drivers]`` as floats.
    """
    x = df.loc[:, list(drivers)].astype(float)
    x = sm.add_constant(x, has_constant="add")
    return x[["const", *drivers]]


def _filter_baseline_period(
    df: pd.DataFrame,
    baseline_period: Optional[tuple[str, str]],
) -> pd.DataFrame:
    """Restrict ``df`` to ``baseline_period`` (inclusive) on its ``date`` column.

    Args:
        df: Source frame; must contain ``date`` if a period is given.
        baseline_period: ``(start, end)`` ISO date strings, or ``None`` for all rows.

    Returns:
        The filtered (or unchanged) frame.

    Raises:
        ValueError: If a period is requested but ``date`` is missing, the dates
            are unparseable, or the filter leaves no rows.
    """
    if baseline_period is None:
        return df

    if "date" not in df.columns:
        raise ValueError("baseline_period was given but the frame has no 'date' column.")

    try:
        start, end = pd.Timestamp(baseline_period[0]), pd.Timestamp(baseline_period[1])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unparseable baseline_period {baseline_period!r}: {exc}") from exc

    dates = pd.to_datetime(df["date"])
    mask = (dates >= start) & (dates <= end)
    filtered = df.loc[mask]
    if filtered.empty:
        raise ValueError(
            f"baseline_period {baseline_period} selected 0 rows "
            f"(data spans {dates.min().date()}..{dates.max().date()})."
        )
    return filtered


def fit_baseline(
    df: pd.DataFrame,
    target: str = config.TARGET_COLUMN,
    drivers: Sequence[str] = config.DEFAULT_DRIVERS,
    baseline_period: Optional[tuple[str, str]] = None,
    significance: float = 0.05,
) -> BaselineModel:
    """Fit an OLS energy baseline, dropping insignificant drivers.

    Performs backward elimination: repeatedly removes the least-significant driver
    while its p-value exceeds ``significance``, leaving a parsimonious model in
    which every retained driver matters. Reports in-sample R² and CV(RMSE).

    Args:
        df: Plant-energy frame including ``target`` and ``drivers`` (and ``date``
            if ``baseline_period`` is used).
        target: Name of the energy column to model.
        drivers: Candidate independent variables.
        baseline_period: Optional ``(start, end)`` ISO dates to restrict the fit;
            ``None`` uses all rows.
        significance: Drivers with p-value above this are dropped (backward
            elimination). At least one driver is always retained.

    Returns:
        A fully populated :class:`BaselineModel`.

    Raises:
        ValueError: If ``target`` or any driver is missing, no drivers are given,
            or there are too few rows to fit reliably.
    """
    if not drivers:
        raise ValueError("At least one candidate driver is required.")

    fit_df = _filter_baseline_period(df, baseline_period)

    missing = [c for c in (target, *drivers) if c not in fit_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    work = fit_df[[target, *drivers]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < len(drivers) + 2:
        raise ValueError(
            f"Too few usable rows ({len(work)}) to fit {len(drivers)} driver(s)."
        )

    y = work[target].astype(float)
    retained: list[str] = list(drivers)
    dropped: list[str] = []

    # Backward elimination on driver p-values (the constant is never dropped).
    while True:
        results = sm.OLS(y, _design_matrix(work, retained)).fit()
        if len(retained) <= 1:
            break
        driver_pvalues = results.pvalues.drop("const", errors="ignore")
        worst_driver = str(driver_pvalues.idxmax())
        worst_p = float(driver_pvalues.max())
        if worst_p > significance:
            retained.remove(worst_driver)
            dropped.append(worst_driver)
            continue
        break

    # In-sample fit quality.
    rmse = float(np.sqrt(np.mean(np.square(results.resid))))
    mean_actual = float(y.mean())
    cv_rmse = rmse / mean_actual if mean_actual else float("inf")

    return BaselineModel(
        drivers=retained,
        coefficients={k: float(v) for k, v in results.params.items()},
        r_squared=float(results.rsquared),
        cv_rmse=cv_rmse,
        n_obs=int(results.nobs),
        dropped=dropped,
        model=results,
    )


def predict_expected(model: BaselineModel, df: pd.DataFrame) -> pd.Series:
    """Return expected energy for each row of ``df`` under ``model``.

    Args:
        model: A fitted :class:`BaselineModel`.
        df: Frame containing the model's retained driver columns.

    Returns:
        Series of expected energy aligned to ``df.index``, named
        ``"expected_energy_kwh"``.

    Raises:
        ValueError: If the model is unfitted or required driver columns are absent.
    """
    if model.model is None:
        raise ValueError("BaselineModel is not fitted (model is None).")

    missing = [c for c in model.drivers if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot predict — missing driver column(s): {missing}")

    design = _design_matrix(df, model.drivers)
    predicted = np.asarray(model.model.predict(design), dtype=float)
    return pd.Series(predicted, index=df.index, name="expected_energy_kwh")


def specific_energy_consumption(
    df: pd.DataFrame,
    energy_col: str = config.TARGET_COLUMN,
    production_col: str = "production_tonnes",
) -> pd.Series:
    """Return Specific Energy Consumption (energy per tonne) per row."""
    return df[energy_col] / df[production_col].replace(0, pd.NA)


def enpi(actual: pd.Series, expected: pd.Series) -> pd.Series:
    """Return the Energy Performance Indicator (actual / expected)."""
    return actual / expected.replace(0, pd.NA)
