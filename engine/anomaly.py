"""Anomaly detection on energy signatures.

Two complementary detectors:

* :func:`detect_residual_anomalies` — univariate, baseline-driven. It looks at the
  residual (actual − baseline-expected), removes any slow drift with a rolling
  median (so a sustained regime change like an efficiency project is *not* flagged),
  and flags single days whose robust z-score exceeds ``sigma``. This catches energy
  *spikes* — the kind of abnormal heat or equipment fault an operator cares about.
* :func:`detect_multivariate_anomalies` — an IsolationForest over the joint
  operating signature (energy + drivers), catching days that are odd in combination
  even if no single variable is extreme.

Pure Python — no Streamlit imports.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from engine import config
from engine.baseline import predict_expected

if TYPE_CHECKING:
    from engine.baseline import BaselineModel

_MULTIVARIATE_FEATURES: tuple[str, ...] = (
    "sec_kwh_per_t",   # specific energy consumption — isolates spikes regardless of load
    "ambient_temp_c",
    "operating_hours",
    "grid_pf",
)


def _build_default_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the default multivariate feature matrix.

    Uses specific energy consumption (energy per tonne) rather than raw energy, so
    a spike stands out even on a low-production day. Pairs it with the ambient,
    runtime, and power-factor signature. Missing values are median-filled.
    """
    feat = pd.DataFrame(index=df.index)
    if "energy_kwh" in df.columns and "production_tonnes" in df.columns:
        prod = df["production_tonnes"].astype(float).replace(0, np.nan)
        feat["sec_kwh_per_t"] = df["energy_kwh"].astype(float) / prod
    for col in ("ambient_temp_c", "operating_hours", "grid_pf"):
        if col in df.columns:
            feat[col] = df[col].astype(float)
    if feat.empty:
        raise ValueError("No usable feature columns for multivariate detection.")
    return feat.fillna(feat.median(numeric_only=True))


def _robust_scale(values: np.ndarray) -> float:
    """Return a robust scale (1.4826·MAD), falling back to std, then 1.0."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad > 0:
        return 1.4826 * mad
    std = float(np.std(values))
    return std if std > 0 else 1.0


def detect_residual_anomalies(
    df: pd.DataFrame,
    model: "BaselineModel",
    sigma: float = config.ANOMALY_SIGMA,
    window: int = 15,
    target: str = config.TARGET_COLUMN,
) -> pd.DataFrame:
    """Flag single-day energy spikes via robust z-score of baseline residuals.

    The residual ``actual − expected`` removes the influence of the operating
    drivers. A centred rolling median then removes any slow drift (seasonal trend,
    or a step change from an efficiency project), leaving single-day spikes as
    large residual deviations. Days beyond ``±sigma`` robust standard deviations
    are flagged.

    Args:
        df: Plant-energy frame with ``target`` and the model's drivers (and
            ideally ``date``).
        model: A fitted :class:`~engine.baseline.BaselineModel` that explains ``df``.
        sigma: Control-limit width in robust standard deviations.
        window: Rolling-median window (days) used to remove slow drift.
        target: Name of the actual-energy column.

    Returns:
        DataFrame with ``date`` (if present), ``actual_kwh``, ``expected_kwh``,
        ``residual``, ``robust_z``, ``is_anomaly`` (bool), ``direction``
        ('high'/'low').

    Raises:
        ValueError: If the model is unfitted or ``target`` is missing.
    """
    if model.model is None:
        raise ValueError("BaselineModel is not fitted (model is None).")
    if target not in df.columns:
        raise ValueError(f"Missing target column '{target}'.")

    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"])

    expected = predict_expected(model, work)
    actual = work[target].astype(float)
    residual = actual - expected

    trend = residual.rolling(window=window, center=True, min_periods=1).median()
    detrended = (residual - trend).to_numpy(dtype=float)

    med = float(np.median(detrended))
    scale = _robust_scale(detrended)
    robust_z = (detrended - med) / scale

    out = pd.DataFrame(index=work.index)
    if "date" in work.columns:
        out["date"] = work["date"].to_numpy()
    out["actual_kwh"] = actual.to_numpy()
    out["expected_kwh"] = expected.to_numpy()
    out["residual"] = residual.to_numpy()
    out["robust_z"] = robust_z
    out["is_anomaly"] = np.abs(robust_z) > sigma
    out["direction"] = np.where(robust_z > 0, "high", "low")
    return out.reset_index(drop=True)


def detect_multivariate_anomalies(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """Flag multivariate outliers in the operating signature via IsolationForest.

    Standardises the chosen features and fits an IsolationForest, which isolates
    points that are unusual in the joint feature space — e.g. high energy for the
    given production and weather.

    Args:
        df: Plant-energy frame.
        features: Columns to use; defaults to energy + drivers + grid power factor
            (those present in ``df``).
        contamination: Expected fraction of anomalies (IsolationForest parameter).
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with ``date`` (if present), ``anomaly_score`` (higher = more
        anomalous), and ``is_anomaly`` (bool).

    Raises:
        ValueError: If no usable feature columns are available.
    """
    if features is None:
        feature_df = _build_default_features(df)
    else:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing feature column(s): {missing}")
        feature_df = df[list(features)].astype(float)

    matrix = feature_df.to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(matrix)

    forest = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = forest.fit_predict(scaled)

    out = pd.DataFrame(index=df.index)
    if "date" in df.columns:
        out["date"] = pd.to_datetime(df["date"]).to_numpy()
    out["anomaly_score"] = -forest.score_samples(scaled)  # higher = more anomalous
    out["is_anomaly"] = predictions == -1
    return out.reset_index(drop=True)
