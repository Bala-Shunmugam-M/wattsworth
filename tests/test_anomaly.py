"""Tests for anomaly detection (engine/anomaly.py).

Confirms both detectors find the 5 injected spike days, that the residual
detector does not false-flag the post-intervention regime, and that error
handling is sound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import anomaly, baseline, data  # noqa: E402


@pytest.fixture(scope="module")
def plant() -> pd.DataFrame:
    return data.generate_synthetic_plant_data(days=540, seed=42)


@pytest.fixture(scope="module")
def anomalies(plant: pd.DataFrame) -> set:
    return set(plant.attrs["anomaly_dates"])


@pytest.fixture(scope="module")
def model(plant: pd.DataFrame) -> baseline.BaselineModel:
    """Baseline fit on the FULL series (so residual detection sees every day)."""
    return baseline.fit_baseline(plant)


# --- Residual (univariate) detector ------------------------------------------
def test_residual_detector_columns(plant: pd.DataFrame, model: baseline.BaselineModel) -> None:
    res = anomaly.detect_residual_anomalies(plant, model)
    for col in ("date", "actual_kwh", "expected_kwh", "residual", "robust_z", "is_anomaly", "direction"):
        assert col in res.columns
    assert res["is_anomaly"].dtype == bool


def test_residual_detector_finds_all_injected(
    plant: pd.DataFrame, model: baseline.BaselineModel, anomalies: set
) -> None:
    res = anomaly.detect_residual_anomalies(plant, model)
    flagged = set(res.loc[res["is_anomaly"], "date"])
    missing = anomalies - flagged
    assert not missing, f"Missed injected anomalies: {sorted(missing)}"


def test_residual_detector_few_false_positives(
    plant: pd.DataFrame, model: baseline.BaselineModel
) -> None:
    res = anomaly.detect_residual_anomalies(plant, model)
    # Should not drown in false positives, and injected spikes are 'high'.
    assert res["is_anomaly"].sum() <= 12
    highs = res.loc[res["is_anomaly"], "direction"]
    assert (highs == "high").sum() >= 5


def test_residual_detector_ignores_regime_shift(
    plant: pd.DataFrame, model: baseline.BaselineModel
) -> None:
    """The 6% post-intervention regime must NOT be flagged wholesale."""
    res = anomaly.detect_residual_anomalies(plant, model)
    post = res[res["date"] >= pd.Timestamp("2025-09-01")]
    # Far fewer than half the post-period days should be anomalies.
    assert post["is_anomaly"].mean() < 0.1


# --- Multivariate detector ---------------------------------------------------
def test_multivariate_detector_finds_most_injected(plant: pd.DataFrame, anomalies: set) -> None:
    res = anomaly.detect_multivariate_anomalies(plant, contamination=0.03)
    flagged = set(res.loc[res["is_anomaly"], "date"])
    hits = len(anomalies & flagged)
    assert hits >= 4, f"IsolationForest only caught {hits}/5 injected anomalies."


def test_multivariate_scores_present(plant: pd.DataFrame) -> None:
    res = anomaly.detect_multivariate_anomalies(plant)
    assert "anomaly_score" in res.columns
    assert "is_anomaly" in res.columns
    assert res["is_anomaly"].any()


# --- Error handling ----------------------------------------------------------
def test_unfitted_model_raises(plant: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not fitted"):
        anomaly.detect_residual_anomalies(plant, baseline.BaselineModel())


def test_multivariate_missing_feature_raises(plant: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing feature"):
        anomaly.detect_multivariate_anomalies(plant, features=["energy_kwh", "nonexistent"])
