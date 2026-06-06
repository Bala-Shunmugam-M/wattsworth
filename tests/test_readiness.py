"""Tests for the M&V Readiness Assessment package (readiness/).

Written RED-first. Verdicts: READY / MARGINAL / NOT_READY. The decisive check is
baseline feasibility — we don't *predict* certifiability, we trial-fit and measure
it. The DAEWOO real-data fixture is a real-world oracle: it must NOT come back READY
(its documented OOS CV(RMSE) ~28% exceeds the IPMVP 20% gate).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import data
from readiness import CheckResult, ReadinessReport, assess_readiness

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DRIVERS = ["production_tonnes", "ambient_temp_c", "operating_hours"]


# --- Fixtures ----------------------------------------------------------------
def clean_ready_df() -> pd.DataFrame:
    """Clean, single-regime, anomaly-free daily data -> should be READY."""
    df = data.generate_synthetic_plant_data(540, 42)
    anomalies = set(df.attrs["anomaly_dates"])
    return df[(df["date"] <= pd.Timestamp("2025-08-31")) & (~df["date"].isin(anomalies))].reset_index(drop=True)


def daewoo_df() -> pd.DataFrame:
    """Real steel-plant daily aggregate (committed). OOS CV(RMSE) ~28% -> not certifiable."""
    return pd.read_csv(ROOT / "data" / "steel_industry_daily.csv", parse_dates=["date"])


def noise_df() -> pd.DataFrame:
    """Energy is pure noise unrelated to drivers -> baseline cannot certify -> NOT_READY."""
    df = clean_ready_df().copy()
    rng = np.random.default_rng(0)
    df["energy_kwh"] = rng.normal(10000.0, 5000.0, len(df)).clip(min=0)
    return df


def _assess(df: pd.DataFrame, drivers=None):
    return assess_readiness(df, target="energy_kwh", drivers=drivers or _DEFAULT_DRIVERS)


def _check(report: ReadinessReport, check_id: str) -> CheckResult:
    matches = [c for c in report.checks if c.id == check_id]
    assert matches, f"check '{check_id}' not present"
    return matches[0]


# --- A/B/C: the three headline verdicts --------------------------------------
def test_A_ready_dataset() -> None:
    r = _assess(clean_ready_df())
    assert r.verdict == "READY"
    assert r.likely_certifiable is True
    assert r.score >= 80
    assert r.expected_cv_rmse is not None and r.expected_cv_rmse < 0.20


def test_B_marginal_dataset_real_daewoo() -> None:
    r = _assess(daewoo_df(), drivers=["load_index", "weekend"])
    assert r.verdict == "MARGINAL"
    assert r.likely_certifiable is False  # the real-data oracle: not certifiable


def test_C_not_ready_dataset_noise() -> None:
    r = _assess(noise_df())
    assert r.verdict == "NOT_READY"
    assert r.likely_certifiable is False
    assert _check(r, "feasibility").status == "FAIL"


# --- D: broken datasets -> specific failures ---------------------------------
def test_D_missing_energy_column() -> None:
    df = clean_ready_df().drop(columns=["energy_kwh"])
    r = _assess(df)
    assert r.verdict == "NOT_READY"
    assert _check(r, "required_columns").status == "FAIL"


def test_D_all_null_energy() -> None:
    df = clean_ready_df()
    df["energy_kwh"] = np.nan
    r = _assess(df)
    assert r.verdict == "NOT_READY"
    assert _check(r, "missing_data").status == "FAIL"


def test_D_single_row() -> None:
    r = _assess(clean_ready_df().head(1))
    assert r.verdict == "NOT_READY"
    assert _check(r, "sample_size").status == "FAIL"


def test_D_empty_dataframe() -> None:
    r = _assess(pd.DataFrame(columns=["date", "energy_kwh", *_DEFAULT_DRIVERS]))
    assert r.verdict == "NOT_READY"
    assert _check(r, "sample_size").status == "FAIL"


def test_D_no_drivers() -> None:
    r = assess_readiness(clean_ready_df(), target="energy_kwh", drivers=[])
    assert r.verdict == "NOT_READY"
    assert _check(r, "required_columns").status == "FAIL"


# --- Coverage: individual checks ---------------------------------------------
def test_duplicate_dates_flagged() -> None:
    df = clean_ready_df()
    dup = pd.concat([df, df.head(20)], ignore_index=True)
    assert _check(_assess(dup), "duplicate_dates").status == "FAIL"


def test_missing_data_warn() -> None:
    df = clean_ready_df().copy()
    idx = df.sample(frac=0.08, random_state=1).index  # ~8% missing -> WARN band
    df.loc[idx, "ambient_temp_c"] = np.nan
    assert _check(_assess(df), "missing_data").status in ("WARN", "FAIL")


def test_date_gaps_flagged() -> None:
    df = clean_ready_df()
    df = df[~df["date"].between("2025-03-01", "2025-05-31")].reset_index(drop=True)  # big hole
    assert _check(_assess(df), "date_gaps").status in ("WARN", "FAIL")


def test_outliers_flagged() -> None:
    df = clean_ready_df().copy()
    df.loc[df.index[:30], "energy_kwh"] *= 4.0  # gross spikes
    assert _check(_assess(df), "outliers").status in ("WARN", "FAIL")


def test_low_variance_driver_flagged() -> None:
    df = clean_ready_df().copy()
    df["operating_hours"] = 22.5  # constant driver
    assert _check(_assess(df), "driver_variance").status in ("WARN", "FAIL")


def test_multicollinearity_preview() -> None:
    df = clean_ready_df().copy()
    df["dup_prod"] = df["production_tonnes"]  # perfect collinearity
    r = assess_readiness(df, target="energy_kwh", drivers=[*_DEFAULT_DRIVERS, "dup_prod"])
    assert _check(r, "multicollinearity").status in ("WARN", "FAIL")


def test_physical_sanity_negative_energy() -> None:
    df = clean_ready_df().copy()
    df.loc[df.index[:5], "energy_kwh"] = -100.0
    assert _check(_assess(df), "physical_sanity").status == "FAIL"


def test_sample_sufficiency_warn_band() -> None:
    r = _assess(clean_ready_df().head(25))  # below WARN threshold, above FAIL
    assert _check(r, "sample_size").status in ("WARN", "FAIL")


def test_feasibility_certifiable_on_clean() -> None:
    r = _assess(clean_ready_df())
    f = _check(r, "feasibility")
    assert f.status == "PASS"
    assert r.likely_certifiable is True


# --- Output contract / success criteria --------------------------------------
def test_report_contract() -> None:
    r = _assess(clean_ready_df())
    assert r.verdict in ("READY", "MARGINAL", "NOT_READY")
    assert 0 <= r.score <= 100
    assert isinstance(r.headline, str) and r.headline
    assert isinstance(r.summary, str) and r.summary
    assert isinstance(r.likely_certifiable, bool)


def test_every_check_has_plain_english() -> None:
    """Success criterion: understandable without statistical expertise."""
    for c in _assess(daewoo_df(), drivers=["load_index", "weekend"]).checks:
        assert c.detail and isinstance(c.detail, str)
        assert c.business_impact and isinstance(c.business_impact, str)
        assert c.action and isinstance(c.action, str)


def test_deterministic() -> None:
    df = clean_ready_df()
    a, b = _assess(df), _assess(df)
    assert a.verdict == b.verdict
    assert a.score == b.score
    assert [(c.id, c.status) for c in a.checks] == [(c.id, c.status) for c in b.checks]


# --- Purity ------------------------------------------------------------------
def test_readiness_imports_no_streamlit() -> None:
    code = (
        "import sys; import readiness, readiness.checks, readiness.feasibility, "
        "readiness.score, readiness.narrative, readiness.config; "
        "assert 'streamlit' not in sys.modules"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_readiness_sources_have_no_streamlit_import() -> None:
    for py in (ROOT / "readiness").glob("*.py"):
        assert "import streamlit" not in py.read_text(encoding="utf-8")
