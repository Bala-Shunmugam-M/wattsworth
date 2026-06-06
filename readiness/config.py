"""Thresholds for M&V readiness checks.

All judgment thresholds live here (auditability: one place, sourced, tunable).
Defaults are conservative and grounded in IPMVP / ASHRAE Guideline 14 practice.
"""
from __future__ import annotations

from engine import config as _engine_config

TARGET: str = _engine_config.TARGET_COLUMN
DEFAULT_DRIVERS: tuple[str, ...] = _engine_config.DEFAULT_DRIVERS

# Sample sufficiency (rows). M&V wants enough baseline points to fit + validate.
MIN_ROWS_FAIL: int = 15
MIN_ROWS_WARN: int = 30

# Missing data, as the worst fraction across the target + driver columns.
MISSING_FRAC_WARN: float = 0.05
MISSING_FRAC_FAIL: float = 0.20

# Calendar gaps, as the fraction of expected periods that are absent.
GAP_FRAC_WARN: float = 0.10
GAP_FRAC_FAIL: float = 0.30

# Target outliers via robust (MAD) z-score.
OUTLIER_SIGMA: float = 5.0
OUTLIER_FRAC_WARN: float = 0.02
OUTLIER_FRAC_FAIL: float = 0.10

# Driver variance: relative std (std / |mean|). A near-constant driver explains nothing.
DRIVER_REL_STD_WARN: float = 0.02
DRIVER_REL_STD_FAIL: float = 0.001

# Multicollinearity (max VIF across offered drivers).
VIF_WARN: float = 10.0
VIF_FAIL: float = 30.0

# Baseline feasibility: CV(RMSE) acceptance (OOS-preferred). Gate from the engine.
CVRMSE_GATE: float = _engine_config.CVRMSE_MAX     # 0.20 — certifiable at/below this
CVRMSE_WARN: float = 0.30                          # marginal between gate and this

# Scoring penalties.
SCORE_WARN_PENALTY: float = 8.0
SCORE_FAIL_PENALTY: float = 25.0
