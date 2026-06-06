"""Individual M&V readiness checks (pure, deterministic).

Each check returns a :class:`~readiness.models.CheckResult` carrying a plain-English
detail, a business impact, and a corrective action — so the output is intelligible
without statistical expertise. No Streamlit, no UI, no report coupling.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from readiness import config
from readiness.models import FAIL, PASS, WARN, CheckResult


def _cr(cid, name, status, value, threshold, detail, impact, action) -> CheckResult:
    return CheckResult(cid, name, status, value, threshold, detail, impact, action)


def _present(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def required_columns(df: pd.DataFrame, target: str, drivers: Sequence[str], date_col: str) -> CheckResult:
    if not drivers:
        return _cr("required_columns", "Required columns", FAIL, None, None,
                   "No explanatory (driver) variables were specified.",
                   "Without drivers, energy use cannot be normalised — no defensible baseline is possible.",
                   "Provide at least one production/weather/operating driver column.")
    missing = [c for c in [target, *drivers] if c not in df.columns]
    if missing:
        return _cr("required_columns", "Required columns", FAIL, None, None,
                   f"Required column(s) are absent: {', '.join(missing)}.",
                   "The model cannot run without the energy meter and its drivers.",
                   f"Add the missing column(s): {', '.join(missing)}.")
    return _cr("required_columns", "Required columns", PASS, None, None,
               "Energy and all driver columns are present.",
               "The dataset has the fields needed to build a baseline.", "No action needed.")


def sample_size(df: pd.DataFrame) -> CheckResult:
    n = int(len(df))
    if n < config.MIN_ROWS_FAIL:
        st = FAIL
    elif n < config.MIN_ROWS_WARN:
        st = WARN
    else:
        st = PASS
    detail = f"{n} observations available."
    impact = ("Too few points to fit and validate a baseline; any savings number would be unreliable."
              if st != PASS else "Enough history to fit and validate a baseline.")
    action = ("Collect more history (aim for at least a year of regular readings)."
              if st != PASS else "No action needed.")
    return _cr("sample_size", "Sample sufficiency", st, float(n), float(config.MIN_ROWS_WARN), detail, impact, action)


def missing_data(df: pd.DataFrame, target: str, drivers: Sequence[str]) -> CheckResult:
    cols = _present(df, [target, *drivers])
    if df.empty or not cols:
        return _cr("missing_data", "Missing data", PASS, None, None,
                   "Not enough data to assess missingness.", "—", "No action needed.")
    frac = max(float(df[c].isna().mean()) for c in cols)
    if frac > config.MISSING_FRAC_FAIL:
        st = FAIL
    elif frac > config.MISSING_FRAC_WARN:
        st = WARN
    else:
        st = PASS
    detail = f"Up to {frac * 100:.1f}% of values are missing in the energy/driver columns."
    impact = ("Large gaps bias the baseline and shrink the usable sample."
              if st != PASS else "Coverage is good across the key columns.")
    action = ("Recover or interpolate the missing readings, or drop affected periods."
              if st != PASS else "No action needed.")
    return _cr("missing_data", "Missing data", st, frac, config.MISSING_FRAC_FAIL, detail, impact, action)


def duplicate_dates(df: pd.DataFrame, date_col: str) -> CheckResult:
    if date_col not in df.columns or df.empty:
        return _cr("duplicate_dates", "Duplicate timestamps", PASS, None, None,
                   "No date column to check.", "—", "No action needed.")
    dups = int(pd.to_datetime(df[date_col], errors="coerce").duplicated().sum())
    st = FAIL if dups > 0 else PASS
    detail = f"{dups} duplicate timestamp(s) found." if dups else "All timestamps are unique."
    impact = ("Duplicate readings double-count energy and corrupt the baseline."
              if dups else "Each period is represented once.")
    action = "De-duplicate readings (keep the corrected value per period)." if dups else "No action needed."
    return _cr("duplicate_dates", "Duplicate timestamps", st, float(dups), 0.0, detail, impact, action)


def date_gaps(df: pd.DataFrame, date_col: str) -> CheckResult:
    if date_col not in df.columns or df.empty:
        return _cr("date_gaps", "Date gaps", PASS, None, None,
                   "No date column to check.", "—", "No action needed.")
    d = pd.to_datetime(df[date_col], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(d) < 3:
        return _cr("date_gaps", "Date gaps", PASS, None, None,
                   "Too few dates to assess gaps.", "—", "No action needed.")
    step = max(int(d.diff().dropna().dt.days.median()), 1)
    span = int((d.iloc[-1] - d.iloc[0]).days)
    expected = span // step + 1
    gap_frac = max(0.0, 1.0 - len(d) / expected) if expected > 0 else 0.0
    if gap_frac > config.GAP_FRAC_FAIL:
        st = FAIL
    elif gap_frac > config.GAP_FRAC_WARN:
        st = WARN
    else:
        st = PASS
    detail = f"About {gap_frac * 100:.0f}% of expected periods are missing from the calendar."
    impact = ("Long gaps mean the baseline misses operating conditions and seasons."
              if st != PASS else "The time series is largely continuous.")
    action = ("Fill the gaps with recovered data, or restrict the baseline to a continuous window."
              if st != PASS else "No action needed.")
    return _cr("date_gaps", "Date gaps", st, gap_frac, config.GAP_FRAC_FAIL, detail, impact, action)


def physical_sanity(df: pd.DataFrame) -> CheckResult:
    if df.empty:
        return _cr("physical_sanity", "Physical sanity", PASS, None, None,
                   "No data to check.", "—", "No action needed.")
    issues: list[str] = []
    if "energy_kwh" in df.columns and (pd.to_numeric(df["energy_kwh"], errors="coerce") < 0).any():
        issues.append("negative energy")
    if "production_tonnes" in df.columns and (pd.to_numeric(df["production_tonnes"], errors="coerce") < 0).any():
        issues.append("negative production")
    if "grid_pf" in df.columns:
        pf = pd.to_numeric(df["grid_pf"], errors="coerce")
        if ((pf < 0) | (pf > 1)).any():
            issues.append("power factor outside 0-1")
    if "operating_hours" in df.columns:
        h = pd.to_numeric(df["operating_hours"], errors="coerce")
        if ((h < 0) | (h > 24)).any():
            issues.append("operating hours outside 0-24")
    st = FAIL if issues else PASS
    detail = ("Impossible values found: " + ", ".join(issues) + "."
              if issues else "All values are within physically sensible ranges.")
    impact = ("Impossible readings indicate meter/sensor faults that will distort results."
              if issues else "No obviously faulty readings.")
    action = "Investigate the meters/sensors and correct or remove the faulty rows." if issues else "No action needed."
    return _cr("physical_sanity", "Physical sanity", st, float(len(issues)), 0.0, detail, impact, action)


def outliers(df: pd.DataFrame, target: str) -> CheckResult:
    if target not in df.columns or df.empty:
        return _cr("outliers", "Outliers", PASS, None, None,
                   "No energy column to check.", "—", "No action needed.")
    s = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(s) < 10:
        return _cr("outliers", "Outliers", PASS, None, None,
                   "Too few points to assess outliers.", "—", "No action needed.")
    med = float(s.median())
    mad = float((s - med).abs().median())
    scale = 1.4826 * mad if mad > 0 else (float(s.std(ddof=0)) or 1.0)
    frac = float((((s - med) / scale).abs() > config.OUTLIER_SIGMA).mean())
    if frac > config.OUTLIER_FRAC_FAIL:
        st = FAIL
    elif frac > config.OUTLIER_FRAC_WARN:
        st = WARN
    else:
        st = PASS
    detail = f"{frac * 100:.1f}% of energy readings are extreme outliers."
    impact = ("Outliers (spikes, meter errors, abnormal days) inflate baseline error."
              if st != PASS else "Energy readings are well behaved.")
    action = ("Review the flagged days; exclude genuine non-routine events from the baseline."
              if st != PASS else "No action needed.")
    return _cr("outliers", "Outliers", st, frac, config.OUTLIER_FRAC_FAIL, detail, impact, action)


def driver_variance(df: pd.DataFrame, drivers: Sequence[str]) -> CheckResult:
    cols = _present(df, drivers)
    if df.empty or not cols:
        return _cr("driver_variance", "Driver variance", PASS, None, None,
                   "No drivers to assess.", "—", "No action needed.")
    worst_name, worst_rel = None, None
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 2:
            continue
        rel = float(s.std(ddof=0) / (abs(float(s.mean())) + 1e-9))
        if worst_rel is None or rel < worst_rel:
            worst_name, worst_rel = c, rel
    if worst_rel is None:
        return _cr("driver_variance", "Driver variance", PASS, None, None,
                   "Drivers could not be assessed.", "—", "No action needed.")
    if worst_rel < config.DRIVER_REL_STD_FAIL:
        st = FAIL
    elif worst_rel < config.DRIVER_REL_STD_WARN:
        st = WARN
    else:
        st = PASS
    detail = f"Driver '{worst_name}' barely varies (relative variation {worst_rel * 100:.2f}%)."
    impact = ("A near-constant driver explains no energy variation — it cannot normalise the baseline."
              if st != PASS else "Drivers vary enough to explain energy use.")
    action = (f"Replace or supplement '{worst_name}' with a driver that actually changes with operations."
              if st != PASS else "No action needed.")
    return _cr("driver_variance", "Driver variance", st, worst_rel, config.DRIVER_REL_STD_WARN, detail, impact, action)


def _max_vif(df: pd.DataFrame, drivers: Sequence[str]) -> float | None:
    cols = _present(df, drivers)
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna() if cols else pd.DataFrame()
    if len(cols) < 2 or len(sub) < len(cols) + 2:
        return None
    design = sm.add_constant(sub, has_constant="add")
    matrix = design.to_numpy(dtype=float)
    vifs: list[float] = []
    # Perfect collinearity yields inf via a 1/(1-R^2)=1/0; that is the intended
    # "very high" signal, so silence the expected divide warning rather than spam CI.
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(1, matrix.shape[1]):
            try:
                vifs.append(float(variance_inflation_factor(matrix, i)))
            except Exception:  # noqa: BLE001
                continue
    return max(vifs) if vifs else None


def multicollinearity(df: pd.DataFrame, drivers: Sequence[str]) -> CheckResult:
    mv = _max_vif(df, drivers)
    if mv is None:
        return _cr("multicollinearity", "Multicollinearity", PASS, None, None,
                   "Not enough drivers/data to assess multicollinearity.", "—", "No action needed.")
    if mv > config.VIF_FAIL:
        st = FAIL
    elif mv > config.VIF_WARN:
        st = WARN
    else:
        st = PASS
    shown = "very high" if mv == float("inf") else f"{mv:.1f}"
    detail = f"Highest variance-inflation factor among drivers is {shown}."
    impact = ("Drivers overlap (collinear), so the model can't separate their effects — coefficients become unstable."
              if st != PASS else "Drivers are sufficiently independent.")
    action = ("Drop or combine redundant drivers (keep the most physically meaningful one)."
              if st != PASS else "No action needed.")
    return _cr("multicollinearity", "Multicollinearity", st,
               (None if mv == float("inf") else mv), config.VIF_WARN, detail, impact, action)
