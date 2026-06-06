"""Static chart rendering for reports (headless matplotlib -> PNG bytes)."""
from __future__ import annotations

from io import BytesIO

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: no display, safe on Streamlit Cloud / CI
from matplotlib.figure import Figure  # noqa: E402  (must follow use("Agg"))


def cusum_png(avoided_df: pd.DataFrame) -> bytes:
    """Render the cumulative-avoided-energy (CUSUM) chart as PNG bytes.

    Uses a bare ``Figure`` (not pyplot) to avoid global state and headless issues.
    """
    fig = Figure(figsize=(7.0, 3.0), dpi=120)
    ax = fig.subplots()
    mwh = avoided_df["cumulative_avoided_kwh"].to_numpy() / 1000.0
    dates = pd.to_datetime(avoided_df["date"])
    ax.fill_between(dates, mwh, color="#2E6F40", alpha=0.25)
    ax.plot(dates, mwh, color="#2E6F40", linewidth=1.8)
    ax.set_ylabel("Cumulative avoided (MWh)")
    ax.set_title("CUSUM — cumulative avoided energy")
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()
