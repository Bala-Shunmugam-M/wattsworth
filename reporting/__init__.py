"""Auditable M&V report generation (PDF + Excel).

Pure-Python, UI-free presentation layer that turns the engine's M&V outputs
(``SavingsSummary`` + ``BaselineModel`` + the per-day ``avoided`` frame) into an
audit-grade, board-ready report. Like ``engine/``, this package imports no
Streamlit and performs no analytics — it only formats existing results.
"""
from __future__ import annotations

from reporting.pdf import build_mv_report_pdf
from reporting.xlsx import build_mv_report_xlsx

__all__ = ["build_mv_report_pdf", "build_mv_report_xlsx"]
