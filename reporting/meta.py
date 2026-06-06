"""Report metadata (provenance the engine dataclasses do not carry).

An audit-grade report needs identity and provenance — facility, preparer, date
ranges, software version — that are not part of the analytics. ``generated_at`` is
injected (not ``datetime.now()``) so reports are reproducible and testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine import config


@dataclass
class ReportMeta:
    """Provenance metadata for an M&V report."""

    facility_name: str = "Demo Plant"
    report_id: str = "WW-MV-0001"
    prepared_by: str = "WattsWorth"
    baseline_start: str = ""
    baseline_end: str = ""
    reporting_start: str = ""
    reporting_end: str = ""
    tariff_inr_per_kwh: float = config.TARIFF_FLAT_INR_PER_KWH
    emission_factor: float = config.GRID_EMISSION_FACTOR_KG_PER_KWH
    generated_at: str = ""
    software_version: str = "WattsWorth 0.1.0"
    ipmvp_option: str = "Option C (Whole-Facility)"
    excluded_dates: tuple[str, ...] = field(default_factory=tuple)
