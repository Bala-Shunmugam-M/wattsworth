"""Pytest configuration for WattsWorth.

Adds the project root to ``sys.path`` so tests can ``from engine import ...``
without per-file path hacks, whether or not the package was installed with
``pip install -e .``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
