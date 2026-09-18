"""Provenance of one GlobalIncidentDemand record (Stage 5 of the Global
Multi-Incident Optimizer refactor, Task 3).
"""
from __future__ import annotations

from enum import Enum


class DemandSource(Enum):
    """How an incident's resource demand was determined."""

    SEVERITY_ASSESSMENT = "severity_assessment"
    INSUFFICIENT_SEVERITY = "insufficient_severity"
