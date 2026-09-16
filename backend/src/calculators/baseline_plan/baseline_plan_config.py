"""Centralized EcoGuard V1 baseline-plan methodology constants.

These identify the deterministic baseline methodology for the eventual
Epic 5 `BaselinePlanResult` contract (frozen in Task 0). They are
centralized here, next to the pure calculator that implements this
methodology, the same way response-target methodology constants live next
to `ResponseTargetCalculator`.
"""
from __future__ import annotations

BASELINE_PLAN_METHODOLOGY = "GREEDY_NEAREST_AVAILABLE"
BASELINE_PLAN_METHODOLOGY_VERSION = "1.0"
