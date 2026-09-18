"""Methodology identity for Stage 2's GlobalPlanningRun (Global Multi-Incident
Optimizer refactor).

"legacy_per_event_orchestration" is deliberately NOT a name that could be
mistaken for a global optimizer: this stage only sequences the existing,
unchanged per-FireEvent ResponsePlanningRefreshOrchestrator across all
active FireEvents inside one audit boundary. Bump METHODOLOGY_VERSION if
this orchestration's own bookkeeping behavior changes (ordering rule,
status-mapping, fingerprint composition); bump/replace METHODOLOGY itself
only when Stage 4 introduces the real global optimizer.
"""
from __future__ import annotations

METHODOLOGY = "legacy_per_event_orchestration"
METHODOLOGY_VERSION = "1.0"
