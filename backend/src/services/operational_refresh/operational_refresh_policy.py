"""Central pure policy for operational refresh trigger decisions.

Terminology:
- Trigger: a domain-level reason to inspect current operational state.
- Reevaluation: prepare/inspect current production US 4.2 spread input.
- Effective-state change: calculator-relevant values differ.
- Recalculation: a later task invokes FireSpreadPredictionAgent.
- No-op: reevaluation happened, but effective state did not change.
- Resource refresh: a separate operational branch that does not invoke
  environmental analysis.
"""
from __future__ import annotations

from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType

SPREAD_REEVALUATION_TRIGGER_TYPES = frozenset(
    {
        OperationalRefreshTriggerType.WEATHER_UPDATE,
        OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        OperationalRefreshTriggerType.SEVERITY_UPDATE,
    }
)


def requires_spread_reevaluation(trigger_type: OperationalRefreshTriggerType) -> bool:
    """Return whether a trigger should inspect current Fire Spread input state."""
    if not isinstance(trigger_type, OperationalRefreshTriggerType):
        raise ValueError(f"trigger_type must be an OperationalRefreshTriggerType, got {trigger_type!r}")
    return trigger_type in SPREAD_REEVALUATION_TRIGGER_TYPES
