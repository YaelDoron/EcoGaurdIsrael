"""Tests for central operational refresh policy."""
from __future__ import annotations

import pytest

from src.models import OperationalRefreshTriggerType
from src.services.operational_refresh import (
    SPREAD_REEVALUATION_TRIGGER_TYPES,
    requires_spread_reevaluation,
)


@pytest.mark.parametrize(
    "trigger_type",
    [
        OperationalRefreshTriggerType.WEATHER_UPDATE,
        OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        OperationalRefreshTriggerType.SEVERITY_UPDATE,
    ],
)
def test_environmental_triggers_require_spread_reevaluation(trigger_type):
    assert requires_spread_reevaluation(trigger_type) is True


def test_resource_status_update_does_not_require_spread_reevaluation():
    assert requires_spread_reevaluation(OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE) is False


def test_refresh_policy_is_deterministic():
    first = [requires_spread_reevaluation(trigger_type) for trigger_type in OperationalRefreshTriggerType]
    second = [requires_spread_reevaluation(trigger_type) for trigger_type in OperationalRefreshTriggerType]

    assert first == second


def test_spread_reevaluation_decisions_are_centralized_in_one_policy_set():
    assert SPREAD_REEVALUATION_TRIGGER_TYPES == frozenset(
        {
            OperationalRefreshTriggerType.WEATHER_UPDATE,
            OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
            OperationalRefreshTriggerType.SEVERITY_UPDATE,
        }
    )


def test_invalid_trigger_type_is_rejected():
    with pytest.raises(ValueError):
        requires_spread_reevaluation("weather_update")
