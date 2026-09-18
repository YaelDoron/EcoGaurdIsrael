"""GlobalIncidentDemandBuilder: reads the canonical latest severity
assessment for each active FireEvent and turns it into a
GlobalIncidentDemand via the configured SeverityDemandPolicy (Stage 5 of
the Global Multi-Incident Optimizer refactor, Task 5).

Read-only: no GA, no routing, no DB writes. Uses the SAME per-event
`get_latest_for_event_as_of(fire_event_id, as_of)` convention
GlobalPlanningInputBuilder already uses for target sets (Stage 3), so
demand generation is anchored to the same `as_of` instant and stays
deterministic/reproducible (Task 38).

Missing/invalid severity (Task 3): only a VALID FireSeverityAssessmentStatus
result is trusted. INSUFFICIENT_DATA, INACTIVE_EVENT, or no assessment at
all falls back to SeverityDemandPolicy's configured fallback demand with
demand_source=INSUFFICIENT_SEVERITY - never a fabricated severity level.
"""
from __future__ import annotations

from datetime import datetime

from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy
from src.models.demand_source import DemandSource
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.global_incident_demand import GlobalIncidentDemand
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository


class GlobalIncidentDemandBuilder:
    """Builds one GlobalIncidentDemand per active FireEvent from its latest severity assessment."""

    def __init__(
        self,
        fire_severity_assessment_repository: FireSeverityAssessmentRepository | None = None,
        policy: SeverityDemandPolicy | None = None,
    ) -> None:
        self._fire_severity_assessment_repository = (
            fire_severity_assessment_repository or FireSeverityAssessmentRepository()
        )
        self._policy = policy or SeverityDemandPolicy()

    def build(self, fire_event_ids: tuple[int, ...], as_of: datetime) -> tuple[GlobalIncidentDemand, ...]:
        demands = []
        for fire_event_id in fire_event_ids:
            stored = self._fire_severity_assessment_repository.get_latest_for_event_as_of(fire_event_id, as_of)
            if stored is not None and stored.assessment.status is FireSeverityAssessmentStatus.VALID:
                minimum_resources, desired_resources = self._policy.demand_for_level(stored.assessment.level)
                demands.append(
                    GlobalIncidentDemand(
                        fire_event_id=fire_event_id,
                        severity_assessment_id=stored.assessment_id,
                        severity_level=stored.assessment.level,
                        severity_score=stored.assessment.score,
                        minimum_resources=minimum_resources,
                        desired_resources=desired_resources,
                        demand_source=DemandSource.SEVERITY_ASSESSMENT,
                        policy_methodology=self._policy.methodology,
                        policy_version=self._policy.methodology_version,
                    )
                )
            else:
                demands.append(
                    GlobalIncidentDemand(
                        fire_event_id=fire_event_id,
                        severity_assessment_id=None,
                        severity_level=None,
                        severity_score=None,
                        minimum_resources=self._policy.fallback_minimum_resources,
                        desired_resources=self._policy.fallback_desired_resources,
                        demand_source=DemandSource.INSUFFICIENT_SEVERITY,
                        policy_methodology=self._policy.methodology,
                        policy_version=self._policy.methodology_version,
                    )
                )
        return tuple(demands)
