"""GlobalIncidentDemand: the canonical Stage-5 answer to "how many
firefighting resources does this incident currently need according to
EcoGuard's configured demand policy?" (Global Multi-Incident Optimizer
refactor, Task 4).

These resource counts are an EcoGuard project/demo decision-support
policy, NOT official Israeli firefighting doctrine - see
SeverityDemandPolicy's own docstring.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.calculators.fire_severity.fire_severity_config import MAX_SEVERITY_SCORE, MIN_SEVERITY_SCORE
from src.models.demand_source import DemandSource
from src.models.fire_severity_level import FireSeverityLevel
from src.models.optimization_validation import validate_non_empty_string, validate_non_negative_int, validate_positive_int


@dataclass(frozen=True)
class GlobalIncidentDemand:
    """One active FireEvent's configured resource demand at input-build time."""

    fire_event_id: int
    severity_assessment_id: int | None
    severity_level: FireSeverityLevel | None
    severity_score: float | None
    minimum_resources: int
    desired_resources: int
    demand_source: DemandSource
    policy_methodology: str
    policy_version: str

    def __post_init__(self) -> None:
        validate_positive_int("fire_event_id", self.fire_event_id)
        if self.severity_assessment_id is not None:
            validate_positive_int("severity_assessment_id", self.severity_assessment_id)
        if self.severity_level is not None and not isinstance(self.severity_level, FireSeverityLevel):
            raise ValueError(f"severity_level must be a FireSeverityLevel or None, got {self.severity_level!r}")
        if self.severity_score is not None:
            self._validate_severity_score(self.severity_score)

        validate_non_negative_int("minimum_resources", self.minimum_resources)
        validate_non_negative_int("desired_resources", self.desired_resources)
        if self.desired_resources < self.minimum_resources:
            raise ValueError(
                "desired_resources must be >= minimum_resources, got desired="
                f"{self.desired_resources!r} < minimum={self.minimum_resources!r}"
            )

        if not isinstance(self.demand_source, DemandSource):
            raise ValueError(f"demand_source must be a DemandSource, got {self.demand_source!r}")
        validate_non_empty_string("policy_methodology", self.policy_methodology)
        validate_non_empty_string("policy_version", self.policy_version)

        if self.demand_source is DemandSource.INSUFFICIENT_SEVERITY:
            if self.severity_assessment_id is not None or self.severity_level is not None or self.severity_score is not None:
                raise ValueError(
                    "INSUFFICIENT_SEVERITY demand must not carry severity_assessment_id/severity_level/severity_score."
                )
        elif self.demand_source is DemandSource.SEVERITY_ASSESSMENT:
            if self.severity_level is None:
                raise ValueError("SEVERITY_ASSESSMENT demand must include severity_level.")

    @staticmethod
    def _validate_severity_score(value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"severity_score must be a finite number, got {value!r}")
        if not MIN_SEVERITY_SCORE <= value <= MAX_SEVERITY_SCORE:
            raise ValueError(
                f"severity_score must be within [{MIN_SEVERITY_SCORE}, {MAX_SEVERITY_SCORE}], got {value!r}"
            )
