"""EcoGuard project/demo severity-to-resource-demand policy (Stage 5 of the
Global Multi-Incident Optimizer refactor, Task 1/2).

IMPORTANT - PROJECT POLICY, NOT OFFICIAL DOCTRINE: the resource counts below
are EcoGuard engineering/demo assumptions for this capstone project. They
are NOT official Israel Fire and Rescue Authority dispatch standards, are
not derived from any published operational doctrine, and must never be
presented to a user as such. They exist so the Global GA has a concrete,
deterministic, auditable, and easily-replaceable answer to "how many
resources does this incident need" - the exact values are expected to be
revisited by domain experts before any real operational use.

One policy object owns every (severity -> resource count) mapping; no
`if severity == CRITICAL: return 4` is scattered through business logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.models.fire_severity_level import FireSeverityLevel

METHODOLOGY = "ecoguard_demo_severity_demand_policy"
METHODOLOGY_VERSION = "1.0"

# EcoGuard demo defaults (Task 2) - see module docstring: NOT official doctrine.
DEFAULT_MINIMUM_BY_LEVEL: dict[FireSeverityLevel, int] = {
    FireSeverityLevel.LOW: 1,
    FireSeverityLevel.MODERATE: 1,
    FireSeverityLevel.HIGH: 2,
    FireSeverityLevel.CRITICAL: 3,
}
DEFAULT_DESIRED_BY_LEVEL: dict[FireSeverityLevel, int] = {
    FireSeverityLevel.LOW: 1,
    FireSeverityLevel.MODERATE: 2,
    FireSeverityLevel.HIGH: 3,
    FireSeverityLevel.CRITICAL: 4,
}
DEFAULT_FALLBACK_MINIMUM_RESOURCES = 1
DEFAULT_FALLBACK_DESIRED_RESOURCES = 1

# Severity ordering used ONLY to break ties when REQUIRED slots compete
# under scarcity (Task 15) - a configured decision-support ordering, not a
# claim about real-world triage doctrine.
_SEVERITY_ORDER = (
    FireSeverityLevel.LOW,
    FireSeverityLevel.MODERATE,
    FireSeverityLevel.HIGH,
    FireSeverityLevel.CRITICAL,
)


@dataclass(frozen=True)
class SeverityDemandPolicy:
    """Maps FireSeverityLevel -> (minimum_resources, desired_resources), plus the
    fallback used when an active FireEvent has no usable severity assessment."""

    minimum_by_level: dict[FireSeverityLevel, int] = field(default_factory=lambda: dict(DEFAULT_MINIMUM_BY_LEVEL))
    desired_by_level: dict[FireSeverityLevel, int] = field(default_factory=lambda: dict(DEFAULT_DESIRED_BY_LEVEL))
    fallback_minimum_resources: int = DEFAULT_FALLBACK_MINIMUM_RESOURCES
    fallback_desired_resources: int = DEFAULT_FALLBACK_DESIRED_RESOURCES
    methodology: str = METHODOLOGY
    methodology_version: str = METHODOLOGY_VERSION

    def __post_init__(self) -> None:
        for mapping_name, mapping in (("minimum_by_level", self.minimum_by_level), ("desired_by_level", self.desired_by_level)):
            if set(mapping) != set(FireSeverityLevel):
                raise ValueError(f"{mapping_name} must define every FireSeverityLevel, got keys {sorted(m.value for m in mapping)!r}")
            for level, value in mapping.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{mapping_name}[{level!r}] must be a non-negative integer, got {value!r}")

        for level in FireSeverityLevel:
            if self.desired_by_level[level] < self.minimum_by_level[level]:
                raise ValueError(
                    f"desired_by_level[{level!r}] ({self.desired_by_level[level]!r}) must be >= "
                    f"minimum_by_level[{level!r}] ({self.minimum_by_level[level]!r})."
                )

        for previous, current in zip(_SEVERITY_ORDER, _SEVERITY_ORDER[1:]):
            if self.desired_by_level[current] < self.desired_by_level[previous]:
                raise ValueError(
                    "desired_by_level must be monotonically non-decreasing by severity "
                    f"(LOW <= MODERATE <= HIGH <= CRITICAL), got {current!r}="
                    f"{self.desired_by_level[current]!r} < {previous!r}={self.desired_by_level[previous]!r}."
                )

        if isinstance(self.fallback_minimum_resources, bool) or not isinstance(self.fallback_minimum_resources, int) or self.fallback_minimum_resources < 0:
            raise ValueError(f"fallback_minimum_resources must be a non-negative integer, got {self.fallback_minimum_resources!r}")
        if isinstance(self.fallback_desired_resources, bool) or not isinstance(self.fallback_desired_resources, int) or self.fallback_desired_resources < self.fallback_minimum_resources:
            raise ValueError(
                "fallback_desired_resources must be an integer >= fallback_minimum_resources, got "
                f"{self.fallback_desired_resources!r} < {self.fallback_minimum_resources!r}"
            )
        if not isinstance(self.methodology, str) or not self.methodology.strip():
            raise ValueError(f"methodology must be a non-empty string, got {self.methodology!r}")
        if not isinstance(self.methodology_version, str) or not self.methodology_version.strip():
            raise ValueError(f"methodology_version must be a non-empty string, got {self.methodology_version!r}")

    def demand_for_level(self, level: FireSeverityLevel) -> tuple[int, int]:
        """Return (minimum_resources, desired_resources) for a severity level."""
        if not isinstance(level, FireSeverityLevel):
            raise ValueError(f"level must be a FireSeverityLevel, got {level!r}")
        return self.minimum_by_level[level], self.desired_by_level[level]

    @staticmethod
    def severity_rank(level: FireSeverityLevel) -> int:
        """Configured severity priority rank (higher = more severe = higher shortage priority) - Task 15."""
        if not isinstance(level, FireSeverityLevel):
            raise ValueError(f"level must be a FireSeverityLevel, got {level!r}")
        return _SEVERITY_ORDER.index(level)
