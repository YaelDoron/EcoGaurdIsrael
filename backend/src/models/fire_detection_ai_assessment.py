"""Result of one AI Hybrid V5 inference for one Fire Detection candidate (Task 9B)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.fire_detection_status import FireDetectionStatus


@dataclass(frozen=True)
class FireDetectionAIAssessment:
    """What the V5 model and the locked AI Hybrid Policy concluded about ONE candidate (plus its event history).

    `probability` is the HGB model's P(fire) WITHIN the synthetic V5 distribution - a model score, NOT a calibrated
    real-world probability of wildfire and never operational certainty. `policy_status` is the status the locked
    policy proposes for it; the FireEvent's stored status can differ (CONFIRMED is never downgraded, SUSPECTED is
    never dismissed by a weaker later assessment).

    `current_satellite_pixel_count` counts only the CURRENT candidate's hotspots (the corroboration guardrail never
    counts historical pixels). `satellite_pass_count` is the effective pass count of candidate + history.
    Deliberately does not carry the 25-feature vector: that is an audit/debug artefact, not a user-facing value.
    """

    probability: float
    policy_status: FireDetectionStatus
    model_name: str
    model_version: str
    feature_schema_version: str
    policy_version: str
    current_satellite_pixel_count: int
    satellite_pass_count: int
    history_available: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, Real)
            or not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError(f"probability must be finite within [0, 1], got {self.probability!r}")
        if not isinstance(self.policy_status, FireDetectionStatus):
            raise ValueError(f"policy_status must be a FireDetectionStatus, got {self.policy_status!r}")
        for field_name in ("model_name", "model_version", "feature_schema_version", "policy_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
        for field_name in ("current_satellite_pixel_count", "satellite_pass_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")
        if not isinstance(self.history_available, bool):
            raise ValueError(f"history_available must be a bool, got {self.history_available!r}")
