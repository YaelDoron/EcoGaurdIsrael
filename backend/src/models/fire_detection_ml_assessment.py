"""Structured result of one runtime ML inference pass over Fire Detection evidence."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class FireDetectionMLAssessment:
    """One ML model's estimate for one Fire Detection candidate.

    `probability` is `predict_proba(...)[1]`: a model-estimated probability
    of the FIRE class WITHIN the synthetic V3 training distribution - it is
    NOT a calibrated real-world probability of an active wildfire, and must
    never be presented as operational certainty (see
    backend/docs/fire_detection_runtime_ml.md).

    `available=False` means inference could not be run at all (model file
    missing/corrupt, metadata/schema mismatch, feature extraction failure,
    predict_proba failure, ...) - `probability` and model-identity fields are
    then None and `failure_reason` explains why. This is never conflated
    with a real low-probability result.
    """

    available: bool
    probability: float | None
    model_name: str | None
    model_version: str | None
    feature_schema_version: str | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if self.available:
            if self.failure_reason is not None:
                raise ValueError("available=True must not carry a failure_reason.")
            if (
                self.probability is None
                or isinstance(self.probability, bool)
                or not isinstance(self.probability, Real)
                or not math.isfinite(self.probability)
            ):
                raise ValueError(f"probability must be a finite number when available, got {self.probability!r}")
            if not 0.0 <= self.probability <= 1.0:
                raise ValueError(f"probability must be within [0, 1], got {self.probability!r}")
            for field_name in ("model_name", "model_version", "feature_schema_version"):
                value = getattr(self, field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{field_name} must be a non-empty string when available, got {value!r}")
        else:
            if self.probability is not None:
                raise ValueError("available=False must not carry a probability.")
            if self.failure_reason is not None and not isinstance(self.failure_reason, str):
                raise ValueError(f"failure_reason must be a string or None, got {self.failure_reason!r}")
