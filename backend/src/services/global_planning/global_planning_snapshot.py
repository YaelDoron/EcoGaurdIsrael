"""GlobalPlanningSnapshot: the state observed at the start of one
GlobalPlanningRun (Stage 2 of the Global Multi-Incident Optimizer
refactor, Tasks 6-7).

Captured exactly ONCE, at the beginning of GlobalPlanningOrchestrator.run():
active_fire_event_ids is a plain tuple, never re-queried mid-cycle, so a
FireEvent that becomes active only after capture belongs to the NEXT run
(Task 6's determinism rule), not this one.

`fingerprint()` is for traceability only in this stage (Task 7) - nothing
in Stage 2 uses it to suppress a global run as a no-op; Stage 4 may.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

from src.models.resource_commitment import ResourceCommitment


@dataclass(frozen=True)
class GlobalPlanningSnapshot:
    """The active-event set, per-event local fingerprints, and current commitments at run start."""

    as_of: datetime
    active_fire_event_ids: tuple[int, ...]
    resource_commitments: tuple[ResourceCommitment, ...]
    per_event_effective_state_fingerprints: dict[int, str | None] = field(default_factory=dict)
    methodology: str = ""
    methodology_version: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, datetime) or self.as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {self.as_of!r}")

        active_fire_event_ids = _coerce_tuple("active_fire_event_ids", self.active_fire_event_ids)
        for fire_event_id in active_fire_event_ids:
            if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
                raise ValueError(f"active_fire_event_ids must contain positive integers, got {fire_event_id!r}")
        if len(set(active_fire_event_ids)) != len(active_fire_event_ids):
            raise ValueError("active_fire_event_ids must not contain duplicates.")
        object.__setattr__(self, "active_fire_event_ids", active_fire_event_ids)

        resource_commitments = _coerce_tuple("resource_commitments", self.resource_commitments)
        for commitment in resource_commitments:
            if not isinstance(commitment, ResourceCommitment):
                raise ValueError(f"resource_commitments must contain ResourceCommitment items, got {commitment!r}")
        object.__setattr__(self, "resource_commitments", resource_commitments)

        if not isinstance(self.per_event_effective_state_fingerprints, dict):
            raise ValueError(
                "per_event_effective_state_fingerprints must be a dict, got "
                f"{self.per_event_effective_state_fingerprints!r}"
            )
        for fire_event_id, fingerprint in self.per_event_effective_state_fingerprints.items():
            if fire_event_id not in active_fire_event_ids:
                raise ValueError(
                    f"per_event_effective_state_fingerprints key {fire_event_id!r} "
                    "is not one of active_fire_event_ids."
                )
            if fingerprint is not None and not isinstance(fingerprint, str):
                raise ValueError(f"per_event_effective_state_fingerprints values must be str or None, got {fingerprint!r}")

        if not isinstance(self.methodology, str) or not self.methodology.strip():
            raise ValueError(f"methodology must be a non-empty string, got {self.methodology!r}")
        if not isinstance(self.methodology_version, str) or not self.methodology_version.strip():
            raise ValueError(f"methodology_version must be a non-empty string, got {self.methodology_version!r}")

    def fingerprint(self) -> str:
        """Stable SHA-256 hex digest of this snapshot's semantic content.

        Order-independent over active_fire_event_ids/resource_commitments/
        fingerprints. Excludes `as_of` (a persistence timestamp, not
        semantic content) and any DB surrogate id without semantic meaning
        - response_plan_id IS included per-commitment because WHICH plan a
        resource is currently committed under is itself part of "what was
        observed," not merely a row identifier.
        """
        payload = json.dumps(self._canonical_payload(), separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _canonical_payload(self) -> tuple[tuple[str, object], ...]:
        sorted_event_ids = tuple(sorted(self.active_fire_event_ids))
        sorted_fingerprints = tuple(
            sorted(
                (
                    (fire_event_id, fingerprint)
                    for fire_event_id, fingerprint in self.per_event_effective_state_fingerprints.items()
                ),
                key=lambda pair: pair[0],
            )
        )
        sorted_commitments = tuple(
            sorted(
                (
                    (commitment.resource_id, commitment.fire_event_id, commitment.response_plan_id)
                    for commitment in self.resource_commitments
                ),
                key=lambda triple: triple[0],
            )
        )
        return (
            ("active_fire_event_ids", sorted_event_ids),
            ("per_event_effective_state_fingerprints", sorted_fingerprints),
            ("resource_commitments", sorted_commitments),
            ("methodology", self.methodology),
            ("methodology_version", self.methodology_version),
        )


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc
