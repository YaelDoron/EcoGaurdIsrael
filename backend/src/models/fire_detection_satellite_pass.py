"""One satellite observation wave ("pass") of a fire's hotspots.

A pass groups hotspot pixels that describe ONE acquisition, so later features count
observation waves - not individual pixels - when asking "was this fire seen again?".

GROUPING RULE (deterministic; see also SATELLITE_PASS_GAP_MINUTES in
services/fire_detection/fire_detection_evidence_config.py for the rationale):
  1. only SATELLITE evidence is grouped;
  2. hotspots are partitioned by (satellite, instrument) - two platforms observing at the
     same minute are two passes, and the code never assumes a particular satellite (None
     is treated as "unknown platform" and grouped with other unknowns);
  3. inside a partition, hotspots are sorted by acquisition time and a new pass starts
     whenever the gap to the previous hotspot exceeds `gap_minutes` (single linkage in
     time - no fixed buckets, so no boundary artifacts).

Only fields that are actually persisted are used: `detected_at` (FIRMS acq_date + acq_time,
minute resolution, timezone semantics not confirmed - see SatelliteHotspotMapper),
`satellite` and `instrument`. Missing FRP / brightness stay missing: a statistic is None
when no hotspot in the pass carries a measurement, never a fabricated 0.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType


@dataclass(frozen=True)
class SatellitePass:
    """The hotspots of one (satellite, instrument) acquired within the pass-gap tolerance of each other."""

    satellite_name: str | None
    satellite_instrument: str | None
    evidence: tuple[FireDetectionEvidence, ...]  # chronological, at least one hotspot

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError("A SatellitePass requires at least one hotspot.")
        for item in self.evidence:
            if item.evidence_type is not FireEvidenceType.SATELLITE:
                raise ValueError("A SatellitePass may only contain SATELLITE evidence.")

    @property
    def platform(self) -> tuple[str | None, str | None]:
        return (self.satellite_name, self.satellite_instrument)

    @property
    def first_observed_at(self) -> datetime:
        return min(item.observed_at for item in self.evidence)

    @property
    def last_observed_at(self) -> datetime:
        return max(item.observed_at for item in self.evidence)

    @property
    def observed_at(self) -> datetime:
        """Representative time of the pass: the midpoint of its hotspot acquisition times."""
        first, last = self.first_observed_at, self.last_observed_at
        return first + (last - first) / 2

    @property
    def pixel_count(self) -> int:
        return len(self.evidence)

    @property
    def centroid(self) -> tuple[float, float]:
        return (
            sum(item.latitude for item in self.evidence) / len(self.evidence),
            sum(item.longitude for item in self.evidence) / len(self.evidence),
        )

    def frp_values(self) -> tuple[float, ...]:
        return tuple(item.satellite_frp for item in self.evidence if item.satellite_frp is not None)

    def brightness_values(self) -> tuple[float, ...]:
        return tuple(item.satellite_brightness for item in self.evidence if item.satellite_brightness is not None)

    def frp_statistic(self, statistic: str) -> float | None:
        return _statistic(self.frp_values(), statistic)

    def brightness_statistic(self, statistic: str) -> float | None:
        return _statistic(self.brightness_values(), statistic)


_STATISTICS = ("max", "mean", "sum")


def _statistic(values: tuple[float, ...], statistic: str) -> float | None:
    if statistic not in _STATISTICS:
        raise ValueError(f"statistic must be one of {_STATISTICS}, got {statistic!r}")
    if not values:
        return None  # nothing was measured: missing, not zero
    if statistic == "max":
        return max(values)
    if statistic == "sum":
        return sum(values)
    return sum(values) / len(values)


def group_satellite_passes(evidence: Iterable[FireDetectionEvidence], gap_minutes: float) -> tuple[SatellitePass, ...]:
    """Group hotspot evidence into observation waves. Deterministic; news evidence is ignored."""
    if isinstance(gap_minutes, bool) or not isinstance(gap_minutes, (int, float)) or gap_minutes <= 0:
        raise ValueError(f"gap_minutes must be a positive number, got {gap_minutes!r}")

    partitions: dict[tuple[str | None, str | None], list[FireDetectionEvidence]] = {}
    for item in evidence:
        if item.evidence_type is FireEvidenceType.SATELLITE:
            partitions.setdefault((item.satellite_name, item.satellite_instrument), []).append(item)

    gap = timedelta(minutes=gap_minutes)
    passes: list[SatellitePass] = []
    for (name, instrument), items in partitions.items():
        items.sort(key=lambda item: (item.observed_at.astimezone(timezone.utc), item.evidence_id))
        current: list[FireDetectionEvidence] = [items[0]]
        for item in items[1:]:
            if item.observed_at - current[-1].observed_at > gap:
                passes.append(SatellitePass(name, instrument, tuple(current)))
                current = []
            current.append(item)
        passes.append(SatellitePass(name, instrument, tuple(current)))

    passes.sort(key=lambda p: (p.first_observed_at.astimezone(timezone.utc), p.satellite_name or "", p.satellite_instrument or ""))
    return tuple(passes)
