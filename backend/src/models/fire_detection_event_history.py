"""Evidence history of an active FireEvent - NOT a FireDetectionCandidate.

Two different concepts (Task 5B):

  FireDetectionCandidate        the CURRENT candidate: evidence that correlates (<= 5 km and
                                <= 60 minutes, chained). Unit of the rule / ML decision.
  FireDetectionEventHistory     the recent OBSERVATION HISTORY of one FireEvent: every piece of
                                evidence already attached to it inside the history window, which
                                may be many hours apart. It deliberately does NOT enforce the
                                60-minute candidate invariant.

The history is read-only state for future features. It performs no ML feature extraction and
no decision. Its summaries are conservative: anything that cannot be computed reliably from
the observed data is None (never a guessed or zero-filled value).

What is deliberately NOT here:
  * `repeat_detection_ratio`: a true ratio needs to know when the site was OBSERVED but had no
    detection, i.e. the satellite overpass schedule / coverage. FIRMS lists detections only, so
    the denominator does not exist in our data. `distinct_satellite_pass_count` is the honest
    primitive.
  * site recurrence over days (the same place hot on different days): a separate future query
    over the hotspot table, not part of one event's history.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math

from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_satellite_pass import SatellitePass, group_satellite_passes
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.utils.geo import haversine_distance_km

MIN_PASSES_FOR_TREND = 3  # one pass has no trend; two passes give a slope but no reliability check

_EVIDENCE_TYPE_ORDER = {FireEvidenceType.SATELLITE: 0, FireEvidenceType.NEWS: 1}


def chronological_key(item: FireDetectionEvidence) -> tuple[float, int, int]:
    """The same ordering the evidence service uses: time, then source type, then id."""
    return (item.observed_at.astimezone(timezone.utc).timestamp(), _EVIDENCE_TYPE_ORDER[item.evidence_type], item.evidence_id)


def evidence_ref(item: FireDetectionEvidence) -> FireEvidenceRef:
    return FireEvidenceRef(evidence_type=item.evidence_type, evidence_id=item.evidence_id)


def correlated_evidence(
    evidence: Sequence[FireDetectionEvidence],
    seed_refs: Sequence[FireEvidenceRef],
) -> tuple[FireDetectionEvidence, ...]:
    """The part of `evidence` that belongs to the SAME current candidate as the seed evidence.

    Starting from the seed items, adds every item that directly correlates (the one
    FireDetectionCandidate rule) with an item already included, until nothing more joins:
    the connected component(s) containing the seeds. Evidence that does not correlate - e.g. an
    earlier observation wave hours ago - is left out; it remains part of the event's HISTORY.
    Input order is preserved. No seed found -> ().
    """
    items = tuple(evidence)
    wanted = set(seed_refs)
    included = {index for index, item in enumerate(items) if evidence_ref(item) in wanted}
    frontier = list(included)
    while frontier:
        current = items[frontier.pop()]
        for index, candidate in enumerate(items):
            if index not in included and FireDetectionCandidate.are_correlated(current, candidate):
                included.add(index)
                frontier.append(index)
    return tuple(items[index] for index in sorted(included))


@dataclass(frozen=True)
class TrendEstimate:
    """Least-squares slope of a per-pass statistic over pass time."""

    slope_per_hour: float
    pass_count: int


def _least_squares_trend(points: list[tuple[float, float]]) -> TrendEstimate | None:
    if len(points) < MIN_PASSES_FOR_TREND:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    if variance == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    return TrendEstimate(slope_per_hour=slope, pass_count=len(points))


# --- pure pass-level helpers -------------------------------------------------------------------
# The ONE implementation of pass trend / span / centroid stability. FireDetectionEventHistory's
# properties and the V5 ML feature extractor both call these, so a runtime history and a training
# history can never be summarised by two slightly different algorithms.


def satellite_pass_trend(
    passes: Sequence[SatellitePass],
    value_of,
    platform: tuple[str | None, str | None] | None = None,
) -> TrendEstimate | None:
    """Least-squares slope (per hour) of a per-pass statistic; None unless >= MIN_PASSES_FOR_TREND passes carry a value.

    `passes` must be chronological (group_satellite_passes returns them so). Without an explicit
    `platform`, passes of several platforms are not comparable -> None.
    """
    passes = tuple(passes)
    if platform is None:
        if len({p.platform for p in passes}) != 1:
            return None
    else:
        passes = tuple(p for p in passes if p.platform == platform)
    first_time = passes[0].observed_at if passes else None
    points = [
        ((p.observed_at - first_time) / timedelta(hours=1), value)
        for p in passes
        if (value := value_of(p)) is not None and not math.isnan(value)
    ]
    return _least_squares_trend(points)


def satellite_pass_time_span_minutes(passes: Sequence[SatellitePass]) -> float | None:
    """Minutes between the earliest and the latest PASS time (pass = midpoint of its hotspots); None without passes.

    One pass -> 0.0 (a single observation wave has no extent in time, however many pixels it has).
    """
    times = [p.observed_at for p in passes]
    if not times:
        return None
    return (max(times) - min(times)).total_seconds() / 60.0


def satellite_pass_centroid_rms_km(passes: Sequence[SatellitePass]) -> float | None:
    """RMS great-circle distance of each pass centroid from the mean of the pass centroids (km).

    Needs >= 2 passes: one pass cannot show movement, so it is None - never 0.0, which would read
    as "observed twice and perfectly stable". Every pass counts once, however many pixels it has.
    """
    centroids = [p.centroid for p in passes]
    if len(centroids) < 2:
        return None
    mean_lat = sum(lat for lat, _ in centroids) / len(centroids)
    mean_lon = sum(lon for _, lon in centroids) / len(centroids)
    squares = [haversine_distance_km(lat, lon, mean_lat, mean_lon) ** 2 for lat, lon in centroids]
    return math.sqrt(sum(squares) / len(squares))


@dataclass(frozen=True)
class FireDetectionEventHistory:
    """Immutable evidence history of one FireEvent inside [window_start, as_of].

    `evidence` is chronological and free of duplicate identities. It is NOT required to form a
    valid FireDetectionCandidate. `satellite_passes` groups its hotspots into observation waves
    (see fire_detection_satellite_pass.py). `unresolved_evidence` lists attached refs that could
    not be reloaded/normalized, so the history is honest about what it could not see.
    """

    fire_event_id: int
    as_of: datetime
    window_start: datetime
    evidence: tuple[FireDetectionEvidence, ...]
    satellite_pass_gap_minutes: float
    unresolved_evidence: tuple[FireEvidenceRef, ...] = ()
    satellite_passes: tuple[SatellitePass, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        for name in ("as_of", "window_start"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise ValueError(f"{name} must be a timezone-aware datetime, got {value!r}")
        if self.window_start > self.as_of:
            raise ValueError("window_start must not be after as_of.")

        items = tuple(self.evidence)
        identities: set[FireEvidenceRef] = set()
        for item in items:
            if not isinstance(item, FireDetectionEvidence):
                raise ValueError(f"evidence must contain FireDetectionEvidence items, got {item!r}")
            identity = evidence_ref(item)
            if identity in identities:
                raise ValueError(f"Duplicate evidence identity in event history: {identity!r}")
            identities.add(identity)
            if not self.window_start <= item.observed_at <= self.as_of:
                raise ValueError(f"Evidence {identity!r} is outside the history window.")

        object.__setattr__(self, "evidence", tuple(sorted(items, key=chronological_key)))
        object.__setattr__(self, "unresolved_evidence", tuple(self.unresolved_evidence))
        object.__setattr__(
            self, "satellite_passes", group_satellite_passes(self.evidence, self.satellite_pass_gap_minutes)
        )

    # --- contents ---

    @property
    def evidence_refs(self) -> tuple[FireEvidenceRef, ...]:
        return tuple(evidence_ref(item) for item in self.evidence)

    @property
    def satellite_evidence(self) -> tuple[FireDetectionEvidence, ...]:
        return tuple(item for item in self.evidence if item.evidence_type is FireEvidenceType.SATELLITE)

    @property
    def news_evidence(self) -> tuple[FireDetectionEvidence, ...]:
        return tuple(item for item in self.evidence if item.evidence_type is FireEvidenceType.NEWS)

    # --- conservative summaries (primitives for future persistence features) ---

    @property
    def distinct_satellite_pass_count(self) -> int:
        """Observation waves (not pixels) in the window, across all platforms."""
        return len(self.satellite_passes)

    @property
    def satellite_observation_span_minutes(self) -> float | None:
        """Minutes from the first hotspot of the first pass to the last hotspot of the last pass; None without hotspots."""
        if not self.satellite_passes:
            return None
        first = min(p.first_observed_at for p in self.satellite_passes)
        last = max(p.last_observed_at for p in self.satellite_passes)
        return (last - first).total_seconds() / 60.0

    @property
    def satellite_centroid_shifts_km(self) -> tuple[float, ...]:
        """Distance between the centroids of consecutive passes, chronologically (empty for < 2 passes)."""
        centroids = [p.centroid for p in self.satellite_passes]
        return tuple(haversine_distance_km(*a, *b) for a, b in zip(centroids, centroids[1:]))

    @property
    def satellite_centroid_max_shift_km(self) -> float | None:
        """Largest consecutive centroid shift; None with fewer than 2 passes (stability is unknowable from one pass)."""
        shifts = self.satellite_centroid_shifts_km
        return max(shifts) if shifts else None

    def frp_trend(self, statistic: str = "max", platform: tuple[str | None, str | None] | None = None) -> TrendEstimate | None:
        """Slope (per hour) of the per-pass FRP statistic, or None when it cannot be computed reliably."""
        return self._trend(lambda p: p.frp_statistic(statistic), platform)

    def brightness_trend(
        self, statistic: str = "max", platform: tuple[str | None, str | None] | None = None
    ) -> TrendEstimate | None:
        return self._trend(lambda p: p.brightness_statistic(statistic), platform)

    def _trend(self, value_of, platform) -> TrendEstimate | None:
        return satellite_pass_trend(self.satellite_passes, value_of, platform)
