"""Deterministic, offline synthetic labeled dataset generator for Fire Detection ML V5.

WHAT A ROW IS
    One current FireDetectionCandidate + the event history known at that assessment time + the
    ground-truth label:

        current candidate  (validated by the real FireDetectionCandidate: <= 5 km / 60 min, chained)
        + FireDetectionEventHistory (the real Task 5B class; None before a FireEvent exists)
        + label

    This module only builds EVIDENCE. It computes no ML feature: features are produced later, for
    training and for runtime alike, by FireDetectionFeatureExtractorV5 (see fire_detection_dataset_v5).
    It does not import that extractor, the feature schema, FireDetectionCalculator, the hybrid policy or
    any ML model, so the label cannot inherit the hand-written scoring rules it is meant to replace.

REGIME != LABEL   (the central V4 fix)
    A REGIME describes what KIND OF EVIDENCE SITUATION exists (single hotspot, news first, persistent
    thermal source, ...). It never encodes whether there is a fire: every regime contains both labels,
    ~50/50, by construction. The label comes from a LATENT PROCESS chosen per row:

        fire:     early_wildfire, established_wildfire, large_wildfire
        no fire:  sensor_noise, industrial_heat_source, controlled_or_agricultural_burn, false_or_rumour_report

    The latent process is drawn BEFORE any evidence and evidence is then sampled from that process
    with heavily overlapping distributions (weak real fires, strong false alarms, steady industrial heat
    that persists for hours). `latent_subtype` is metadata for analysis; it is never a feature.

THINGS SHARED vs THINGS DRAWN
    * ENVIRONMENT (`environment_id`): nuisance properties shared by many rows - location, pixel size,
      sensor/background offsets, FRP/brightness missing rates, revisit period, coverage (clouds), share
      of night overpasses. Rows of one environment must stay in one evaluation fold (Task 7).
    * PAIR (`pair_id`): a fire and a no-fire row sharing environment, regime, observation opportunities,
      timing, day/night regime and coverage, but with INDEPENDENT latent-process draws that overlap.

DETERMINISM
    Every row draws from its own random.Random seeded from (seed, unit index); environments from
    (seed, environment id). Identical output for the same seed. No network, DB, LLM or wall clock.

HISTORY
    Only PERSISTENT_THERMAL rows carry earlier satellite passes (>= 1). Hotspots of every pass are
    produced by the same pixel model; whether an earlier opportunity produced a detection depends on the
    latent process (a wildfire that started recently, an industrial source that has always been there, a
    burn that lasts a few hours, noise that only sometimes repeats) AND on shared coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
import random

from src.calculators.fire_detection.fire_detection_config import (  # correlation limits ONLY
    MAX_EVIDENCE_DISTANCE_KM,
    MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_event_history import FireDetectionEventHistory
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.services.fire_detection.fire_detection_evidence_config import (
    FIRE_EVENT_EVIDENCE_HISTORY_HOURS,
    SATELLITE_PASS_GAP_MINUTES,
)

DEFAULT_TRAINING_DATA_SEED_V5 = 42
DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V5 = 10000
ROWS_PER_ENVIRONMENT = 40
FRACTION_OF_ROWS_IN_PAIRS = 0.5

SATELLITE_NAME = "NOAA-20"  # the runtime feed is effectively one platform; no multi-platform signal
SATELLITE_INSTRUMENT = "VIIRS"

_ISRAEL_LATITUDE_RANGE = (29.6, 33.2)
_ISRAEL_LONGITUDE_RANGE = (34.3, 35.7)
_BASE_TIME_START = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
_BASE_TIME_RANGE_DAYS = 365
_AS_OF_DELAY_MAX_MINUTES = 10

# Every item of the CURRENT candidate lies within this radius of the source centre, so any two items are
# < 2 * radius = 4.5 km apart and correlate directly under the runtime's 5 km limit.
MAX_CANDIDATE_RADIUS_KM = 2.25
_MAX_PASS_CENTRE_JITTER_KM = 2.8
_MAX_NEWS_OFFSET_MINUTES = 50
_HISTORY_MARGIN_HOURS = 1.0  # history opportunities stay well inside the 24 h window
assert 2 * MAX_CANDIDATE_RADIUS_KM < MAX_EVIDENCE_DISTANCE_KM
assert _MAX_NEWS_OFFSET_MINUTES + 3 < MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES

_KM_PER_DEGREE_LATITUDE = 111.195


class RegimeV5(Enum):
    """The kind of evidence situation. Contains BOTH labels; never a label."""

    SATELLITE_ONLY_SINGLE_PASS = "satellite_only_single_pass"  # exactly one hotspot, no news, no history
    MULTI_PIXEL_SINGLE_PASS = "multi_pixel_single_pass"  # >= 2 hotspots of one pass, no news, no history
    NEWS_LED = "news_led"  # news first; the satellite may follow or never appear
    SATELLITE_NEWS = "satellite_news"  # hotspots and news together, news before or after
    PERSISTENT_THERMAL = "persistent_thermal"  # earlier satellite passes exist for the same source
    SPARSE_EARLY_EVIDENCE = "sparse_early_evidence"  # one weak item only (a weak hotspot or a weak report)


class LatentSubtypeV5(Enum):
    """The latent process behind a row. The label is a property of it, decided before any evidence."""

    EARLY_WILDFIRE = "early_wildfire"
    ESTABLISHED_WILDFIRE = "established_wildfire"
    LARGE_WILDFIRE = "large_wildfire"
    SENSOR_NOISE = "sensor_noise"
    INDUSTRIAL_HEAT_SOURCE = "industrial_heat_source"
    CONTROLLED_OR_AGRICULTURAL_BURN = "controlled_or_agricultural_burn"
    FALSE_OR_RUMOUR_REPORT = "false_or_rumour_report"

    @property
    def is_fire(self) -> bool:
        return self in _FIRE_SUBTYPES


_FIRE_SUBTYPES = frozenset(
    (LatentSubtypeV5.EARLY_WILDFIRE, LatentSubtypeV5.ESTABLISHED_WILDFIRE, LatentSubtypeV5.LARGE_WILDFIRE)
)
FIRE_SUBTYPES_V5 = tuple(sorted(_FIRE_SUBTYPES, key=lambda s: s.value))
NO_FIRE_SUBTYPES_V5 = tuple(s for s in LatentSubtypeV5 if not s.is_fire)

_REGIME_WEIGHTS: dict[RegimeV5, float] = {
    RegimeV5.SATELLITE_ONLY_SINGLE_PASS: 0.16,
    RegimeV5.MULTI_PIXEL_SINGLE_PASS: 0.16,
    RegimeV5.NEWS_LED: 0.14,
    RegimeV5.SATELLITE_NEWS: 0.18,
    RegimeV5.PERSISTENT_THERMAL: 0.26,
    RegimeV5.SPARSE_EARLY_EVIDENCE: 0.10,
}

PAIR_TYPE_BY_REGIME: dict[RegimeV5, str] = {
    RegimeV5.SATELLITE_ONLY_SINGLE_PASS: "weak_early_fire_vs_weak_satellite_noise",
    RegimeV5.MULTI_PIXEL_SINGLE_PASS: "multi_pixel_fire_vs_industrial_cluster",
    RegimeV5.NEWS_LED: "news_led_fire_vs_repeated_false_reports",
    RegimeV5.SATELLITE_NEWS: "satellite_and_real_news_fire_vs_thermal_source_and_false_news",
    RegimeV5.PERSISTENT_THERMAL: "persistent_wildfire_vs_persistent_industrial_heat",
    RegimeV5.SPARSE_EARLY_EVIDENCE: "sparse_early_fire_vs_sparse_false_signal",
}

_NEWS_CHOICES: tuple[NewsWildfireSignalStrength | None, ...] = (
    NewsWildfireSignalStrength.NONE,
    NewsWildfireSignalStrength.WEAK,
    NewsWildfireSignalStrength.MODERATE,
    NewsWildfireSignalStrength.STRONG,
    None,  # analysis unavailable
)
_WEAK_NEWS = (NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.WEAK, None)
# Weights over _NEWS_CHOICES (NONE, WEAK, MODERATE, STRONG, UNKNOWN). Real fires lean stronger, but every
# profile - including false reports - can produce every signal, STRONG included.
_NEWS_PROFILES: dict[str, tuple[float, ...]] = {
    "early_wildfire": (0.08, 0.26, 0.30, 0.28, 0.08),
    "established_wildfire": (0.04, 0.13, 0.32, 0.43, 0.08),
    "large_wildfire": (0.02, 0.07, 0.24, 0.59, 0.08),
    "false": (0.15, 0.32, 0.26, 0.19, 0.08),
    "industrial": (0.14, 0.30, 0.28, 0.20, 0.08),
    "controlled": (0.08, 0.22, 0.32, 0.30, 0.08),
    "unrelated": (0.25, 0.32, 0.21, 0.14, 0.08),
}
_NEWS_PROFILE_OF: dict[LatentSubtypeV5, str] = {
    LatentSubtypeV5.EARLY_WILDFIRE: "early_wildfire",
    LatentSubtypeV5.ESTABLISHED_WILDFIRE: "established_wildfire",
    LatentSubtypeV5.LARGE_WILDFIRE: "large_wildfire",
    LatentSubtypeV5.SENSOR_NOISE: "unrelated",
    LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: "industrial",
    LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN: "controlled",
    LatentSubtypeV5.FALSE_OR_RUMOUR_REPORT: "false",
}


@dataclass(frozen=True)
class _ThermalProfile:
    """How a thermal source behaves. Ranges are drawn per row so that profiles overlap heavily."""

    log_frp: tuple[float, float]  # log10(total FRP, MW) at the current pass: (mean, sd)
    drift: tuple[float, float]  # change of log10 FRP per observation opportunity: (mean, sd)
    walk: float  # additional per-pass log10 FRP noise
    spread_km: tuple[float, float]  # pixel scatter around the pass centre (uniform range)
    jitter_km: tuple[float, float]  # pass-to-pass centre movement (uniform range)
    detect: tuple[float, float]  # chance an existing source is detected on a covered earlier pass
    confidence_bias: float
    brightness_bias: float  # K
    brightness_noise: float  # K


_THERMAL_PROFILES: dict[LatentSubtypeV5, _ThermalProfile] = {
    LatentSubtypeV5.EARLY_WILDFIRE: _ThermalProfile((0.70, 0.45), (0.06, 0.10), 0.10, (0.25, 0.70), (0.15, 0.90), (0.55, 0.90), 0.10, 0.0, 6.0),
    LatentSubtypeV5.ESTABLISHED_WILDFIRE: _ThermalProfile((1.20, 0.45), (0.03, 0.10), 0.10, (0.40, 1.20), (0.15, 0.90), (0.65, 0.95), 0.15, 1.0, 6.0),
    LatentSubtypeV5.LARGE_WILDFIRE: _ThermalProfile((1.90, 0.40), (0.00, 0.10), 0.10, (0.70, 1.70), (0.15, 1.00), (0.70, 0.97), 0.25, 2.0, 6.0),
    LatentSubtypeV5.SENSOR_NOISE: _ThermalProfile((0.50, 0.55), (0.00, 0.15), 0.20, (0.30, 1.40), (0.40, 1.50), (0.20, 0.50), -0.35, -2.0, 9.0),
    LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: _ThermalProfile((1.15, 0.50), (0.00, 0.03), 0.05, (0.15, 0.60), (0.03, 0.50), (0.80, 0.97), 0.00, 3.0, 4.0),
    LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN: _ThermalProfile((1.00, 0.50), (-0.12, 0.10), 0.10, (0.30, 1.00), (0.15, 0.80), (0.60, 0.90), -0.05, 0.0, 6.0),
}
# A false report has no thermal source; when a hotspot happens to be present it is background clutter.
_THERMAL_PROFILE_OF_FALSE_REPORT = LatentSubtypeV5.SENSOR_NOISE

# Latent-subtype mixes per (regime, label). "with_satellite" / "news_only" split matters only where a
# row may lack hotspots. Weights are relative.
_FIRE_MIX_DEFAULT = {
    LatentSubtypeV5.EARLY_WILDFIRE: 0.45,
    LatentSubtypeV5.ESTABLISHED_WILDFIRE: 0.35,
    LatentSubtypeV5.LARGE_WILDFIRE: 0.20,
}
_FIRE_MIX_SMALL = {  # single weak hotspot / weak report: mostly early fires
    LatentSubtypeV5.EARLY_WILDFIRE: 0.65,
    LatentSubtypeV5.ESTABLISHED_WILDFIRE: 0.28,
    LatentSubtypeV5.LARGE_WILDFIRE: 0.07,
}
_FIRE_MIX_SPARSE = {
    LatentSubtypeV5.EARLY_WILDFIRE: 0.85,
    LatentSubtypeV5.ESTABLISHED_WILDFIRE: 0.15,
}
_FIRE_MIX_MULTI = {
    LatentSubtypeV5.EARLY_WILDFIRE: 0.30,
    LatentSubtypeV5.ESTABLISHED_WILDFIRE: 0.40,
    LatentSubtypeV5.LARGE_WILDFIRE: 0.30,
}
_NO_FIRE_MIX_SATELLITE = {
    LatentSubtypeV5.SENSOR_NOISE: 0.40,
    LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: 0.30,
    LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN: 0.30,
}
_NO_FIRE_MIX_MULTI = {
    LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: 0.45,
    LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN: 0.35,
    LatentSubtypeV5.SENSOR_NOISE: 0.20,
}
_NO_FIRE_MIX_NEWS_ONLY = {
    LatentSubtypeV5.FALSE_OR_RUMOUR_REPORT: 0.65,
    LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN: 0.20,
    LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: 0.15,
}
_NO_FIRE_MIX_MIXED = {  # hotspots AND news
    LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: 0.30,
    LatentSubtypeV5.FALSE_OR_RUMOUR_REPORT: 0.30,
    LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN: 0.25,
    LatentSubtypeV5.SENSOR_NOISE: 0.15,
}
_NO_FIRE_MIX_PERSISTENT = {
    LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: 0.60,
    LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN: 0.28,
    LatentSubtypeV5.SENSOR_NOISE: 0.12,
}
_NO_FIRE_MIX_SPARSE_SATELLITE = {
    LatentSubtypeV5.SENSOR_NOISE: 0.60,
    LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: 0.20,
    LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN: 0.20,
}
_NO_FIRE_MIX_SPARSE_NEWS = {
    LatentSubtypeV5.FALSE_OR_RUMOUR_REPORT: 0.80,
    LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN: 0.20,
}
# The no-fire side of a PAIR is pinned to the pair type's counterpart with this probability.
_PAIR_NO_FIRE_COUNTERPART: dict[RegimeV5, dict[LatentSubtypeV5, float]] = {
    RegimeV5.SATELLITE_ONLY_SINGLE_PASS: {LatentSubtypeV5.SENSOR_NOISE: 1.0},
    RegimeV5.MULTI_PIXEL_SINGLE_PASS: {LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: 1.0},
    RegimeV5.PERSISTENT_THERMAL: {LatentSubtypeV5.INDUSTRIAL_HEAT_SOURCE: 1.0},
}


def _weighted_choice(rng: random.Random, weights: dict) -> object:
    total = sum(weights.values())
    threshold = rng.random() * total
    running = 0.0
    chosen = None
    for key, weight in weights.items():  # dict order is insertion order: deterministic
        running += weight
        chosen = key
        if threshold < running:
            break
    return chosen


def _poisson(rng: random.Random, lam: float) -> int:
    limit = math.exp(-lam)
    count, product = 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def _is_night(moment: datetime) -> bool:
    """Israel standard time (UTC+2): night when the local hour is 19:00-05:59."""
    local_hour = (moment.hour + 2) % 24
    return local_hour >= 19 or local_hour < 6


@dataclass(frozen=True)
class _Environment:
    environment_id: int
    latitude: float
    longitude: float
    pixel_scale: float  # pixel footprint / 0.5 km (scan angle): widens pixel scatter
    frp_shift_dex: float  # sensor / background calibration offset on log10 FRP
    background_kelvin: float
    frp_missing_rate: float
    brightness_missing_rate: float
    revisit_hours: float
    coverage: float  # chance an opportunity yields a valid look at the site (cloud etc.)
    night_share: float
    confidence_shift: float
    day_night_unknown_rate: float

    @staticmethod
    def draw(seed: int, environment_id: int) -> _Environment:
        rng = random.Random(f"ecoguard-fire-detection-v5-environment:{seed}:{environment_id}")
        return _Environment(
            environment_id=environment_id,
            latitude=rng.uniform(*_ISRAEL_LATITUDE_RANGE),
            longitude=rng.uniform(*_ISRAEL_LONGITUDE_RANGE),
            pixel_scale=rng.uniform(0.75, 1.6),
            frp_shift_dex=rng.gauss(0.0, 0.12),
            background_kelvin=rng.uniform(295.0, 315.0),
            frp_missing_rate=rng.uniform(0.0, 0.25),
            brightness_missing_rate=rng.uniform(0.0, 0.15),
            revisit_hours=rng.uniform(2.5, 9.0),
            coverage=rng.uniform(0.55, 0.95),
            night_share=rng.uniform(0.25, 0.75),
            confidence_shift=rng.gauss(0.0, 0.12),
            day_night_unknown_rate=rng.uniform(0.0, 0.06),
        )


@dataclass(frozen=True)
class _Structure:
    """What a pair shares: the observation opportunities and their conditions (never the latent draws)."""

    anchor: datetime  # current pass time (or the first report when there is no hotspot)
    night: bool
    day_night_known: bool
    previous_times: tuple[datetime, ...]  # earlier observation opportunities, chronological
    previous_covered: tuple[bool, ...]
    news_only: bool  # NEWS_LED / sparse: no hotspot in the current candidate
    sparse_mode_news: bool  # SPARSE_EARLY_EVIDENCE: the single item is a report
    persistent_news: bool  # PERSISTENT_THERMAL: the current candidate also holds news


@dataclass(frozen=True)
class _Unit:
    regime: RegimeV5
    pair_id: int | None
    label: int | None  # None for a pair (one row of each label)


@dataclass(frozen=True)
class FireDetectionTrainingSampleV5:
    """One synthetic labeled Fire Detection V5 row: candidate + history + ground truth.

    `label` follows from `latent_subtype` (a wildfire process exists or not) and never from evidence.
    sample_id, seed, environment_id, regime, latent_subtype, pair_id, pair_type and as_of are metadata for
    splitting and analysis only and must never be used as ML features.
    """

    sample_id: int
    seed: int
    environment_id: int
    regime: RegimeV5
    latent_subtype: LatentSubtypeV5
    pair_id: int | None
    pair_type: str | None
    label: int
    candidate: FireDetectionCandidate
    history: FireDetectionEventHistory | None
    as_of: datetime

    def __post_init__(self) -> None:
        if self.label != int(self.latent_subtype.is_fire):
            raise ValueError("label must equal the ground truth of latent_subtype.")
        if not isinstance(self.candidate, FireDetectionCandidate):
            raise ValueError(f"candidate must be a FireDetectionCandidate, got {self.candidate!r}")
        if self.history is not None and not isinstance(self.history, FireDetectionEventHistory):
            raise ValueError(f"history must be a FireDetectionEventHistory or None, got {self.history!r}")
        if not isinstance(self.as_of, datetime) or self.as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {self.as_of!r}")
        if (self.pair_id is None) != (self.pair_type is None):
            raise ValueError("pair_id and pair_type must both be set or both be None.")


@dataclass
class _EvidenceIds:
    satellite: int = 0
    news: int = 0

    def next_satellite(self) -> int:
        self.satellite += 1
        return self.satellite

    def next_news(self) -> int:
        self.news += 1
        return self.news


@dataclass
class _Source:
    subtype: LatentSubtypeV5
    thermal: _ThermalProfile
    latitude: float
    longitude: float
    log_frp_current: float
    drift: float
    spread_km: float
    jitter_km: float
    detect_probability: float
    news_profile: str


class FireDetectionTrainingDataGeneratorV5:
    """Generate a reproducible synthetic labeled Fire Detection V5 dataset."""

    def __init__(self, seed: int = DEFAULT_TRAINING_DATA_SEED_V5) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        self._seed = seed

    # --- plan ---------------------------------------------------------------------------------

    def generate(self, num_samples: int = DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V5) -> tuple[FireDetectionTrainingSampleV5, ...]:
        """Return num_samples rows, ~50/50 by label INSIDE every regime, spread over many environments."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < len(_REGIME_WEIGHTS) * 2:
            raise ValueError(f"num_samples must be an integer >= {len(_REGIME_WEIGHTS) * 2}, got {num_samples!r}")

        units = self._plan_units(num_samples)
        environment_count = max(4, round(num_samples / ROWS_PER_ENVIRONMENT))
        environments = {i: _Environment.draw(self._seed, i) for i in range(1, environment_count + 1)}

        samples: list[FireDetectionTrainingSampleV5] = []
        for unit_index, unit in enumerate(units):
            environment = environments[(unit_index % environment_count) + 1]
            rng = random.Random(f"ecoguard-fire-detection-v5:{self._seed}:{unit_index}")
            samples.extend(self._generate_unit(rng, unit, environment, first_sample_id=len(samples) + 1))
        return tuple(samples)

    def _plan_units(self, num_samples: int) -> list[_Unit]:
        # Largest-remainder allocation of rows to regimes (each regime gets an even-friendly share).
        raw = {regime: num_samples * weight for regime, weight in _REGIME_WEIGHTS.items()}
        counts = {regime: int(value) for regime, value in raw.items()}
        leftovers = sorted(raw, key=lambda regime: (raw[regime] - counts[regime]), reverse=True)
        for regime in leftovers[: num_samples - sum(counts.values())]:
            counts[regime] += 1

        units: list[_Unit] = []
        next_pair_id = 1
        for index, regime in enumerate(_REGIME_WEIGHTS):
            count = counts[regime]
            pairs = int(round(FRACTION_OF_ROWS_IN_PAIRS * count / 2))
            singles = count - 2 * pairs
            for _ in range(pairs):
                units.append(_Unit(regime, next_pair_id, None))
                next_pair_id += 1
            fire_singles = singles // 2 + (singles % 2 if index % 2 == 0 else 0)  # alternate the odd row
            units.extend(_Unit(regime, None, 1) for _ in range(fire_singles))
            units.extend(_Unit(regime, None, 0) for _ in range(singles - fire_singles))
        random.Random(f"ecoguard-fire-detection-v5-plan:{self._seed}").shuffle(units)
        return units

    # --- one unit (a single row or a pair) --------------------------------------------------------

    def _generate_unit(
        self, rng: random.Random, unit: _Unit, environment: _Environment, first_sample_id: int
    ) -> list[FireDetectionTrainingSampleV5]:
        structure = self._draw_structure(rng, unit.regime, environment)
        if unit.pair_id is None:
            labels = [unit.label]
        else:
            labels = [1, 0] if rng.random() < 0.5 else [0, 1]
        samples = []
        for offset, label in enumerate(labels):
            member_rng = random.Random(f"{rng.random():.17f}:{offset}")
            samples.append(
                self._generate_row(
                    member_rng,
                    sample_id=first_sample_id + offset,
                    unit=unit,
                    label=label,
                    environment=environment,
                    structure=structure,
                )
            )
        return samples

    def _draw_structure(self, rng: random.Random, regime: RegimeV5, environment: _Environment) -> _Structure:
        night = rng.random() < environment.night_share
        hours = [h for h in range(24) if _is_night(datetime(2025, 1, 1, h, tzinfo=timezone.utc)) == night]
        day_offset = rng.randrange(_BASE_TIME_RANGE_DAYS)
        anchor = _BASE_TIME_START + timedelta(days=day_offset, hours=rng.choice(hours), minutes=rng.randrange(60))
        news_only = regime is RegimeV5.NEWS_LED and rng.random() < 0.40
        sparse_news = regime is RegimeV5.SPARSE_EARLY_EVIDENCE and rng.random() < 0.20
        persistent_news = regime is RegimeV5.PERSISTENT_THERMAL and rng.random() < 0.35

        previous_times: list[datetime] = []
        previous_covered: list[bool] = []
        if regime is RegimeV5.PERSISTENT_THERMAL:
            target = _weighted_choice(rng, {1: 0.20, 2: 0.25, 3: 0.22, 4: 0.18, 5: 0.15})
            moment = anchor
            horizon = FIRE_EVENT_EVIDENCE_HISTORY_HOURS - _HISTORY_MARGIN_HOURS
            for _ in range(target):
                moment = moment - timedelta(hours=environment.revisit_hours * rng.uniform(0.8, 1.25))
                if (anchor - moment) > timedelta(hours=horizon):
                    break
                previous_times.append(moment)
            if not previous_times:
                previous_times.append(anchor - timedelta(hours=environment.revisit_hours))
            previous_times.reverse()
            previous_covered = [rng.random() < environment.coverage for _ in previous_times]
            if not any(previous_covered):
                previous_covered[rng.randrange(len(previous_covered))] = True  # a persistent row has a previous look
        return _Structure(
            anchor=anchor,
            night=night,
            day_night_known=rng.random() >= environment.day_night_unknown_rate,
            previous_times=tuple(previous_times),
            previous_covered=tuple(previous_covered),
            news_only=news_only,
            sparse_mode_news=sparse_news,
            persistent_news=persistent_news,
        )

    # --- one row ----------------------------------------------------------------------------------

    def _generate_row(
        self,
        rng: random.Random,
        sample_id: int,
        unit: _Unit,
        label: int,
        environment: _Environment,
        structure: _Structure,
    ) -> FireDetectionTrainingSampleV5:
        regime = unit.regime
        subtype = self._choose_subtype(rng, regime, label, structure, paired=unit.pair_id is not None)
        source = self._draw_source(rng, subtype, environment)
        ids = _EvidenceIds()
        anchor = structure.anchor + timedelta(minutes=rng.randint(-10, 10)) if unit.pair_id is not None else structure.anchor
        shift = anchor - structure.anchor

        satellite: list[FireDetectionEvidence] = []
        news: list[FireDetectionEvidence] = []
        history_items: list[FireDetectionEvidence] = []

        has_satellite = not (
            structure.news_only or (regime is RegimeV5.SPARSE_EARLY_EVIDENCE and structure.sparse_mode_news)
        )
        if has_satellite:
            satellite = self._current_pass(rng, regime, source, environment, structure, anchor, ids)

        if regime is RegimeV5.PERSISTENT_THERMAL:
            history_items = self._history_passes(rng, source, environment, structure, shift, ids)

        news = self._current_news(rng, regime, source, structure, anchor, ids)

        candidate_items = tuple(satellite) + tuple(news)
        candidate = FireDetectionCandidate(candidate_items)  # the runtime's own validity rules
        latest = max(item.observed_at for item in candidate.evidence)
        as_of = latest + timedelta(minutes=rng.randint(0, _AS_OF_DELAY_MAX_MINUTES))

        history = None
        if history_items:
            evidence = list(history_items)
            if rng.random() < 0.30:  # the event already has this candidate's hotspots attached: must dedupe
                evidence.extend(satellite)
            history = FireDetectionEventHistory(
                fire_event_id=sample_id,
                as_of=as_of,
                window_start=as_of - timedelta(hours=FIRE_EVENT_EVIDENCE_HISTORY_HOURS),
                evidence=tuple(evidence),
                satellite_pass_gap_minutes=SATELLITE_PASS_GAP_MINUTES,
            )
        return FireDetectionTrainingSampleV5(
            sample_id=sample_id,
            seed=self._seed,
            environment_id=environment.environment_id,
            regime=regime,
            latent_subtype=subtype,
            pair_id=unit.pair_id,
            pair_type=PAIR_TYPE_BY_REGIME[regime] if unit.pair_id is not None else None,
            label=label,
            candidate=candidate,
            history=history,
            as_of=as_of,
        )

    def _choose_subtype(
        self, rng: random.Random, regime: RegimeV5, label: int, structure: _Structure, paired: bool
    ) -> LatentSubtypeV5:
        news_only = structure.news_only or (regime is RegimeV5.SPARSE_EARLY_EVIDENCE and structure.sparse_mode_news)
        if label == 1:
            mix = {
                RegimeV5.SATELLITE_ONLY_SINGLE_PASS: _FIRE_MIX_SMALL,
                RegimeV5.MULTI_PIXEL_SINGLE_PASS: _FIRE_MIX_MULTI,
                RegimeV5.SPARSE_EARLY_EVIDENCE: _FIRE_MIX_SPARSE,
            }.get(regime, _FIRE_MIX_DEFAULT)
            return _weighted_choice(rng, mix)

        if paired and regime in _PAIR_NO_FIRE_COUNTERPART and rng.random() < 0.8:
            return _weighted_choice(rng, _PAIR_NO_FIRE_COUNTERPART[regime])
        if regime is RegimeV5.NEWS_LED:
            mix = _NO_FIRE_MIX_NEWS_ONLY if news_only else _NO_FIRE_MIX_MIXED
            if paired:
                mix = {LatentSubtypeV5.FALSE_OR_RUMOUR_REPORT: 1.0} if rng.random() < 0.75 else mix
        else:
            mix = {
                RegimeV5.SATELLITE_ONLY_SINGLE_PASS: _NO_FIRE_MIX_SATELLITE,
                RegimeV5.MULTI_PIXEL_SINGLE_PASS: _NO_FIRE_MIX_MULTI,
                RegimeV5.SATELLITE_NEWS: _NO_FIRE_MIX_MIXED,
                RegimeV5.PERSISTENT_THERMAL: _NO_FIRE_MIX_PERSISTENT,
                RegimeV5.SPARSE_EARLY_EVIDENCE: (
                    _NO_FIRE_MIX_SPARSE_NEWS if news_only else _NO_FIRE_MIX_SPARSE_SATELLITE
                ),
            }[regime]
        return _weighted_choice(rng, mix)

    # --- latent source -----------------------------------------------------------------------------

    def _draw_source(self, rng: random.Random, subtype: LatentSubtypeV5, environment: _Environment) -> _Source:
        thermal_subtype = _THERMAL_PROFILE_OF_FALSE_REPORT if subtype is LatentSubtypeV5.FALSE_OR_RUMOUR_REPORT else subtype
        profile = _THERMAL_PROFILES[thermal_subtype]
        radius = 6.0 * math.sqrt(rng.random())
        bearing = rng.uniform(0.0, 2.0 * math.pi)
        latitude, longitude = self._offset(environment.latitude, environment.longitude, radius, bearing)
        return _Source(
            subtype=subtype,
            thermal=profile,
            latitude=latitude,
            longitude=longitude,
            log_frp_current=rng.gauss(*profile.log_frp),
            drift=rng.gauss(*profile.drift),
            spread_km=rng.uniform(*profile.spread_km) * environment.pixel_scale,
            jitter_km=rng.uniform(*profile.jitter_km),
            detect_probability=rng.uniform(*profile.detect),
            news_profile=_NEWS_PROFILE_OF[subtype],
        )

    @staticmethod
    def _offset(latitude: float, longitude: float, distance_km: float, bearing_rad: float) -> tuple[float, float]:
        north = distance_km * math.cos(bearing_rad)
        east = distance_km * math.sin(bearing_rad)
        new_latitude = latitude + north / _KM_PER_DEGREE_LATITUDE
        new_longitude = longitude + east / (_KM_PER_DEGREE_LATITUDE * math.cos(math.radians(latitude)))
        return new_latitude, new_longitude

    # --- satellite -----------------------------------------------------------------------------------

    def _current_pass(
        self,
        rng: random.Random,
        regime: RegimeV5,
        source: _Source,
        environment: _Environment,
        structure: _Structure,
        anchor: datetime,
        ids: _EvidenceIds,
    ) -> list[FireDetectionEvidence]:
        weak = regime is RegimeV5.SPARSE_EARLY_EVIDENCE
        forced_pixels: int | None = 1 if regime in (RegimeV5.SATELLITE_ONLY_SINGLE_PASS, RegimeV5.SPARSE_EARLY_EVIDENCE) else None
        minimum_pixels = 2 if regime is RegimeV5.MULTI_PIXEL_SINGLE_PASS else 1
        maximum_pixels = 4 if regime is RegimeV5.NEWS_LED else 8 if regime is not RegimeV5.MULTI_PIXEL_SINGLE_PASS else 10
        scale = 0.5 if regime is RegimeV5.NEWS_LED else 0.8 if regime is RegimeV5.MULTI_PIXEL_SINGLE_PASS else 1.0
        day_night = ("N" if structure.night else "D") if structure.day_night_known else None
        return self._draw_pass(
            rng,
            source=source,
            environment=environment,
            pass_time=anchor,
            centre=(source.latitude, source.longitude),
            log_frp=source.log_frp_current + rng.gauss(0.0, source.thermal.walk),
            pixels=forced_pixels,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            lambda_scale=scale,
            day_night=day_night,
            weak=weak,
            ids=ids,
        )

    def _history_passes(
        self,
        rng: random.Random,
        source: _Source,
        environment: _Environment,
        structure: _Structure,
        shift: timedelta,
        ids: _EvidenceIds,
    ) -> list[FireDetectionEvidence]:
        previous = len(structure.previous_times)
        detected = [
            index
            for index in range(previous)
            if self._exists_at(rng, source.subtype, index, previous)
            and structure.previous_covered[index]
            and rng.random() < source.detect_probability
        ]
        if not detected:  # a persistent row always has at least one earlier detection
            candidates = [i for i in range(previous) if structure.previous_covered[i]]
            detected = [rng.choice(candidates)]
        evidence: list[FireDetectionEvidence] = []
        for index in detected:
            pass_time = structure.previous_times[index] + shift
            distance = previous - index  # opportunities before the current pass
            jitter_radius = min(_MAX_PASS_CENTRE_JITTER_KM, abs(rng.gauss(0.0, source.jitter_km)))
            centre = self._offset(source.latitude, source.longitude, jitter_radius, rng.uniform(0.0, 2.0 * math.pi))
            log_frp = source.log_frp_current - source.drift * distance + rng.gauss(0.0, source.thermal.walk)
            day_night = None
            if rng.random() >= environment.day_night_unknown_rate:
                day_night = "N" if _is_night(pass_time) else "D"
            evidence.extend(
                self._draw_pass(
                    rng,
                    source=source,
                    environment=environment,
                    pass_time=pass_time,
                    centre=centre,
                    log_frp=log_frp,
                    pixels=None,
                    minimum_pixels=1,
                    maximum_pixels=8,
                    lambda_scale=1.0,
                    day_night=day_night,
                    weak=False,
                    ids=ids,
                )
            )
        return evidence

    @staticmethod
    def _exists_at(rng: random.Random, subtype: LatentSubtypeV5, index: int, previous: int) -> bool:
        """Whether the latent source existed at earlier opportunity `index` (0 = oldest, `previous` = now)."""
        ago = previous - index
        if subtype is LatentSubtypeV5.EARLY_WILDFIRE:
            return ago <= 3
        if subtype is LatentSubtypeV5.ESTABLISHED_WILDFIRE:
            return ago <= 5 or rng.random() < 0.3
        if subtype is LatentSubtypeV5.CONTROLLED_OR_AGRICULTURAL_BURN:
            return ago <= 3
        return True  # large wildfire, industrial source and noise are not tied to a start time

    def _draw_pass(
        self,
        rng: random.Random,
        *,
        source: _Source,
        environment: _Environment,
        pass_time: datetime,
        centre: tuple[float, float],
        log_frp: float,
        pixels: int | None,
        minimum_pixels: int,
        maximum_pixels: int,
        lambda_scale: float,
        day_night: str | None,
        weak: bool,
        ids: _EvidenceIds,
    ) -> list[FireDetectionEvidence]:
        total_frp = 10.0 ** (log_frp + environment.frp_shift_dex)
        if pixels is None:
            lam = min(6.0, 0.25 * total_frp**0.6) * lambda_scale
            count = min(maximum_pixels, minimum_pixels + _poisson(rng, lam))
        else:
            count = pixels
        weights = [math.exp(rng.gauss(0.0, 0.55)) for _ in range(count)]
        weight_sum = sum(weights)

        items: list[FireDetectionEvidence] = []
        for weight in weights:
            pixel_frp = total_frp * weight / weight_sum if count > 1 else total_frp
            brightness, confidence = self._pixel_measurements(rng, source, environment, pixel_frp, day_night)
            if weak:  # rejection-sample a WEAK pixel from the same latent process (never forced by label)
                for _ in range(200):
                    if pixel_frp < 7.0 and confidence in ("low", "nominal"):
                        break
                    pixel_frp = 10.0 ** (rng.gauss(*source.thermal.log_frp) + environment.frp_shift_dex)
                    brightness, confidence = self._pixel_measurements(rng, source, environment, pixel_frp, day_night)
            radius = min(MAX_CANDIDATE_RADIUS_KM, abs(rng.gauss(0.0, source.spread_km)))
            latitude, longitude = self._offset(centre[0], centre[1], radius, rng.uniform(0.0, 2.0 * math.pi))
            observed_at = pass_time + timedelta(minutes=rng.randint(-3, 3))
            items.append(
                FireDetectionEvidence(
                    evidence_id=ids.next_satellite(),
                    evidence_type=FireEvidenceType.SATELLITE,
                    latitude=latitude,
                    longitude=longitude,
                    observed_at=observed_at,
                    satellite_confidence=confidence,
                    satellite_frp=None if rng.random() < environment.frp_missing_rate else round(pixel_frp, 2),
                    satellite_brightness=(
                        None if rng.random() < environment.brightness_missing_rate else round(brightness, 1)
                    ),
                    satellite_day_night=day_night,
                    satellite_name=SATELLITE_NAME,
                    satellite_instrument=SATELLITE_INSTRUMENT,
                )
            )
        return items

    @staticmethod
    def _pixel_measurements(
        rng: random.Random,
        source: _Source,
        environment: _Environment,
        pixel_frp: float,
        day_night: str | None,
    ) -> tuple[float, str]:
        log_term = math.log10(1.0 + pixel_frp)
        brightness = (
            environment.background_kelvin
            + 18.0 * log_term
            + source.thermal.brightness_bias
            + rng.gauss(0.0, source.thermal.brightness_noise)
        )
        brightness = max(290.0, min(367.0, brightness))
        score = (
            1.1 * (log_term - 0.9)
            + rng.gauss(0.0, 0.55)
            + source.thermal.confidence_bias
            + environment.confidence_shift
            + (-0.15 if day_night == "D" else 0.0)  # day-time glint makes weak hotspots less certain
        )
        confidence = "low" if score < -0.25 else "nominal" if score < 0.55 else "high"
        return brightness, confidence

    # --- news ------------------------------------------------------------------------------------------

    def _current_news(
        self,
        rng: random.Random,
        regime: RegimeV5,
        source: _Source,
        structure: _Structure,
        anchor: datetime,
        ids: _EvidenceIds,
    ) -> list[FireDetectionEvidence]:
        fire = source.subtype.is_fire
        # Offsets (minutes) relative to `anchor` (the current pass, or the first report when no hotspot).
        if regime in (RegimeV5.SATELLITE_ONLY_SINGLE_PASS, RegimeV5.MULTI_PIXEL_SINGLE_PASS):
            return []
        if regime is RegimeV5.SPARSE_EARLY_EVIDENCE:
            if not structure.sparse_mode_news:
                return []
            return self._news_items(rng, source, anchor, [0], ids, weak=True)
        if regime is RegimeV5.PERSISTENT_THERMAL:
            if not structure.persistent_news:
                return []
            count = 1 + (rng.random() < 0.35)
            offsets = [self._clip_offset(rng.gauss(8.0, 25.0)) for _ in range(count)]
            return self._news_items(rng, source, anchor, offsets, ids)
        if regime is RegimeV5.NEWS_LED:
            count = _weighted_choice(rng, {1: 0.35, 2: 0.35, 3: 0.20, 4: 0.10})
            if structure.news_only:
                offsets = [0] + sorted(rng.randint(0, 45) for _ in range(count - 1))
            else:  # the first report precedes the hotspot
                lead = rng.randint(6, _MAX_NEWS_OFFSET_MINUTES)
                offsets = [-lead] + sorted(rng.randint(-lead, 20) for _ in range(count - 1))
            return self._news_items(rng, source, anchor, offsets, ids, repeated=(not fire and count >= 2 and rng.random() < 0.6))
        # SATELLITE_NEWS: the report may come before or after the hotspots
        count = _weighted_choice(rng, {1: 0.5, 2: 0.3, 3: 0.2})
        centre = 12.0 if fire else 4.0
        offsets = [self._clip_offset(rng.gauss(centre, 26.0)) for _ in range(count)]
        return self._news_items(rng, source, anchor, offsets, ids)

    @staticmethod
    def _clip_offset(minutes: float) -> int:
        return int(round(max(-_MAX_NEWS_OFFSET_MINUTES, min(_MAX_NEWS_OFFSET_MINUTES, minutes))))

    def _news_items(
        self,
        rng: random.Random,
        source: _Source,
        anchor: datetime,
        offsets: list[int],
        ids: _EvidenceIds,
        weak: bool = False,
        repeated: bool = False,
    ) -> list[FireDetectionEvidence]:
        """One report per offset. `repeated` reports echo one another (a rumour repeated); real fires echo too."""
        profile = _NEWS_PROFILES[source.news_profile]
        echo = 0.7 if repeated else 0.35
        items: list[FireDetectionEvidence] = []
        previous: NewsWildfireSignalStrength | None = None
        for position, offset in enumerate(offsets):
            while True:
                if position > 0 and rng.random() < echo:
                    strength = previous
                else:
                    strength = _NEWS_CHOICES[_weighted_choice(rng, dict(enumerate(profile)))]
                if not weak or strength in _WEAK_NEWS:
                    break
            previous = strength
            radius = min(MAX_CANDIDATE_RADIUS_KM, abs(rng.gauss(0.0, 1.0)))  # geocoding is coarse
            latitude, longitude = self._offset(source.latitude, source.longitude, radius, rng.uniform(0.0, 2.0 * math.pi))
            items.append(
                FireDetectionEvidence(
                    evidence_id=ids.next_news(),
                    evidence_type=FireEvidenceType.NEWS,
                    latitude=latitude,
                    longitude=longitude,
                    observed_at=anchor + timedelta(minutes=offset),
                    news_wildfire_signal_strength=strength,
                )
            )
        return items
