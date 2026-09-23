"""Deterministic synthetic labeled dataset generator for Fire Detection ML V3.

Sibling to fire_detection_training_data_generator.py (V1/V2), left completely
untouched - V3 is a new, separately versioned generator producing evidence
enriched with satellite FRP/brightness and a news wildfire-signal-strength,
using the Task 4 FireDetectionEvidence fields. No network calls, no LLM
calls, no Neon dependency - this is a pure, deterministic, synthetic
generator; the "news signal" values here are chosen by scenario family, not
produced by the real LLM (see backend/docs/
fire_detection_feature_representation_v3.md, "real runtime signal vs
synthetic training signal").

Ground truth (label) comes from the synthetic scenario family, never from
FireDetectionCalculator - same principle as V1/V2.

Every generated sample is a valid FireDetectionCandidate: evidence items in a
sample are placed within a bounded radius of a shared center point and a
bounded time window of a shared base timestamp (max_radius_km <= 2.2,
max_time_spread_minutes <= 55), the same safety margins V1/V2 use, so every
pair correlates directly under FireDetectionCandidate's rules.

FRP/brightness ranges and news-signal pools deliberately OVERLAP between
label=1 and label=0 scenario families - see each family's comment - so no
single physical measurement or news signal strength perfectly separates the
two classes. This specifically targets the V2 "twin family" pairs
(news_only_before_satellite/news_rumor_no_fire and
satellite_only_before_news/high_confidence_satellite_false_positive), which
this generator gives genuinely different (but still overlapping)
distributions rather than the identical feature vectors V1/V2 evidence gave
them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
import random

from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

DEFAULT_TRAINING_DATA_SEED_V3 = 42
DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V3 = 4000

_ACTIVE_ISRAEL_LATITUDE_RANGE = (29.6, 33.2)
_ACTIVE_ISRAEL_LONGITUDE_RANGE = (34.3, 35.7)
_BASE_TIME_RANGE_DAYS = 365
_BASE_TIME_START = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

# NONE_SIGNAL is a sentinel used inside a family's news_signal_pool to mean
# "no reliable LLM analysis available" (FireDetectionEvidence.
# news_wildfire_signal_strength=None) - distinct from
# NewsWildfireSignalStrength.NONE ("analyzed, no signal found").
UNKNOWN_SIGNAL = None


class ScenarioArchetype(Enum):
    """Higher-level evidence-composition grouping across scenario families (Part 26).

    Unlike scenario_family, every archetype here contains BOTH labels - it
    groups families by evidence *shape* (what source types/counts are
    present), not by ground truth, so it is a meaningful axis for a
    leave-one-archetype-out generalization check. Metadata only - never a
    feature.
    """

    NEWS_ONLY = "news_only"
    SATELLITE_ONLY = "satellite_only"
    SATELLITE_NEWS = "satellite_news"
    MULTI_SATELLITE = "multi_satellite"
    MIXED_CONFIDENCE = "mixed_confidence"


@dataclass(frozen=True)
class _ScenarioFamilyV3:
    """One V3 synthetic scenario family: evidence generation rules and ground-truth label."""

    name: str
    label: int
    archetype: ScenarioArchetype
    satellite_count_range: tuple[int, int]
    satellite_confidence_pool: tuple[str, ...]
    frp_range: tuple[float, float] | None
    frp_missing_probability: float
    brightness_range: tuple[float, float] | None
    brightness_missing_probability: float
    news_count_range: tuple[int, int]
    news_signal_pool: tuple[NewsWildfireSignalStrength | None, ...]
    max_radius_km: float
    max_time_spread_minutes: float


@dataclass(frozen=True)
class FireDetectionTrainingSampleV3:
    """One synthetic labeled Fire Detection V3 candidate.

    label=1 means the synthetic ground truth contains an active wildfire;
    label=0 means it does not. sample_id/scenario_family/scenario_archetype
    are metadata for dataset inspection only and must never be used as ML
    features.
    """

    sample_id: int
    scenario_family: str
    scenario_archetype: ScenarioArchetype
    evidence: tuple[FireDetectionEvidence, ...]
    label: int


# --- POSITIVE (label=1): synthetic ground truth contains an active wildfire ---
# FRP/brightness ranges and news-signal pools are chosen to overlap with the
# negative families below - see each negative family's comment for the
# specific counterpart it is designed to overlap with.
_POSITIVE_FAMILIES_V3: tuple[_ScenarioFamilyV3, ...] = (
    _ScenarioFamilyV3(
        name="high_confidence_satellite_with_news",
        label=1,
        archetype=ScenarioArchetype.SATELLITE_NEWS,
        satellite_count_range=(1, 3),
        satellite_confidence_pool=("high",),
        frp_range=(30.0, 150.0),
        frp_missing_probability=0.15,
        brightness_range=(310.0, 400.0),
        brightness_missing_probability=0.15,
        news_count_range=(1, 2),
        news_signal_pool=(NewsWildfireSignalStrength.MODERATE, NewsWildfireSignalStrength.STRONG, NewsWildfireSignalStrength.STRONG),
        max_radius_km=2.0,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamilyV3(
        name="nominal_satellite_with_news",
        label=1,
        archetype=ScenarioArchetype.SATELLITE_NEWS,
        satellite_count_range=(1, 2),
        satellite_confidence_pool=("nominal",),
        frp_range=(15.0, 80.0),
        frp_missing_probability=0.2,
        brightness_range=(300.0, 360.0),
        brightness_missing_probability=0.2,
        news_count_range=(1, 2),
        news_signal_pool=(NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE, NewsWildfireSignalStrength.STRONG),
        max_radius_km=2.0,
        max_time_spread_minutes=40.0,
    ),
    _ScenarioFamilyV3(
        name="multi_satellite_real_fire",
        label=1,
        archetype=ScenarioArchetype.MULTI_SATELLITE,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("nominal", "nominal", "high"),
        frp_range=(20.0, 120.0),
        frp_missing_probability=0.2,
        brightness_range=(300.0, 380.0),
        brightness_missing_probability=0.2,
        news_count_range=(0, 0),
        news_signal_pool=(),
        max_radius_km=2.2,
        max_time_spread_minutes=45.0,
    ),
    # Twin target vs. high_confidence_satellite_false_positive (below): same
    # archetype (SATELLITE_ONLY), deliberately overlapping FRP/brightness.
    _ScenarioFamilyV3(
        name="satellite_only_before_news",
        label=1,
        archetype=ScenarioArchetype.SATELLITE_ONLY,
        satellite_count_range=(1, 2),
        satellite_confidence_pool=("nominal", "high"),
        frp_range=(10.0, 100.0),
        frp_missing_probability=0.35,
        brightness_range=(295.0, 370.0),
        brightness_missing_probability=0.3,
        news_count_range=(0, 0),
        news_signal_pool=(),
        max_radius_km=1.5,
        max_time_spread_minutes=20.0,
    ),
    # Twin target vs. news_rumor_no_fire (below): same archetype (NEWS_ONLY),
    # deliberately overlapping news-signal pool (a real early fire can read
    # WEAK; occasionally MODERATE).
    _ScenarioFamilyV3(
        name="news_only_before_satellite",
        label=1,
        archetype=ScenarioArchetype.NEWS_ONLY,
        satellite_count_range=(0, 0),
        satellite_confidence_pool=(),
        frp_range=None,
        frp_missing_probability=1.0,
        brightness_range=None,
        brightness_missing_probability=1.0,
        news_count_range=(1, 2),
        news_signal_pool=(NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE, UNKNOWN_SIGNAL),
        max_radius_km=1.0,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamilyV3(
        name="multi_satellite_plus_multi_news",
        label=1,
        archetype=ScenarioArchetype.SATELLITE_NEWS,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("nominal", "high"),
        frp_range=(20.0, 130.0),
        frp_missing_probability=0.15,
        brightness_range=(300.0, 390.0),
        brightness_missing_probability=0.15,
        news_count_range=(2, 3),
        news_signal_pool=(NewsWildfireSignalStrength.MODERATE, NewsWildfireSignalStrength.STRONG, NewsWildfireSignalStrength.STRONG),
        max_radius_km=2.2,
        max_time_spread_minutes=50.0,
    ),
    _ScenarioFamilyV3(
        name="mixed_confidence_satellite_real_fire",
        label=1,
        archetype=ScenarioArchetype.MIXED_CONFIDENCE,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("low", "nominal", "high"),
        frp_range=(10.0, 110.0),
        frp_missing_probability=0.25,
        brightness_range=(295.0, 380.0),
        brightness_missing_probability=0.25,
        news_count_range=(0, 2),
        news_signal_pool=(NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE, NewsWildfireSignalStrength.STRONG),
        max_radius_km=2.2,
        max_time_spread_minutes=45.0,
    ),
    _ScenarioFamilyV3(
        name="high_low_satellite_with_news_real_fire",
        label=1,
        archetype=ScenarioArchetype.MIXED_CONFIDENCE,
        satellite_count_range=(2, 3),
        satellite_confidence_pool=("high", "low"),
        frp_range=(15.0, 130.0),
        frp_missing_probability=0.2,
        brightness_range=(300.0, 390.0),
        brightness_missing_probability=0.2,
        news_count_range=(1, 2),
        news_signal_pool=(NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE, NewsWildfireSignalStrength.STRONG),
        max_radius_km=2.0,
        max_time_spread_minutes=35.0,
    ),
    _ScenarioFamilyV3(
        name="nominal_low_satellite_with_news_real_fire",
        label=1,
        archetype=ScenarioArchetype.MIXED_CONFIDENCE,
        satellite_count_range=(2, 3),
        satellite_confidence_pool=("nominal", "low"),
        frp_range=(10.0, 90.0),
        frp_missing_probability=0.25,
        brightness_range=(295.0, 370.0),
        brightness_missing_probability=0.25,
        news_count_range=(1, 2),
        news_signal_pool=(NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE),
        max_radius_km=2.0,
        max_time_spread_minutes=35.0,
    ),
    _ScenarioFamilyV3(
        name="multi_confidence_persistent_real_fire",
        label=1,
        archetype=ScenarioArchetype.MULTI_SATELLITE,
        satellite_count_range=(3, 5),
        satellite_confidence_pool=("low", "nominal", "high"),
        frp_range=(15.0, 120.0),
        frp_missing_probability=0.2,
        brightness_range=(300.0, 385.0),
        brightness_missing_probability=0.2,
        news_count_range=(0, 2),
        news_signal_pool=(NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE, NewsWildfireSignalStrength.STRONG),
        max_radius_km=2.2,
        max_time_spread_minutes=50.0,
    ),
)

# --- NEGATIVE (label=0): synthetic ground truth contains no active wildfire ---
_NEGATIVE_FAMILIES_V3: tuple[_ScenarioFamilyV3, ...] = (
    _ScenarioFamilyV3(
        name="low_confidence_satellite_noise",
        label=0,
        archetype=ScenarioArchetype.SATELLITE_ONLY,
        satellite_count_range=(1, 3),
        satellite_confidence_pool=("low",),
        frp_range=(5.0, 40.0),
        frp_missing_probability=0.4,
        brightness_range=(290.0, 330.0),
        brightness_missing_probability=0.4,
        news_count_range=(0, 0),
        news_signal_pool=(),
        max_radius_km=2.2,
        max_time_spread_minutes=40.0,
    ),
    _ScenarioFamilyV3(
        name="nominal_satellite_false_heat_source",
        label=0,
        archetype=ScenarioArchetype.SATELLITE_ONLY,
        satellite_count_range=(1, 2),
        satellite_confidence_pool=("nominal",),
        frp_range=(10.0, 60.0),
        frp_missing_probability=0.3,
        brightness_range=(295.0, 340.0),
        brightness_missing_probability=0.3,
        news_count_range=(0, 0),
        news_signal_pool=(),
        max_radius_km=1.0,
        max_time_spread_minutes=15.0,
    ),
    # Twin target vs. satellite_only_before_news: a false positive (e.g. a
    # gas flare or industrial fire) can still register HIGH confidence with
    # substantial FRP - overlapping deliberately with the real-fire range.
    _ScenarioFamilyV3(
        name="high_confidence_satellite_false_positive",
        label=0,
        archetype=ScenarioArchetype.SATELLITE_ONLY,
        satellite_count_range=(1, 1),
        satellite_confidence_pool=("high",),
        frp_range=(20.0, 140.0),
        frp_missing_probability=0.25,
        brightness_range=(305.0, 395.0),
        brightness_missing_probability=0.25,
        news_count_range=(0, 0),
        news_signal_pool=(),
        max_radius_km=0.3,
        max_time_spread_minutes=0.0,
    ),
    # Twin target vs. news_only_before_satellite: an unverified/false rumor
    # is usually NONE/WEAK, but occasionally reads MODERATE or even STRONG
    # (misleading wording) - "strong wording != guaranteed truth".
    _ScenarioFamilyV3(
        name="news_rumor_no_fire",
        label=0,
        archetype=ScenarioArchetype.NEWS_ONLY,
        satellite_count_range=(0, 0),
        satellite_confidence_pool=(),
        frp_range=None,
        frp_missing_probability=1.0,
        brightness_range=None,
        brightness_missing_probability=1.0,
        news_count_range=(1, 1),
        news_signal_pool=(NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE, NewsWildfireSignalStrength.STRONG, UNKNOWN_SIGNAL),
        max_radius_km=0.3,
        max_time_spread_minutes=0.0,
    ),
    _ScenarioFamilyV3(
        name="multiple_news_same_false_alarm",
        label=0,
        archetype=ScenarioArchetype.NEWS_ONLY,
        satellite_count_range=(0, 0),
        satellite_confidence_pool=(),
        frp_range=None,
        frp_missing_probability=1.0,
        brightness_range=None,
        brightness_missing_probability=1.0,
        news_count_range=(2, 4),
        news_signal_pool=(NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE),
        max_radius_km=1.0,
        max_time_spread_minutes=45.0,
    ),
    _ScenarioFamilyV3(
        name="correlated_satellite_news_false_alarm",
        label=0,
        archetype=ScenarioArchetype.SATELLITE_NEWS,
        satellite_count_range=(1, 2),
        satellite_confidence_pool=("low", "nominal"),
        frp_range=(5.0, 70.0),
        frp_missing_probability=0.3,
        brightness_range=(290.0, 350.0),
        brightness_missing_probability=0.3,
        news_count_range=(1, 2),
        news_signal_pool=(NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE, NewsWildfireSignalStrength.MODERATE),
        max_radius_km=2.0,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamilyV3(
        name="persistent_non_wildfire_heat_source",
        label=0,
        archetype=ScenarioArchetype.MULTI_SATELLITE,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("low", "nominal"),
        frp_range=(10.0, 90.0),
        frp_missing_probability=0.25,
        brightness_range=(295.0, 360.0),
        brightness_missing_probability=0.25,
        news_count_range=(0, 0),
        news_signal_pool=(),
        max_radius_km=2.2,
        max_time_spread_minutes=50.0,
    ),
    _ScenarioFamilyV3(
        name="mixed_confidence_non_fire_heat_source",
        label=0,
        archetype=ScenarioArchetype.MIXED_CONFIDENCE,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("low", "nominal", "high"),
        frp_range=(10.0, 110.0),
        frp_missing_probability=0.3,
        brightness_range=(295.0, 375.0),
        brightness_missing_probability=0.3,
        news_count_range=(0, 0),
        news_signal_pool=(),
        max_radius_km=1.5,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamilyV3(
        name="high_low_satellite_false_alarm",
        label=0,
        archetype=ScenarioArchetype.MIXED_CONFIDENCE,
        satellite_count_range=(2, 3),
        satellite_confidence_pool=("high", "low"),
        frp_range=(15.0, 120.0),
        frp_missing_probability=0.25,
        brightness_range=(300.0, 385.0),
        brightness_missing_probability=0.25,
        news_count_range=(0, 1),
        news_signal_pool=(NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.WEAK),
        max_radius_km=1.8,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamilyV3(
        name="mixed_satellite_with_false_news",
        label=0,
        archetype=ScenarioArchetype.MIXED_CONFIDENCE,
        satellite_count_range=(1, 3),
        satellite_confidence_pool=("low", "nominal", "high"),
        frp_range=(10.0, 100.0),
        frp_missing_probability=0.3,
        brightness_range=(295.0, 370.0),
        brightness_missing_probability=0.3,
        news_count_range=(1, 2),
        news_signal_pool=(NewsWildfireSignalStrength.NONE, NewsWildfireSignalStrength.WEAK, NewsWildfireSignalStrength.MODERATE),
        max_radius_km=2.0,
        max_time_spread_minutes=35.0,
    ),
)


class FireDetectionTrainingDataGeneratorV3:
    """Generate a reproducible synthetic labeled Fire Detection V3 dataset.

    Determinism: the same seed and num_samples always produce equivalent
    output. Randomness is drawn only from an explicitly seeded
    random.Random instance - no uncontrolled global randomness.
    """

    def __init__(self, seed: int = DEFAULT_TRAINING_DATA_SEED_V3) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        self._seed = seed

    def generate(
        self,
        num_samples: int = DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V3,
    ) -> tuple[FireDetectionTrainingSampleV3, ...]:
        """Return num_samples deterministic labeled samples, ~50/50 balanced by label."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer, got {num_samples!r}")

        rng = random.Random(self._seed)
        labels = self._balanced_labels(num_samples, rng)

        satellite_id_counter = 1
        news_id_counter = 1
        samples: list[FireDetectionTrainingSampleV3] = []
        for sample_index, label in enumerate(labels):
            family = rng.choice(_POSITIVE_FAMILIES_V3 if label == 1 else _NEGATIVE_FAMILIES_V3)
            evidence, satellite_id_counter, news_id_counter = self._generate_evidence(
                rng=rng,
                family=family,
                next_satellite_id=satellite_id_counter,
                next_news_id=news_id_counter,
            )
            samples.append(
                FireDetectionTrainingSampleV3(
                    sample_id=sample_index + 1,
                    scenario_family=family.name,
                    scenario_archetype=family.archetype,
                    evidence=evidence,
                    label=label,
                )
            )

        return tuple(samples)

    @staticmethod
    def _balanced_labels(num_samples: int, rng: random.Random) -> list[int]:
        positive_count = num_samples // 2
        negative_count = num_samples - positive_count
        labels = [1] * positive_count + [0] * negative_count
        rng.shuffle(labels)
        return labels

    @classmethod
    def _generate_evidence(
        cls,
        rng: random.Random,
        family: _ScenarioFamilyV3,
        next_satellite_id: int,
        next_news_id: int,
    ) -> tuple[tuple[FireDetectionEvidence, ...], int, int]:
        center_latitude, center_longitude = cls._random_center(rng)
        base_time = cls._random_base_time(rng)

        satellite_count = rng.randint(*family.satellite_count_range)
        news_count = rng.randint(*family.news_count_range)

        evidence_items: list[FireDetectionEvidence] = []
        for _ in range(satellite_count):
            latitude, longitude = cls._jittered_point(rng, center_latitude, center_longitude, family.max_radius_km)
            observed_at = cls._jittered_time(rng, base_time, family.max_time_spread_minutes)
            evidence_items.append(
                FireDetectionEvidence(
                    evidence_id=next_satellite_id,
                    evidence_type=FireEvidenceType.SATELLITE,
                    latitude=latitude,
                    longitude=longitude,
                    observed_at=observed_at,
                    satellite_confidence=rng.choice(family.satellite_confidence_pool),
                    satellite_frp=cls._sample_measurement(rng, family.frp_range, family.frp_missing_probability),
                    satellite_brightness=cls._sample_measurement(
                        rng, family.brightness_range, family.brightness_missing_probability
                    ),
                )
            )
            next_satellite_id += 1

        for _ in range(news_count):
            latitude, longitude = cls._jittered_point(rng, center_latitude, center_longitude, family.max_radius_km)
            observed_at = cls._jittered_time(rng, base_time, family.max_time_spread_minutes)
            evidence_items.append(
                FireDetectionEvidence(
                    evidence_id=next_news_id,
                    evidence_type=FireEvidenceType.NEWS,
                    latitude=latitude,
                    longitude=longitude,
                    observed_at=observed_at,
                    news_wildfire_signal_strength=rng.choice(family.news_signal_pool),
                )
            )
            next_news_id += 1

        return tuple(evidence_items), next_satellite_id, next_news_id

    @staticmethod
    def _sample_measurement(
        rng: random.Random,
        value_range: tuple[float, float] | None,
        missing_probability: float,
    ) -> float | None:
        if value_range is None or rng.random() < missing_probability:
            return None
        return round(rng.uniform(*value_range), 2)

    @staticmethod
    def _random_center(rng: random.Random) -> tuple[float, float]:
        return (
            rng.uniform(*_ACTIVE_ISRAEL_LATITUDE_RANGE),
            rng.uniform(*_ACTIVE_ISRAEL_LONGITUDE_RANGE),
        )

    @staticmethod
    def _random_base_time(rng: random.Random) -> datetime:
        offset_minutes = rng.uniform(0.0, _BASE_TIME_RANGE_DAYS * 24 * 60)
        return _BASE_TIME_START + timedelta(minutes=offset_minutes)

    @staticmethod
    def _jittered_point(
        rng: random.Random,
        center_latitude: float,
        center_longitude: float,
        max_radius_km: float,
    ) -> tuple[float, float]:
        if max_radius_km <= 0.0:
            return round(center_latitude, 6), round(center_longitude, 6)

        radius_km = rng.uniform(0.0, max_radius_km)
        bearing_radians = rng.uniform(0.0, 2.0 * math.pi)
        north_km = radius_km * math.cos(bearing_radians)
        east_km = radius_km * math.sin(bearing_radians)

        km_per_latitude_degree = 111.32
        latitude_delta = north_km / km_per_latitude_degree
        longitude_scale = km_per_latitude_degree * math.cos(math.radians(center_latitude))
        longitude_delta = east_km / longitude_scale
        return (
            round(center_latitude + latitude_delta, 6),
            round(center_longitude + longitude_delta, 6),
        )

    @staticmethod
    def _jittered_time(rng: random.Random, base_time: datetime, max_spread_minutes: float) -> datetime:
        if max_spread_minutes <= 0.0:
            return base_time
        return base_time + timedelta(minutes=rng.uniform(0.0, max_spread_minutes))
