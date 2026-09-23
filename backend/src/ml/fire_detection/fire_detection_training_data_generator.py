"""Deterministic synthetic labeled dataset generator for Fire Detection ML.

Ground truth (label) comes from the synthetic scenario family, NEVER from
FireDetectionCalculator - using the rule-based calculator's own output as a
label would only teach a classifier to reproduce the existing manually
configured weights/thresholds. This generator has no import of, or
dependency on, FireDetectionCalculator.

Every generated sample is a valid FireDetectionCandidate: all evidence items
in a sample are placed within a bounded radius of a shared center point and
a bounded time window of a shared base timestamp, so every pair correlates
directly (a strictly stronger, and therefore safe, condition than the
connected-component requirement FireDetectionCandidate enforces). Candidate
correlation itself remains FireDetectionEvidenceService/FireDetectionCandidate's
responsibility - this generator does not reimplement or loosen it.

Evidence source-type combinations intentionally overlap between the two
labels (e.g. satellite-only samples, news-only samples, and satellite+news
samples all occur for both label=1 and label=0) so the dataset cannot be
trivially separated by "evidence exists" or "which sources are present"
alone - see backend/docs/fire_detection_ml.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import random

from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType

DEFAULT_TRAINING_DATA_SEED = 42
DEFAULT_TRAINING_DATA_SAMPLE_COUNT = 2000
DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V2 = 3000

DATASET_VERSION_V1 = "v1"
DATASET_VERSION_V2 = "v2"
SUPPORTED_DATASET_VERSIONS = (DATASET_VERSION_V1, DATASET_VERSION_V2)

# Every family's max radius stays <= 2.2km so that two independently jittered
# points (each up to max_radius_km from a shared center) can never exceed
# ~4.4km apart - safely under FireDetectionCandidate's 5.0km correlation
# limit. Every family's max time spread stays <= 55 minutes, safely under
# the 60-minute correlation limit (both items are offset from the same base
# timestamp by no more than the spread, so their pairwise difference cannot
# exceed it).
_ACTIVE_ISRAEL_LATITUDE_RANGE = (29.6, 33.2)
_ACTIVE_ISRAEL_LONGITUDE_RANGE = (34.3, 35.7)
_BASE_TIME_RANGE_DAYS = 365
_BASE_TIME_START = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _ScenarioFamily:
    """One synthetic scenario family: how to generate evidence and its ground-truth label."""

    name: str
    label: int
    satellite_count_range: tuple[int, int]
    satellite_confidence_pool: tuple[str, ...]
    news_count_range: tuple[int, int]
    max_radius_km: float
    max_time_spread_minutes: float


@dataclass(frozen=True)
class FireDetectionTrainingSample:
    """One synthetic labeled Fire Detection candidate.

    label=1 means the synthetic ground truth contains an active wildfire;
    label=0 means it does not. sample_id/scenario_family are metadata for
    dataset inspection only and must never be used as ML features.
    """

    sample_id: int
    scenario_family: str
    evidence: tuple[FireDetectionEvidence, ...]
    label: int


# --- POSITIVE (label=1): synthetic ground truth contains an active wildfire ---
_POSITIVE_FAMILIES: tuple[_ScenarioFamily, ...] = (
    _ScenarioFamily(
        name="high_confidence_satellite_with_news",
        label=1,
        satellite_count_range=(1, 3),
        satellite_confidence_pool=("high",),
        news_count_range=(1, 2),
        max_radius_km=2.0,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamily(
        name="nominal_satellite_with_news",
        label=1,
        satellite_count_range=(1, 2),
        satellite_confidence_pool=("nominal",),
        news_count_range=(1, 2),
        max_radius_km=2.0,
        max_time_spread_minutes=40.0,
    ),
    _ScenarioFamily(
        name="multi_satellite_real_fire",
        label=1,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("nominal", "nominal", "high"),
        news_count_range=(0, 0),
        max_radius_km=2.2,
        max_time_spread_minutes=45.0,
    ),
    _ScenarioFamily(
        name="satellite_only_before_news",
        label=1,
        satellite_count_range=(1, 2),
        satellite_confidence_pool=("nominal", "high"),
        news_count_range=(0, 0),
        max_radius_km=1.5,
        max_time_spread_minutes=20.0,
    ),
    _ScenarioFamily(
        name="news_only_before_satellite",
        label=1,
        satellite_count_range=(0, 0),
        satellite_confidence_pool=(),
        news_count_range=(1, 2),
        max_radius_km=1.0,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamily(
        name="multi_satellite_plus_multi_news",
        label=1,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("nominal", "high"),
        news_count_range=(2, 3),
        max_radius_km=2.2,
        max_time_spread_minutes=50.0,
    ),
)

# --- NEGATIVE (label=0): synthetic ground truth contains no active wildfire ---
_NEGATIVE_FAMILIES: tuple[_ScenarioFamily, ...] = (
    _ScenarioFamily(
        name="low_confidence_satellite_noise",
        label=0,
        satellite_count_range=(1, 3),
        satellite_confidence_pool=("low",),
        news_count_range=(0, 0),
        max_radius_km=2.2,
        max_time_spread_minutes=40.0,
    ),
    _ScenarioFamily(
        name="nominal_satellite_false_heat_source",
        label=0,
        satellite_count_range=(1, 2),
        satellite_confidence_pool=("nominal",),
        news_count_range=(0, 0),
        max_radius_km=1.0,
        max_time_spread_minutes=15.0,
    ),
    _ScenarioFamily(
        name="high_confidence_satellite_false_positive",
        label=0,
        satellite_count_range=(1, 1),
        satellite_confidence_pool=("high",),
        news_count_range=(0, 0),
        max_radius_km=0.0,
        max_time_spread_minutes=0.0,
    ),
    _ScenarioFamily(
        name="news_rumor_no_fire",
        label=0,
        satellite_count_range=(0, 0),
        satellite_confidence_pool=(),
        news_count_range=(1, 1),
        max_radius_km=0.0,
        max_time_spread_minutes=0.0,
    ),
    _ScenarioFamily(
        name="multiple_news_same_false_alarm",
        label=0,
        satellite_count_range=(0, 0),
        satellite_confidence_pool=(),
        news_count_range=(2, 4),
        max_radius_km=1.0,
        max_time_spread_minutes=45.0,
    ),
    _ScenarioFamily(
        name="correlated_satellite_news_false_alarm",
        label=0,
        satellite_count_range=(1, 2),
        satellite_confidence_pool=("low", "nominal"),
        news_count_range=(1, 2),
        max_radius_km=2.0,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamily(
        name="persistent_non_wildfire_heat_source",
        label=0,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("low", "nominal"),
        news_count_range=(0, 0),
        max_radius_km=2.2,
        max_time_spread_minutes=50.0,
    ),
)

# --- V2 additions only (never used for dataset_version="v1") ---
#
# V1 finding: no positive family's satellite_confidence_pool ever included
# "low", so satellite_low_count > 0 was a perfect (though unintended) negative
# indicator - a generator artifact, not a realistic wildfire property. A real
# active-fire candidate can include a weak/low-confidence satellite pass
# alongside stronger evidence (e.g. HIGH + LOW, or NOMINAL + LOW + news). The
# families below add that realistic mixed-confidence evidence to the positive
# class. The original 13 V1 families (above) are kept unchanged and are still
# part of the V2 family pool (see _family_pools) - V2 adds to V1, it does not
# replace it.
_POSITIVE_FAMILIES_V2: tuple[_ScenarioFamily, ...] = (
    _ScenarioFamily(
        name="mixed_confidence_satellite_real_fire",
        label=1,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("low", "nominal", "high"),
        news_count_range=(0, 2),
        max_radius_km=2.2,
        max_time_spread_minutes=45.0,
    ),
    _ScenarioFamily(
        name="high_low_satellite_with_news_real_fire",
        label=1,
        satellite_count_range=(2, 3),
        satellite_confidence_pool=("high", "low"),
        news_count_range=(1, 2),
        max_radius_km=2.0,
        max_time_spread_minutes=35.0,
    ),
    _ScenarioFamily(
        name="nominal_low_satellite_with_news_real_fire",
        label=1,
        satellite_count_range=(2, 3),
        satellite_confidence_pool=("nominal", "low"),
        news_count_range=(1, 2),
        max_radius_km=2.0,
        max_time_spread_minutes=35.0,
    ),
    _ScenarioFamily(
        name="multi_confidence_persistent_real_fire",
        label=1,
        satellite_count_range=(3, 5),
        satellite_confidence_pool=("low", "nominal", "high"),
        news_count_range=(0, 2),
        max_radius_km=2.2,
        max_time_spread_minutes=50.0,
    ),
)

# Matching mixed-confidence NEGATIVE families, so "mixed confidence" or
# "has both a low and a high reading" cannot itself become a new (equally
# unintended) positive indicator - the overlap must hold in both directions.
_NEGATIVE_FAMILIES_V2: tuple[_ScenarioFamily, ...] = (
    _ScenarioFamily(
        name="mixed_confidence_non_fire_heat_source",
        label=0,
        satellite_count_range=(2, 4),
        satellite_confidence_pool=("low", "nominal", "high"),
        news_count_range=(0, 0),
        max_radius_km=1.5,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamily(
        name="high_low_satellite_false_alarm",
        label=0,
        satellite_count_range=(2, 3),
        satellite_confidence_pool=("high", "low"),
        news_count_range=(0, 1),
        max_radius_km=1.8,
        max_time_spread_minutes=30.0,
    ),
    _ScenarioFamily(
        name="mixed_satellite_with_false_news",
        label=0,
        satellite_count_range=(1, 3),
        satellite_confidence_pool=("low", "nominal", "high"),
        news_count_range=(1, 2),
        max_radius_km=2.0,
        max_time_spread_minutes=35.0,
    ),
)


class FireDetectionTrainingDataGenerator:
    """Generate a reproducible synthetic labeled Fire Detection dataset.

    Determinism: the same seed and num_samples always produce equivalent
    output. Randomness is drawn only from an explicitly seeded
    random.Random instance - no uncontrolled global randomness.
    """

    def __init__(self, seed: int = DEFAULT_TRAINING_DATA_SEED) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        self._seed = seed

    def generate(
        self,
        num_samples: int = DEFAULT_TRAINING_DATA_SAMPLE_COUNT,
        dataset_version: str = DATASET_VERSION_V1,
    ) -> tuple[FireDetectionTrainingSample, ...]:
        """Return num_samples deterministic labeled samples, ~50/50 balanced by label.

        dataset_version="v1" (default) draws only from the original 13
        families and is byte-for-byte identical to the generator's behavior
        before V2 was added (verified by
        test_v1_generation_is_unchanged_by_the_v2_addition). dataset_version="v2"
        draws from the V1 families plus the additional mixed-confidence V2
        families, fixing the V1 generator artifact where satellite_low_count
        was 0 for every positive row.
        """
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer, got {num_samples!r}")
        if dataset_version not in SUPPORTED_DATASET_VERSIONS:
            raise ValueError(f"dataset_version must be one of {SUPPORTED_DATASET_VERSIONS}, got {dataset_version!r}")

        positive_families, negative_families = self._family_pools(dataset_version)

        rng = random.Random(self._seed)

        labels = self._balanced_labels(num_samples, rng)

        satellite_id_counter = 1
        news_id_counter = 1
        samples: list[FireDetectionTrainingSample] = []
        for sample_index, label in enumerate(labels):
            family = rng.choice(positive_families if label == 1 else negative_families)
            evidence, satellite_id_counter, news_id_counter = self._generate_evidence(
                rng=rng,
                family=family,
                next_satellite_id=satellite_id_counter,
                next_news_id=news_id_counter,
            )
            samples.append(
                FireDetectionTrainingSample(
                    sample_id=sample_index + 1,
                    scenario_family=family.name,
                    evidence=evidence,
                    label=label,
                )
            )

        return tuple(samples)

    @staticmethod
    def _family_pools(dataset_version: str) -> tuple[tuple[_ScenarioFamily, ...], tuple[_ScenarioFamily, ...]]:
        if dataset_version == DATASET_VERSION_V2:
            return _POSITIVE_FAMILIES + _POSITIVE_FAMILIES_V2, _NEGATIVE_FAMILIES + _NEGATIVE_FAMILIES_V2
        return _POSITIVE_FAMILIES, _NEGATIVE_FAMILIES

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
        family: _ScenarioFamily,
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
                )
            )
            next_news_id += 1

        return tuple(evidence_items), next_satellite_id, next_news_id

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
