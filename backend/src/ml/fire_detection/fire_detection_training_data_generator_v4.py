"""Deterministic, offline synthetic labeled dataset generator for Fire Detection ML V4.

WHAT A ROW IS
    One Fire Detection candidate at one assessment time, i.e. what the ML
    classifier receives AFTER evidence validation/correlation:
        raw evidence -> correlation -> FireDetectionCandidate -> features (+ context)
    Every generated candidate is checked against FireDetectionCandidate, the
    same connectivity definition the runtime uses (max 5 km / 60 min).

WHAT THE LABEL IS
    `label = 1` iff a wildfire ACTUALLY EXISTS in the scenario, taken from the
    scenario family's ground truth (`fire_exists`) via the simulation's own
    ScenarioType (ACTIVE_FIRE vs *_NO_FIRE). It is decided BEFORE any evidence
    is generated and is never computed from the evidence. Nothing here imports
    or calls FireDetectionCalculator, FireDetectionHybridPolicy, a rule
    confidence, SUSPECTED/CONFIRMED, FireEvent status, or any ML model, so the
    label cannot inherit the hand-written scoring rules it is meant to replace.

WHY IT IS HARDER THAN V3 (see backend/docs/fire_detection_dataset_v4.md)
    * Evidence quality is generated from overlapping distributions in both
      labels: real fires with a single weak hotspot, false alarms with strong
      hotspots and confident-sounding news.
    * Candidate geometry (distance/time spread) is drawn from the same ranges
      for both labels wherever the evidence shape allows it.
    * Fire Danger (FFWI) comes from the real weather profiles + the real
      FFWICalculator, but the weather regime is chosen per family
      INDEPENDENTLY of the ground truth, so HIGH/EXTREME FFWI without a fire
      and LOW/MODERATE FFWI with a fire are both common. Fire Danger is
      context, not evidence of a fire.
    * Fire Danger is sometimes missing for both labels; missingness is kept
      explicit rather than replaced with a fake FFWI.

DETERMINISM AND INDEPENDENCE
    Each row draws from its own random.Random seeded from (seed, row index), so
    rows are independent and the output is identical for the same seed. No
    network, database, LLM, or wall-clock dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
import random

from src.calculators.fire_danger.ffwi_calculator import FFWICalculator
from src.calculators.fire_danger.ffwi_config import EXTREME_THRESHOLD
from src.calculators.fire_detection.fire_detection_config import (  # correlation limits ONLY
    MAX_EVIDENCE_DISTANCE_KM,
    MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES,
)
from src.models.fire_danger_input import FireDangerInput
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_context import FireDetectionContext
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.services.fire_detection.fire_detection_context_config import MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES
from src.simulation.generators.weather_data_generator import WEATHER_SCENARIO_PROFILES
from src.simulation.scenario_type import ScenarioType

DEFAULT_TRAINING_DATA_SEED_V4 = 42
DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4 = 9000

_ACTIVE_ISRAEL_LATITUDE_RANGE = (29.6, 33.2)
_ACTIVE_ISRAEL_LONGITUDE_RANGE = (34.3, 35.7)
_BASE_TIME_START = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
_BASE_TIME_RANGE_MINUTES = 365 * 24 * 60
_AS_OF_DELAY_MAX_MINUTES = 10.0

# Safety margins so every pair of evidence items in a row correlates directly
# under the runtime's own limits (any two points in a disk of radius r are
# <= 2r apart).
MAX_CANDIDATE_RADIUS_KM = 2.4
MAX_CANDIDATE_TIME_SPREAD_MINUTES = 58.0
assert 2 * MAX_CANDIDATE_RADIUS_KM < MAX_EVIDENCE_DISTANCE_KM
assert MAX_CANDIDATE_TIME_SPREAD_MINUTES < MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES

_MAX_EXTREME_REJECTION_ATTEMPTS = 500


class ScenarioArchetypeV4(Enum):
    """Dominant evidence SHAPE of a family (never a label).

    Every archetype contains both fire and no-fire families, so holding one out
    is a meaningful generalization test (leave-one-archetype-out).
    """

    SATELLITE_ONLY = "satellite_only"
    NEWS_LED = "news_led"
    SATELLITE_NEWS = "satellite_news"
    MULTI_SATELLITE = "multi_satellite"


class WeatherRegime(Enum):
    """Weather regime a scenario's Fire Danger is drawn from.

    LOW / MODERATE / HIGH reuse the simulation's own weather profile boxes
    (WEATHER_SCENARIO_PROFILES); EXTREME is the HIGH box restricted (by
    rejection against the real FFWICalculator) to EXTREME-level scores. The
    regime describes the WEATHER only - it is chosen independently of whether a
    fire exists.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


_REGIME_PROFILE_KEY: dict[WeatherRegime, ScenarioType] = {
    WeatherRegime.LOW: ScenarioType.LOW_RISK_NO_FIRE,
    WeatherRegime.MODERATE: ScenarioType.MODERATE_RISK_NO_FIRE,
    WeatherRegime.HIGH: ScenarioType.HIGH_RISK_NO_FIRE,
    WeatherRegime.EXTREME: ScenarioType.HIGH_RISK_NO_FIRE,
}
_REGIME_ORDER: tuple[WeatherRegime, ...] = (
    WeatherRegime.LOW,
    WeatherRegime.MODERATE,
    WeatherRegime.HIGH,
    WeatherRegime.EXTREME,
)

# NewsWildfireSignalStrength order for `news_signal_weights`: NONE, WEAK,
# MODERATE, STRONG, UNKNOWN (no reliable LLM analysis -> evidence value None).
_NEWS_SIGNAL_CHOICES: tuple[NewsWildfireSignalStrength | None, ...] = (
    NewsWildfireSignalStrength.NONE,
    NewsWildfireSignalStrength.WEAK,
    NewsWildfireSignalStrength.MODERATE,
    NewsWildfireSignalStrength.STRONG,
    None,
)
_SATELLITE_CONFIDENCE_CHOICES: tuple[str, ...] = ("low", "nominal", "high")

# Baseline weather regime mixes (LOW, MODERATE, HIGH, EXTREME) for families whose
# story does not concern Fire Danger. Fires are somewhat more common in hot, dry
# weather, so the fire baseline leans hotter than the no-fire baseline - a WEAK
# prior that makes Fire Danger useful context without letting it decide the
# label (every band keeps both labels; see the validation report's FFWI band x
# label cross-tab and its univariate AUC).
_FIRE_BASELINE_REGIME_WEIGHTS = (0.14, 0.20, 0.42, 0.24)
_NO_FIRE_BASELINE_REGIME_WEIGHTS = (0.32, 0.30, 0.28, 0.10)
_BASELINE_FIRE_DANGER_MISSING_PROBABILITY = 0.12
_MOSTLY_MISSING_FIRE_DANGER_PROBABILITY = 0.85


@dataclass(frozen=True)
class _FamilyV4:
    """One V4 scenario family: ground truth + how (noisy) evidence is generated."""

    name: str
    fire_exists: bool  # GROUND TRUTH - independent of everything generated below
    archetype: ScenarioArchetypeV4
    description: str
    satellite_count_range: tuple[int, int]
    satellite_confidence_weights: tuple[float, float, float]  # low, nominal, high
    news_count_range: tuple[int, int]
    news_signal_weights: tuple[float, float, float, float, float]  # NONE, WEAK, MODERATE, STRONG, UNKNOWN
    radius_range_km: tuple[float, float]
    time_spread_range_minutes: tuple[float, float]
    frp_range: tuple[float, float] | None = None
    brightness_range: tuple[float, float] | None = None
    frp_missing_probability: float = 0.0
    brightness_missing_probability: float = 0.0
    satellite_offset_fraction_range: tuple[float, float] = (0.0, 1.0)
    news_offset_fraction_range: tuple[float, float] = (0.0, 1.0)
    weather_regime_weights: tuple[float, float, float, float] | None = None  # None -> baseline for fire_exists
    fire_danger_missing_probability: float = _BASELINE_FIRE_DANGER_MISSING_PROBABILITY

    def __post_init__(self) -> None:
        if self.satellite_count_range[0] + self.news_count_range[0] < 1:
            raise ValueError(f"{self.name}: every candidate needs at least one evidence item.")
        if self.satellite_count_range[1] > 0 and self.frp_range is None and self.brightness_range is None:
            raise ValueError(f"{self.name}: satellite families need FRP/brightness ranges.")
        if self.radius_range_km[1] > MAX_CANDIDATE_RADIUS_KM:
            raise ValueError(f"{self.name}: radius exceeds the correlation-safe maximum.")
        if self.time_spread_range_minutes[1] > MAX_CANDIDATE_TIME_SPREAD_MINUTES:
            raise ValueError(f"{self.name}: time spread exceeds the correlation-safe maximum.")
        if len(self.regime_weights) != len(_REGIME_ORDER) or sum(self.regime_weights) <= 0:
            raise ValueError(f"{self.name}: invalid weather regime weights.")

    @property
    def regime_weights(self) -> tuple[float, float, float, float]:
        if self.weather_regime_weights is not None:
            return self.weather_regime_weights
        return _FIRE_BASELINE_REGIME_WEIGHTS if self.fire_exists else _NO_FIRE_BASELINE_REGIME_WEIGHTS


def ground_truth_label(scenario_type: ScenarioType) -> int:
    """The V4 label: 1 iff the scenario's ground truth is an ACTIVE_FIRE. Nothing else feeds it."""
    if not isinstance(scenario_type, ScenarioType):
        raise ValueError(f"scenario_type must be a ScenarioType, got {scenario_type!r}")
    return 1 if scenario_type is ScenarioType.ACTIVE_FIRE else 0


_SA = ScenarioArchetypeV4.SATELLITE_ONLY
_NL = ScenarioArchetypeV4.NEWS_LED
_SN = ScenarioArchetypeV4.SATELLITE_NEWS
_MS = ScenarioArchetypeV4.MULTI_SATELLITE

# --- POSITIVE families: a wildfire really exists ---
_POSITIVE_FAMILIES_V4: tuple[_FamilyV4, ...] = (
    _FamilyV4(
        name="P1_strong_satellite_strong_news_high_danger",
        fire_exists=True,
        archetype=_SN,
        description="Fire with strong satellite + strong news under mostly HIGH/EXTREME danger (the easy case).",
        satellite_count_range=(1, 3),
        satellite_confidence_weights=(0.05, 0.35, 0.60),
        news_count_range=(1, 2),
        news_signal_weights=(0.0, 0.10, 0.35, 0.50, 0.05),
        radius_range_km=(0.2, 2.0),
        time_spread_range_minutes=(5.0, 35.0),
        frp_range=(35.0, 150.0),
        brightness_range=(320.0, 400.0),
        frp_missing_probability=0.10,
        brightness_missing_probability=0.10,
        weather_regime_weights=(0.05, 0.15, 0.50, 0.30),
    ),
    _FamilyV4(
        name="P2_satellite_only_fire",
        fire_exists=True,
        archetype=_SA,
        description="Fire seen only by satellite (no news yet).",
        satellite_count_range=(1, 2),
        satellite_confidence_weights=(0.05, 0.45, 0.50),
        news_count_range=(0, 0),
        news_signal_weights=(0.0, 0.0, 0.0, 0.0, 1.0),
        radius_range_km=(0.1, 1.5),
        time_spread_range_minutes=(0.0, 25.0),
        frp_range=(15.0, 110.0),
        brightness_range=(300.0, 380.0),
        frp_missing_probability=0.25,
        brightness_missing_probability=0.25,
    ),
    _FamilyV4(
        name="P3_news_first_satellite_later",
        fire_exists=True,
        archetype=_NL,
        description="Fire reported by news first; the satellite hotspot (if any yet) arrives later.",
        satellite_count_range=(0, 2),
        satellite_confidence_weights=(0.25, 0.50, 0.25),
        news_count_range=(1, 3),
        news_signal_weights=(0.02, 0.30, 0.35, 0.28, 0.05),
        radius_range_km=(0.2, 2.0),
        time_spread_range_minutes=(15.0, 55.0),
        frp_range=(10.0, 80.0),
        brightness_range=(295.0, 360.0),
        frp_missing_probability=0.30,
        brightness_missing_probability=0.30,
        satellite_offset_fraction_range=(0.55, 1.0),
        news_offset_fraction_range=(0.0, 0.45),
    ),
    _FamilyV4(
        name="P4_fire_moderate_danger",
        fire_exists=True,
        archetype=_SN,
        description="Real fire while FFWI is only MODERATE.",
        satellite_count_range=(1, 3),
        satellite_confidence_weights=(0.15, 0.50, 0.35),
        news_count_range=(0, 2),
        news_signal_weights=(0.05, 0.30, 0.35, 0.25, 0.05),
        radius_range_km=(0.2, 2.0),
        time_spread_range_minutes=(0.0, 45.0),
        frp_range=(15.0, 100.0),
        brightness_range=(300.0, 380.0),
        frp_missing_probability=0.20,
        brightness_missing_probability=0.20,
        weather_regime_weights=(0.10, 0.70, 0.20, 0.0),
    ),
    _FamilyV4(
        name="P5_fire_low_ffwi_despite_ignition",
        fire_exists=True,
        archetype=_SN,
        description="Real ignition under LOW/MODERATE FFWI (same evidence as P4).",
        satellite_count_range=(1, 3),
        satellite_confidence_weights=(0.15, 0.50, 0.35),
        news_count_range=(0, 2),
        news_signal_weights=(0.05, 0.30, 0.35, 0.25, 0.05),
        radius_range_km=(0.2, 2.0),
        time_spread_range_minutes=(0.0, 45.0),
        frp_range=(15.0, 100.0),
        brightness_range=(300.0, 380.0),
        frp_missing_probability=0.20,
        brightness_missing_probability=0.20,
        weather_regime_weights=(0.75, 0.25, 0.0, 0.0),
    ),
    _FamilyV4(
        name="P6_fire_missing_fire_danger",
        fire_exists=True,
        archetype=_SN,
        description="Real fire where no usable Fire Danger assessment exists.",
        satellite_count_range=(1, 3),
        satellite_confidence_weights=(0.10, 0.45, 0.45),
        news_count_range=(0, 2),
        news_signal_weights=(0.05, 0.25, 0.35, 0.30, 0.05),
        radius_range_km=(0.2, 2.0),
        time_spread_range_minutes=(0.0, 40.0),
        frp_range=(20.0, 130.0),
        brightness_range=(300.0, 390.0),
        frp_missing_probability=0.20,
        brightness_missing_probability=0.20,
        fire_danger_missing_probability=_MOSTLY_MISSING_FIRE_DANGER_PROBABILITY,
    ),
    _FamilyV4(
        name="P7_fire_weak_satellite_measurements",
        fire_exists=True,
        archetype=_SA,
        description="Real (e.g. early/small) fire with only weak, often unmeasured hotspots.",
        satellite_count_range=(1, 2),
        satellite_confidence_weights=(0.40, 0.50, 0.10),
        news_count_range=(0, 0),
        news_signal_weights=(0.0, 0.0, 0.0, 0.0, 1.0),
        radius_range_km=(0.1, 1.2),
        time_spread_range_minutes=(0.0, 20.0),
        frp_range=(5.0, 40.0),
        brightness_range=(290.0, 335.0),
        frp_missing_probability=0.40,
        brightness_missing_probability=0.40,
    ),
    _FamilyV4(
        name="P8_fire_evidence_spread_over_time",
        fire_exists=True,
        archetype=_MS,
        description="Real fire whose hotspots/reports accumulate over most of the correlation window.",
        satellite_count_range=(2, 4),
        satellite_confidence_weights=(0.20, 0.45, 0.35),
        news_count_range=(0, 1),
        news_signal_weights=(0.10, 0.30, 0.30, 0.25, 0.05),
        radius_range_km=(0.5, 2.4),
        time_spread_range_minutes=(30.0, 58.0),
        frp_range=(10.0, 120.0),
        brightness_range=(295.0, 385.0),
        frp_missing_probability=0.20,
        brightness_missing_probability=0.20,
    ),
    _FamilyV4(
        name="P9_fire_persistent_multi_satellite",
        fire_exists=True,
        archetype=_MS,
        description="Real, persistent fire seen by several hotspots.",
        satellite_count_range=(3, 5),
        satellite_confidence_weights=(0.20, 0.40, 0.40),
        news_count_range=(0, 2),
        news_signal_weights=(0.05, 0.25, 0.35, 0.30, 0.05),
        radius_range_km=(0.3, 2.2),
        time_spread_range_minutes=(20.0, 55.0),
        frp_range=(20.0, 140.0),
        brightness_range=(300.0, 390.0),
        frp_missing_probability=0.20,
        brightness_missing_probability=0.20,
    ),
    _FamilyV4(
        name="P10_fire_news_led_multiple_reports",
        fire_exists=True,
        archetype=_NL,
        description="Real fire reported by several news items; at most one late, weak hotspot.",
        satellite_count_range=(0, 1),
        satellite_confidence_weights=(0.30, 0.50, 0.20),
        news_count_range=(2, 4),
        news_signal_weights=(0.02, 0.22, 0.38, 0.33, 0.05),
        radius_range_km=(0.2, 2.4),
        time_spread_range_minutes=(5.0, 50.0),
        frp_range=(10.0, 80.0),
        brightness_range=(295.0, 360.0),
        frp_missing_probability=0.30,
        brightness_missing_probability=0.30,
        satellite_offset_fraction_range=(0.5, 1.0),
    ),
)

# --- NEGATIVE families: NO wildfire exists (many are deliberate hard negatives) ---
_NEGATIVE_FAMILIES_V4: tuple[_FamilyV4, ...] = (
    _FamilyV4(
        name="N1_extreme_danger_no_fire",
        fire_exists=False,
        archetype=_SA,
        description="EXTREME fire weather, no fire; weak ambiguous hotspots and a stray report.",
        satellite_count_range=(1, 2),
        satellite_confidence_weights=(0.35, 0.50, 0.15),
        news_count_range=(0, 1),
        news_signal_weights=(0.30, 0.40, 0.20, 0.05, 0.05),
        radius_range_km=(0.1, 2.0),
        time_spread_range_minutes=(0.0, 35.0),
        frp_range=(5.0, 60.0),
        brightness_range=(290.0, 350.0),
        frp_missing_probability=0.35,
        brightness_missing_probability=0.35,
        weather_regime_weights=(0.0, 0.0, 0.35, 0.65),
    ),
    _FamilyV4(
        name="N2_high_danger_no_fire_rumor",
        fire_exists=False,
        archetype=_NL,
        description="HIGH danger, no fire; smoke rumors, at most a faint hotspot.",
        satellite_count_range=(0, 1),
        satellite_confidence_weights=(0.50, 0.40, 0.10),
        news_count_range=(1, 2),
        news_signal_weights=(0.25, 0.40, 0.25, 0.05, 0.05),
        radius_range_km=(0.2, 2.0),
        time_spread_range_minutes=(0.0, 45.0),
        frp_range=(5.0, 50.0),
        brightness_range=(290.0, 340.0),
        frp_missing_probability=0.35,
        brightness_missing_probability=0.35,
        weather_regime_weights=(0.0, 0.05, 0.75, 0.20),
    ),
    _FamilyV4(
        name="N3_isolated_low_confidence_hotspot_noise",
        fire_exists=False,
        archetype=_SA,
        description="Satellite noise: isolated low-confidence hotspots, no news.",
        satellite_count_range=(1, 3),
        satellite_confidence_weights=(0.80, 0.20, 0.0),
        news_count_range=(0, 0),
        news_signal_weights=(0.0, 0.0, 0.0, 0.0, 1.0),
        radius_range_km=(0.1, 2.0),
        time_spread_range_minutes=(0.0, 40.0),
        frp_range=(5.0, 35.0),
        brightness_range=(290.0, 330.0),
        frp_missing_probability=0.40,
        brightness_missing_probability=0.40,
    ),
    _FamilyV4(
        name="N4_false_or_weak_news_report",
        fire_exists=False,
        archetype=_NL,
        description="News-only false alarm; wording is usually weak but sometimes convincing.",
        satellite_count_range=(0, 0),
        satellite_confidence_weights=(1.0, 0.0, 0.0),
        news_count_range=(1, 2),
        news_signal_weights=(0.35, 0.35, 0.15, 0.10, 0.05),
        radius_range_km=(0.1, 1.5),
        time_spread_range_minutes=(0.0, 40.0),
    ),
    _FamilyV4(
        name="N5_satellite_plus_unrelated_news",
        fire_exists=False,
        archetype=_SN,
        description="A hotspot plus news that is not about a wildfire at that place.",
        satellite_count_range=(1, 2),
        satellite_confidence_weights=(0.15, 0.50, 0.35),
        news_count_range=(1, 2),
        news_signal_weights=(0.45, 0.30, 0.15, 0.02, 0.08),
        radius_range_km=(0.2, 2.0),
        time_spread_range_minutes=(0.0, 40.0),
        frp_range=(15.0, 90.0),
        brightness_range=(300.0, 370.0),
        frp_missing_probability=0.25,
        brightness_missing_probability=0.25,
    ),
    _FamilyV4(
        name="N6_multiple_weak_pieces_no_incident",
        fire_exists=False,
        archetype=_MS,
        description="Several weak hotspots and stray reports that never amount to a fire.",
        satellite_count_range=(2, 4),
        satellite_confidence_weights=(0.70, 0.25, 0.05),
        news_count_range=(1, 2),
        news_signal_weights=(0.30, 0.50, 0.15, 0.0, 0.05),
        radius_range_km=(0.5, 2.4),
        time_spread_range_minutes=(25.0, 58.0),
        frp_range=(5.0, 40.0),
        brightness_range=(290.0, 335.0),
        frp_missing_probability=0.35,
        brightness_missing_probability=0.35,
    ),
    _FamilyV4(
        name="N7_no_fire_missing_fire_danger",
        fire_exists=False,
        archetype=_SN,
        description="No fire, ambiguous hotspot+news, and no usable Fire Danger assessment.",
        satellite_count_range=(1, 2),
        satellite_confidence_weights=(0.15, 0.55, 0.30),
        news_count_range=(1, 2),
        news_signal_weights=(0.30, 0.35, 0.25, 0.05, 0.05),
        radius_range_km=(0.2, 2.0),
        time_spread_range_minutes=(0.0, 40.0),
        frp_range=(15.0, 100.0),
        brightness_range=(300.0, 375.0),
        frp_missing_probability=0.25,
        brightness_missing_probability=0.25,
        fire_danger_missing_probability=_MOSTLY_MISSING_FIRE_DANGER_PROBABILITY,
    ),
    _FamilyV4(
        name="N8_realistic_strong_looking_false_positive",
        fire_exists=False,
        archetype=_SN,
        description="Nominal/high hotspots with sizeable FRP and convincing-sounding news, yet no wildfire (e.g. industrial/agricultural burning).",
        satellite_count_range=(1, 3),
        satellite_confidence_weights=(0.10, 0.50, 0.40),
        news_count_range=(1, 2),
        news_signal_weights=(0.10, 0.30, 0.30, 0.20, 0.10),
        radius_range_km=(0.2, 1.5),
        time_spread_range_minutes=(0.0, 40.0),
        frp_range=(25.0, 110.0),
        brightness_range=(305.0, 385.0),
        frp_missing_probability=0.25,
        brightness_missing_probability=0.25,
    ),
    _FamilyV4(
        name="N9_persistent_non_wildfire_heat_source",
        fire_exists=False,
        archetype=_MS,
        description="A stationary industrial-type heat source seen repeatedly over the window.",
        satellite_count_range=(3, 5),
        satellite_confidence_weights=(0.20, 0.55, 0.25),
        news_count_range=(0, 1),
        news_signal_weights=(0.60, 0.30, 0.05, 0.0, 0.05),
        radius_range_km=(0.05, 1.5),
        time_spread_range_minutes=(15.0, 55.0),
        frp_range=(15.0, 80.0),
        brightness_range=(300.0, 360.0),
        frp_missing_probability=0.25,
        brightness_missing_probability=0.25,
    ),
    _FamilyV4(
        name="N10_repeated_reports_of_same_false_alarm",
        fire_exists=False,
        archetype=_NL,
        description="Several news items repeating the same unfounded alarm; no hotspot.",
        satellite_count_range=(0, 0),
        satellite_confidence_weights=(1.0, 0.0, 0.0),
        news_count_range=(2, 4),
        news_signal_weights=(0.15, 0.35, 0.25, 0.20, 0.05),
        radius_range_km=(0.2, 2.4),
        time_spread_range_minutes=(5.0, 50.0),
    ),
)

FAMILIES_V4: tuple[_FamilyV4, ...] = _POSITIVE_FAMILIES_V4 + _NEGATIVE_FAMILIES_V4


@dataclass(frozen=True)
class FireDetectionTrainingSampleV4:
    """One synthetic labeled Fire Detection candidate + its Fire Danger context.

    `label` is ground truth (see `ground_truth_label`). sample_id, seed,
    scenario_*, ground_truth_scenario_type and as_of are metadata for
    splitting/analysis only and must never be used as ML features.
    """

    sample_id: int
    seed: int
    scenario_family: str
    scenario_archetype: ScenarioArchetypeV4
    ground_truth_scenario_type: ScenarioType
    label: int
    evidence: tuple[FireDetectionEvidence, ...]
    context: FireDetectionContext
    as_of: datetime

    def __post_init__(self) -> None:
        if self.label != ground_truth_label(self.ground_truth_scenario_type):
            raise ValueError("label must equal the ground truth of ground_truth_scenario_type.")
        if not isinstance(self.context, FireDetectionContext):
            raise ValueError(f"context must be a FireDetectionContext, got {self.context!r}")
        if not isinstance(self.as_of, datetime) or self.as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {self.as_of!r}")


class FireDetectionTrainingDataGeneratorV4:
    """Generate a reproducible synthetic labeled Fire Detection V4 dataset."""

    def __init__(self, seed: int = DEFAULT_TRAINING_DATA_SEED_V4) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        self._seed = seed
        self._ffwi = FFWICalculator()

    def generate(
        self,
        num_samples: int = DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4,
    ) -> tuple[FireDetectionTrainingSampleV4, ...]:
        """Return num_samples samples, ~50/50 by label and spread evenly across families."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer, got {num_samples!r}")

        assignment = self._family_assignment(num_samples)
        satellite_id = 1
        news_id = 1
        samples: list[FireDetectionTrainingSampleV4] = []
        for index, family in enumerate(assignment):
            rng = random.Random(f"ecoguard-fire-detection-v4:{self._seed}:{index}")
            sample, satellite_id, news_id = self._generate_sample(
                rng=rng,
                family=family,
                sample_id=index + 1,
                next_satellite_id=satellite_id,
                next_news_id=news_id,
            )
            samples.append(sample)
        return tuple(samples)

    def _family_assignment(self, num_samples: int) -> list[_FamilyV4]:
        positives = num_samples // 2
        negatives = num_samples - positives
        assignment: list[_FamilyV4] = []
        for families, count in ((_POSITIVE_FAMILIES_V4, positives), (_NEGATIVE_FAMILIES_V4, negatives)):
            base, remainder = divmod(count, len(families))
            for family_index, family in enumerate(families):
                assignment.extend([family] * (base + (1 if family_index < remainder else 0)))
        random.Random(f"ecoguard-fire-detection-v4-assignment:{self._seed}").shuffle(assignment)
        return assignment

    def _generate_sample(
        self,
        rng: random.Random,
        family: _FamilyV4,
        sample_id: int,
        next_satellite_id: int,
        next_news_id: int,
    ) -> tuple[FireDetectionTrainingSampleV4, int, int]:
        center_latitude = rng.uniform(*_ACTIVE_ISRAEL_LATITUDE_RANGE)
        center_longitude = rng.uniform(*_ACTIVE_ISRAEL_LONGITUDE_RANGE)
        base_time = _BASE_TIME_START + timedelta(minutes=rng.uniform(0.0, _BASE_TIME_RANGE_MINUTES))
        radius_km = rng.uniform(*family.radius_range_km)
        spread_minutes = rng.uniform(*family.time_spread_range_minutes)
        intensity = rng.random()  # per-scenario latent: brighter fires tend to have higher FRP

        evidence: list[FireDetectionEvidence] = []
        for _ in range(rng.randint(*family.satellite_count_range)):
            latitude, longitude = _point_in_disk(rng, center_latitude, center_longitude, radius_km)
            evidence.append(
                FireDetectionEvidence(
                    evidence_id=next_satellite_id,
                    evidence_type=FireEvidenceType.SATELLITE,
                    latitude=latitude,
                    longitude=longitude,
                    observed_at=base_time
                    + timedelta(minutes=spread_minutes * rng.uniform(*family.satellite_offset_fraction_range)),
                    satellite_confidence=rng.choices(
                        _SATELLITE_CONFIDENCE_CHOICES, weights=family.satellite_confidence_weights
                    )[0],
                    satellite_frp=_measurement(rng, family.frp_range, family.frp_missing_probability, intensity),
                    satellite_brightness=_measurement(
                        rng, family.brightness_range, family.brightness_missing_probability, intensity
                    ),
                )
            )
            next_satellite_id += 1

        for _ in range(rng.randint(*family.news_count_range)):
            latitude, longitude = _point_in_disk(rng, center_latitude, center_longitude, radius_km)
            evidence.append(
                FireDetectionEvidence(
                    evidence_id=next_news_id,
                    evidence_type=FireEvidenceType.NEWS,
                    latitude=latitude,
                    longitude=longitude,
                    observed_at=base_time
                    + timedelta(minutes=spread_minutes * rng.uniform(*family.news_offset_fraction_range)),
                    news_wildfire_signal_strength=rng.choices(
                        _NEWS_SIGNAL_CHOICES, weights=family.news_signal_weights
                    )[0],
                )
            )
            next_news_id += 1

        evidence_tuple = tuple(evidence)
        FireDetectionCandidate(evidence_tuple)  # same connectivity rule as runtime; raises if violated

        as_of = max(item.observed_at for item in evidence_tuple) + timedelta(
            minutes=rng.uniform(0.0, _AS_OF_DELAY_MAX_MINUTES)
        )
        score, level = self._sample_weather_fire_danger(rng, family)
        context = self._fire_danger_context(rng, family, sample_id, as_of, score, level)

        scenario_type = _ground_truth_scenario_type(family, level)
        sample = FireDetectionTrainingSampleV4(
            sample_id=sample_id,
            seed=self._seed,
            scenario_family=family.name,
            scenario_archetype=family.archetype,
            ground_truth_scenario_type=scenario_type,
            label=ground_truth_label(scenario_type),
            evidence=evidence_tuple,
            context=context,
            as_of=as_of,
        )
        return sample, next_satellite_id, next_news_id

    def _sample_weather_fire_danger(self, rng: random.Random, family: _FamilyV4) -> tuple[float, FireDangerLevel]:
        """Draw weather from a regime's box and score it with the real FFWICalculator."""
        regime = rng.choices(_REGIME_ORDER, weights=family.regime_weights)[0]
        profile = WEATHER_SCENARIO_PROFILES[_REGIME_PROFILE_KEY[regime]]

        def draw() -> FireDangerInput:
            return FireDangerInput(
                temperature_c=rng.uniform(*profile.temperature_celsius),
                relative_humidity_pct=rng.uniform(*profile.relative_humidity_percent),
                wind_speed_kmh=rng.uniform(*profile.wind_speed_kmh),
            )

        calculation = self._ffwi.calculate(draw())
        if regime is WeatherRegime.EXTREME:
            attempts = 1
            while calculation.score < EXTREME_THRESHOLD and attempts < _MAX_EXTREME_REJECTION_ATTEMPTS:
                calculation = self._ffwi.calculate(draw())
                attempts += 1
            if calculation.score < EXTREME_THRESHOLD:  # pathological: fall back to the box's hottest corner
                calculation = self._ffwi.calculate(
                    FireDangerInput(
                        temperature_c=profile.temperature_celsius[1],
                        relative_humidity_pct=profile.relative_humidity_percent[0],
                        wind_speed_kmh=profile.wind_speed_kmh[1],
                    )
                )
        return round(calculation.score, 4), calculation.level

    @staticmethod
    def _fire_danger_context(
        rng: random.Random,
        family: _FamilyV4,
        sample_id: int,
        as_of: datetime,
        score: float,
        level: FireDangerLevel,
    ) -> FireDetectionContext:
        # Availability and age are drawn independently of the label, of the
        # weather, and of the evidence: the same process for every family except
        # the explicit "missing Fire Danger" families.
        if rng.random() < family.fire_danger_missing_probability:
            return FireDetectionContext.unavailable()
        age_minutes = round(rng.uniform(0.0, MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES), 2)
        return FireDetectionContext(
            fire_danger_available=True,
            fire_danger_score=score,
            fire_danger_age_minutes=age_minutes,
            fire_danger_level=level,
            fire_danger_assessment_id=sample_id,  # synthetic, only for traceability
            fire_danger_assessed_at=as_of - timedelta(minutes=age_minutes),
        )


def _ground_truth_scenario_type(family: _FamilyV4, level: FireDangerLevel) -> ScenarioType:
    """Scenario type from ground truth (fire_exists); no-fire rows are named by their true weather."""
    if family.fire_exists:
        return ScenarioType.ACTIVE_FIRE
    if level is FireDangerLevel.LOW:
        return ScenarioType.LOW_RISK_NO_FIRE
    if level is FireDangerLevel.MODERATE:
        return ScenarioType.MODERATE_RISK_NO_FIRE
    return ScenarioType.HIGH_RISK_NO_FIRE


def _measurement(
    rng: random.Random,
    value_range: tuple[float, float] | None,
    missing_probability: float,
    intensity: float,
) -> float | None:
    """A satellite FRP/brightness value, or None when unmeasured."""
    if value_range is None or rng.random() < missing_probability:
        return None
    low, high = value_range
    position = min(max(intensity + rng.gauss(0.0, 0.12), 0.0), 1.0)
    return round(low + (high - low) * position, 2)


def _point_in_disk(
    rng: random.Random,
    center_latitude: float,
    center_longitude: float,
    radius_km: float,
) -> tuple[float, float]:
    distance_km = radius_km * math.sqrt(rng.random())
    bearing = rng.uniform(0.0, 2.0 * math.pi)
    km_per_latitude_degree = 111.32
    latitude_delta = distance_km * math.cos(bearing) / km_per_latitude_degree
    longitude_delta = distance_km * math.sin(bearing) / (
        km_per_latitude_degree * math.cos(math.radians(center_latitude))
    )
    return round(center_latitude + latitude_delta, 6), round(center_longitude + longitude_delta, 6)
