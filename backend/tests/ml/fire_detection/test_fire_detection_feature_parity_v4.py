"""Train/runtime feature parity for V4: the dataset and runtime inference use the same extractor."""
from __future__ import annotations

import ast
import math
from datetime import timedelta
from pathlib import Path

import pytest

from src.ml.fire_detection.fire_detection_dataset_v4 import (
    extract_features_for_sample,
    feature_values_for_sample,
    load_training_dataset_rows_v4,
    row_from_sample,
)
from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_feature_extractor_v4 import FireDetectionFeatureExtractorV4
from src.ml.fire_detection.fire_detection_features_v4 import FIRE_DETECTION_FEATURE_NAMES_V4
from src.ml.fire_detection.fire_detection_training_data_generator_v4 import FireDetectionTrainingDataGeneratorV4
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_context import FireDetectionContext
from src.models.fire_evidence_type import FireEvidenceType

ML_ROOT = Path(__file__).resolve().parents[3] / "src" / "ml" / "fire_detection"
COMMITTED_CSV = Path(__file__).resolve().parents[3] / "data" / "fire_detection" / "training_v4.csv"


def same_values(first, second):
    """Exact element-wise equality where NaN == NaN and None == None."""
    if len(first) != len(second):
        return False
    for a, b in zip(first, second):
        a_missing = a is None or (isinstance(a, float) and math.isnan(a))
        b_missing = b is None or (isinstance(b, float) and math.isnan(b))
        if a_missing or b_missing:
            if not (a_missing and b_missing):
                return False
        elif a != b:
            return False
    return True


def legacy_task2_feature_values(sample):
    """How Task 2 computed the 19 values: V3 evidence dict plus hand-copied context fields."""
    values = dict(FireDetectionFeatureExtractorV3().extract(sample.evidence).as_dict())
    context = sample.context
    values["fire_danger_available"] = 1 if context.fire_danger_available else 0
    values["fire_danger_score"] = context.fire_danger_score
    values["fire_danger_age_minutes"] = context.fire_danger_age_minutes
    return tuple(values[name] for name in FIRE_DETECTION_FEATURE_NAMES_V4)


def independent_runtime_features(sample):
    """What a runtime caller would do: rebuild candidate + context on its own and call the extractor.

    The candidate is built from REVERSED evidence and the context is rebuilt with a wrong level,
    assessment id and timestamp on purpose: only availability, score and age may matter.
    """
    candidate = FireDetectionCandidate(tuple(reversed(sample.evidence)))
    context = sample.context
    if context.fire_danger_available:
        rebuilt = FireDetectionContext(
            fire_danger_available=True,
            fire_danger_score=context.fire_danger_score,
            fire_danger_age_minutes=context.fire_danger_age_minutes,
            fire_danger_level=FireDangerLevel.LOW,
            fire_danger_assessment_id=context.fire_danger_assessment_id + 1000,
            fire_danger_assessed_at=context.fire_danger_assessed_at - timedelta(days=1),
        )
    else:
        rebuilt = FireDetectionContext.unavailable()
    return FireDetectionFeatureExtractorV4().extract(candidate, rebuilt)


def first_sample(samples, family_prefix, predicate=lambda sample: True):
    return next(s for s in samples if s.scenario_family.startswith(family_prefix) and predicate(s))


# --- the dataset is produced by the canonical extractor ---


def test_dataset_rows_are_produced_by_the_canonical_v4_extractor(v4_samples, monkeypatch):
    def refuse(self, candidate, context):  # noqa: ANN001
        raise AssertionError("dataset generation went through the canonical extractor")

    monkeypatch.setattr(FireDetectionFeatureExtractorV4, "extract", refuse)

    with pytest.raises(AssertionError, match="canonical extractor"):
        row_from_sample(v4_samples[0])


def test_a_change_to_the_v3_evidence_extractor_reaches_the_dataset_rows(v4_samples, monkeypatch):
    """Training rows depend on the V3 evidence logic only through the shared extractor chain."""
    original = FireDetectionFeatureExtractorV3.extract

    def shifted(self, evidence):  # noqa: ANN001
        features = original(self, evidence)
        return type(features)(**{**features.as_dict(), "satellite_low_count": features.satellite_low_count + 5})

    monkeypatch.setattr(FireDetectionFeatureExtractorV3, "extract", shifted)
    sample = v4_samples[0]

    assert row_from_sample(sample).feature("satellite_low_count") == 5 + sum(
        1 for item in sample.evidence if item.satellite_confidence == "low"
    )
    assert independent_runtime_features(sample)["satellite_low_count"] == row_from_sample(sample).feature("satellite_low_count")


def test_dataset_and_generator_modules_hold_no_feature_calculation_of_their_own():
    dataset_tree = ast.parse((ML_ROOT / "fire_detection_dataset_v4.py").read_text(encoding="utf-8"))
    generator_source = (ML_ROOT / "fire_detection_training_data_generator_v4.py").read_text(encoding="utf-8")

    # Code only (docstrings may mention the names): no V3 extractor use, no hand-built context
    # feature names or distance maths in the dataset module.
    code_names = (
        {node.id for node in ast.walk(dataset_tree) if isinstance(node, ast.Name)}
        | {node.attr for node in ast.walk(dataset_tree) if isinstance(node, ast.Attribute)}
        | {alias.name for node in ast.walk(dataset_tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
        | {node.value for node in ast.walk(dataset_tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    )
    assert "FireDetectionFeatureExtractorV3" not in code_names
    assert "FireDetectionFeatureExtractorV4" in code_names
    for context_feature in ("fire_danger_available", "fire_danger_score", "fire_danger_age_minutes"):
        assert context_feature not in code_names, context_feature
    assert not any("haversine" in name.lower() for name in code_names)
    tree = ast.parse(generator_source)
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not {module for module in imported if "feature_extractor" in module or module.endswith("features_v4")}


# --- offline == runtime, for every row ---


def test_refactored_dataset_features_equal_the_task2_computation_for_every_sample(v4_samples):
    for sample in v4_samples:
        assert same_values(extract_features_for_sample(sample).to_nullable_tuple(), legacy_task2_feature_values(sample)), sample.sample_id


def test_runtime_extraction_equals_dataset_features_for_every_sample(v4_samples):
    for sample in v4_samples:
        offline = row_from_sample(sample).features
        runtime = independent_runtime_features(sample).to_nullable_tuple()
        assert same_values(offline, runtime), sample.sample_id


def test_runtime_extraction_equals_the_committed_csv_for_every_row(v4_samples):
    rows = load_training_dataset_rows_v4(COMMITTED_CSV)

    assert len(rows) == len(v4_samples)
    for sample, row in zip(v4_samples, rows):
        assert row.sample_id == sample.sample_id
        assert same_values(independent_runtime_features(sample).as_tuple(), row.to_features_v4().as_tuple()), sample.sample_id


def test_every_stored_row_satisfies_the_feature_contract():
    rows = load_training_dataset_rows_v4(COMMITTED_CSV)

    for row in rows:
        features = row.to_features_v4()
        assert features.names == FIRE_DETECTION_FEATURE_NAMES_V4
        assert features.fire_danger_available == (row.feature("fire_danger_available") == 1)


# --- representative scenario families ---

SCENARIOS = {
    "strong_real_fire": lambda samples: first_sample(
        samples,
        "P1",
        lambda s: any(i.satellite_confidence == "high" for i in s.evidence)
        and any(i.evidence_type is FireEvidenceType.NEWS for i in s.evidence)
        and s.context.fire_danger_available,
    ),
    "weak_real_fire": lambda samples: first_sample(
        samples, "P7", lambda s: len(s.evidence) == 1 and s.evidence[0].satellite_confidence == "low"
    ),
    "hard_negative": lambda samples: first_sample(
        samples,
        "N8",
        lambda s: any(i.satellite_confidence == "high" for i in s.evidence)
        and any(i.evidence_type is FireEvidenceType.NEWS for i in s.evidence),
    ),
    "extreme_danger_no_fire": lambda samples: first_sample(
        samples, "N1", lambda s: s.context.fire_danger_available and s.context.fire_danger_level is FireDangerLevel.EXTREME
    ),
    "missing_danger_fire": lambda samples: first_sample(samples, "P6", lambda s: not s.context.fire_danger_available),
    "missing_danger_no_fire": lambda samples: first_sample(samples, "N7", lambda s: not s.context.fire_danger_available),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenario_family_parity_between_generator_and_runtime_extractor(v4_samples, scenario):
    sample = SCENARIOS[scenario](v4_samples)

    offline = row_from_sample(sample).features
    runtime = independent_runtime_features(sample)

    assert same_values(offline, runtime.to_nullable_tuple())
    assert len(runtime.as_tuple()) == 19
    # the 16 evidence features are exactly V3's
    assert runtime.as_tuple()[:16] == FireDetectionFeatureExtractorV3().extract(sample.evidence).as_tuple()
    # the 3 context features are exactly the context's availability / score / age
    context = sample.context
    if context.fire_danger_available:
        assert runtime.as_tuple()[16:] == (1, context.fire_danger_score, context.fire_danger_age_minutes)
    else:
        assert runtime["fire_danger_available"] == 0
        assert math.isnan(runtime["fire_danger_score"]) and math.isnan(runtime["fire_danger_age_minutes"])


def test_scenario_labels_are_what_the_families_promise(v4_samples):
    assert SCENARIOS["strong_real_fire"](v4_samples).label == 1
    assert SCENARIOS["weak_real_fire"](v4_samples).label == 1
    assert SCENARIOS["missing_danger_fire"](v4_samples).label == 1
    assert SCENARIOS["hard_negative"](v4_samples).label == 0
    assert SCENARIOS["extreme_danger_no_fire"](v4_samples).label == 0
    assert SCENARIOS["missing_danger_no_fire"](v4_samples).label == 0


# --- determinism and stored-form parity ---


@pytest.mark.parametrize("seed", [42, 7, 2026])
def test_feature_extraction_is_deterministic_for_a_seed(seed):
    first = [feature_values_for_sample(s) for s in FireDetectionTrainingDataGeneratorV4(seed=seed).generate(300)]
    second = [feature_values_for_sample(s) for s in FireDetectionTrainingDataGeneratorV4(seed=seed).generate(300)]

    assert first == second


def test_stored_none_and_runtime_nan_describe_the_same_missing_danger(v4_samples):
    sample = SCENARIOS["missing_danger_fire"](v4_samples)
    stored = row_from_sample(sample).features
    runtime = independent_runtime_features(sample)

    assert stored[16] == 0 and stored[17] is None and stored[18] is None
    assert runtime["fire_danger_available"] == 0
    assert math.isnan(runtime["fire_danger_score"]) and math.isnan(runtime["fire_danger_age_minutes"])
    assert runtime.to_nullable_tuple() == stored


def test_a_real_zero_ffwi_stays_distinct_from_missing_across_the_storage_round_trip(v4_samples, tmp_path):
    from dataclasses import replace

    from src.ml.fire_detection.fire_detection_dataset_v4 import write_training_dataset_v4

    base = v4_samples[0]
    zero = replace(
        base,
        context=FireDetectionContext(
            fire_danger_available=True,
            fire_danger_score=0.0,
            fire_danger_age_minutes=0.0,
            fire_danger_level=FireDangerLevel.LOW,
            fire_danger_assessment_id=1,
            fire_danger_assessed_at=base.as_of,
        ),
    )
    missing = replace(base, context=FireDetectionContext.unavailable())
    path = write_training_dataset_v4((zero, missing), tmp_path / "v4.csv")

    zero_row, missing_row = load_training_dataset_rows_v4(path)
    zero_features, missing_features = zero_row.to_features_v4(), missing_row.to_features_v4()

    assert zero_features["fire_danger_score"] == 0.0 and zero_features["fire_danger_available"] == 1
    assert math.isnan(missing_features["fire_danger_score"]) and missing_features["fire_danger_available"] == 0
    assert zero_features.as_tuple()[:16] == missing_features.as_tuple()[:16]
