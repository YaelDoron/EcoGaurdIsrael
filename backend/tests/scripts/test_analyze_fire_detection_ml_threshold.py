"""Tests for the pure evaluation logic in scripts/analyze_fire_detection_ml_threshold.py.

Does not run the full cross_val_predict pipeline (that's exercised by
actually running the script) - just the threshold metric/selection math.
"""
from __future__ import annotations

import scripts.analyze_fire_detection_ml_threshold as threshold_script


def test_evaluate_threshold_computes_precision_recall_fpr():
    probabilities = [0.9, 0.8, 0.3, 0.2, 0.95]
    labels = [1, 0, 1, 0, 1]

    result = threshold_script.evaluate_threshold(probabilities, labels, threshold=0.5)

    # predicted positive: 0.9, 0.8, 0.95 (indices 0,1,4); true labels there: 1,0,1
    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_negative"] == 1
    assert result["precision"] == 2 / 3
    assert result["recall"] == 2 / 3
    assert result["false_positive_rate"] == 1 / 2
    assert result["sample_count"] == 5


def test_evaluate_threshold_handles_no_predicted_positives():
    probabilities = [0.1, 0.2]
    labels = [0, 1]

    result = threshold_script.evaluate_threshold(probabilities, labels, threshold=0.9)

    assert result["predicted_positive_count"] == 0
    assert result["precision"] == 0.0


def test_select_threshold_picks_smallest_qualifying_threshold():
    results = [
        {"threshold": 0.5, "precision": 0.7, "predicted_positive_count": 10},
        {"threshold": 0.6, "precision": 0.92, "predicted_positive_count": 8},
        {"threshold": 0.7, "precision": 0.95, "predicted_positive_count": 5},
    ]

    selected = threshold_script.select_threshold(results, precision_target=0.90)

    assert selected == 0.6


def test_select_threshold_returns_none_when_no_threshold_qualifies():
    results = [
        {"threshold": 0.5, "precision": 0.5, "predicted_positive_count": 10},
        {"threshold": 0.9, "precision": 0.8, "predicted_positive_count": 2},
    ]

    selected = threshold_script.select_threshold(results, precision_target=0.90)

    assert selected is None


def test_select_threshold_ignores_zero_predicted_positive_thresholds():
    results = [{"threshold": 0.99, "precision": 0.0, "predicted_positive_count": 0}]

    selected = threshold_script.select_threshold(results, precision_target=0.90)

    assert selected is None


def test_candidate_thresholds_are_sorted_ascending():
    assert list(threshold_script.CANDIDATE_THRESHOLDS) == sorted(threshold_script.CANDIDATE_THRESHOLDS)


def test_saved_threshold_analysis_artifact_matches_selected_config():
    """Sanity-check the already-generated artifact reflects the documented selection (0.70)."""
    import json

    if not threshold_script.DEFAULT_OUTPUT_PATH.exists():
        return  # artifact not generated in this environment - skip rather than fail
    data = json.loads(threshold_script.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    assert data["selected_threshold"] == 0.70
    assert data["escalation_enabled"] is True
