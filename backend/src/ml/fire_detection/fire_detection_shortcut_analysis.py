"""Univariate shortcut and feature-vector collision analysis (Task 4 Parts 28-29).

Generic over any (features, labels, feature_names) triple, so it applies
identically to V1, V2, or V3 datasets - used to compare collision rates
across dataset versions with the exact same logic, and to flag any single
feature that alone nearly determines the label (a synthetic-shortcut smell).
"""
from __future__ import annotations

from dataclasses import dataclass
import statistics

from sklearn.metrics import roc_auc_score

SUSPICIOUS_AUC_THRESHOLD = 0.95


@dataclass(frozen=True)
class UnivariateFeatureStats:
    """One feature's standalone discriminatory power (Part 28)."""

    feature_name: str
    mean_label_0: float
    mean_label_1: float
    median_label_0: float
    median_label_1: float
    min_label_0: float
    max_label_0: float
    min_label_1: float
    max_label_1: float
    auc: float  # best-orientation AUC: max(raw_auc, 1 - raw_auc) - a feature can be discriminative in either direction
    is_suspicious_shortcut: bool  # auc >= SUSPICIOUS_AUC_THRESHOLD


def compute_univariate_shortcut_analysis(
    features: list[list[float]],
    labels: list[int],
    feature_names: tuple[str, ...],
) -> tuple[UnivariateFeatureStats, ...]:
    """Per-feature mean/median/range by label, plus a univariate ROC-AUC using that feature alone as the score."""
    results = []
    for index, name in enumerate(feature_names):
        values = [row[index] for row in features]
        values_0 = [value for value, label in zip(values, labels) if label == 0]
        values_1 = [value for value, label in zip(values, labels) if label == 1]

        raw_auc = float(roc_auc_score(labels, values)) if len(set(labels)) == 2 else 0.5
        best_auc = max(raw_auc, 1 - raw_auc)

        results.append(
            UnivariateFeatureStats(
                feature_name=name,
                mean_label_0=statistics.mean(values_0) if values_0 else 0.0,
                mean_label_1=statistics.mean(values_1) if values_1 else 0.0,
                median_label_0=statistics.median(values_0) if values_0 else 0.0,
                median_label_1=statistics.median(values_1) if values_1 else 0.0,
                min_label_0=min(values_0) if values_0 else 0.0,
                max_label_0=max(values_0) if values_0 else 0.0,
                min_label_1=min(values_1) if values_1 else 0.0,
                max_label_1=max(values_1) if values_1 else 0.0,
                auc=best_auc,
                is_suspicious_shortcut=bool(best_auc >= SUSPICIOUS_AUC_THRESHOLD),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class CollisionAnalysisResult:
    """Exact feature-vector collisions between opposite labels (Part 29)."""

    total_rows: int
    distinct_feature_vectors: int
    colliding_vectors: int  # distinct feature vectors that occur under BOTH labels somewhere in the dataset
    colliding_rows: int  # rows whose exact feature vector also occurs under the opposite label
    collision_rate: float  # colliding_rows / total_rows


def compute_feature_vector_collisions(features: list[list[float]], labels: list[int]) -> CollisionAnalysisResult:
    labels_by_vector: dict[tuple, set[int]] = {}
    count_by_vector: dict[tuple, int] = {}
    for row, label in zip(features, labels):
        key = tuple(row)
        labels_by_vector.setdefault(key, set()).add(label)
        count_by_vector[key] = count_by_vector.get(key, 0) + 1

    colliding_vectors = [key for key, seen_labels in labels_by_vector.items() if len(seen_labels) > 1]
    colliding_rows = sum(count_by_vector[key] for key in colliding_vectors)
    total_rows = len(features)

    return CollisionAnalysisResult(
        total_rows=total_rows,
        distinct_feature_vectors=len(labels_by_vector),
        colliding_vectors=len(colliding_vectors),
        colliding_rows=colliding_rows,
        collision_rate=colliding_rows / total_rows if total_rows else 0.0,
    )
