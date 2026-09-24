"""Evaluation library for the Task 7 V5 model comparison (pure functions over arrays / TrainingMatrixV5).

Primary evaluation is GROUPED BY ENVIRONMENT: samples of one `environment_id` (which includes every pair
member) never appear on both sides of a split. A random split exists only as a leakage diagnostic.

All data is synthetic; nothing here claims real-world wildfire detection accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.ml.fire_detection.fire_detection_dataset_v5 import (
    grouped_environment_folds,
    leave_one_no_fire_subtype_out,
    leave_one_regime_out,
    row_from_sample,
)
from src.ml.fire_detection.fire_detection_model_config_v5 import (
    BOOTSTRAP_REPEATS,
    CONFIRM_MIN_RECALL,
    CONFIRM_PRECISION_TARGET,
    DEFAULT_THRESHOLD,
    GATE_V5,
    HIGH_CONFIDENCE_FALSE_POSITIVE_THRESHOLD,
    N_ENVIRONMENT_FOLDS,
    PERMUTATION_BLOCKS_V5,
    PERMUTATION_REPEATS,
    RANDOM_SEED_V5,
    SIMPLICITY_MARGIN_ROC_AUC,
    SIMPLICITY_RANK_V5,
    SPARSE_HIGH_EXTREME,
    SPARSE_LOW_EXTREME,
    SPARSE_REGIME,
    PERSISTENT_REGIME,
    SPARSE_UNCERTAIN_BAND,
    SUSPECT_RECALL_TARGET,
    THRESHOLD_GRID,
)
from src.ml.fire_detection.fire_detection_model_v5 import TrainingMatrixV5
from src.models.fire_detection_status import FireDetectionStatus

METRIC_KEYS = ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier")


# ---------------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------------


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Confusion-matrix metrics at `threshold` (prediction = score >= threshold) plus ROC-AUC, PR-AUC and Brier of the scores."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    positive = y_true == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    two_classes = len(set(y_true.tolist())) == 2
    return {
        "rows": int(len(y_true)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, scores)) if two_classes else None,
        "pr_auc": float(average_precision_score(y_true, scores)) if two_classes else None,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else None,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "brier": float(np.mean((scores - y_true) ** 2)) if len(y_true) else None,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def summarize_folds(fold_metrics: Sequence[dict], keys: Iterable[str] = METRIC_KEYS) -> dict:
    """mean / std (population) / min / max of each metric across folds."""
    summary = {}
    for key in keys:
        values = np.array([fold[key] for fold in fold_metrics if fold.get(key) is not None], dtype=float)
        summary[key] = (
            {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max())}
            if len(values)
            else None
        )
    return summary


def calibration_report(y: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict:
    """Brier score, expected calibration error and equal-width reliability bins."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    prevalence = float(y.mean())
    brier = float(np.mean((scores - y) ** 2))
    baseline = prevalence * (1 - prevalence)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece = [], 0.0
    for index in range(n_bins):
        last = index == n_bins - 1
        mask = (scores >= edges[index]) & ((scores <= edges[index + 1]) if last else (scores < edges[index + 1]))
        count = int(mask.sum())
        if count:
            mean_predicted, observed = float(scores[mask].mean()), float(y[mask].mean())
            ece += count / len(y) * abs(mean_predicted - observed)
        else:
            mean_predicted = observed = None
        bins.append(
            {"lower": float(edges[index]), "upper": float(edges[index + 1]), "count": count,
             "mean_predicted": mean_predicted, "observed_fire_fraction": observed}
        )
    return {
        "rows": int(len(y)),
        "brier_score": brier,
        "brier_score_constant_prevalence_baseline": baseline,
        "brier_skill_score": float(1 - brier / baseline) if baseline else None,
        "expected_calibration_error": float(ece),
        "bins": bins,
    }


# ---------------------------------------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------------------------------------


def non_sparse_mask(matrix: TrainingMatrixV5) -> np.ndarray:
    return matrix.regimes != SPARSE_REGIME


def hard_negative_mask(matrix: TrainingMatrixV5) -> np.ndarray:
    """A no-fire row carrying fire-looking evidence: a nominal/high hotspot or a MODERATE/STRONG report.

    (Mirrors the V4 definition. Sparse rows are excluded from the hard-negative GATE metric by the caller.)
    """
    fire_looking = (
        (matrix.feature("satellite_nominal_count") + matrix.feature("satellite_high_count") > 0)
        | (matrix.feature("news_moderate_count") + matrix.feature("news_strong_count") > 0)
    )
    return (matrix.y == 0) & fire_looking


def satellite_total(matrix: TrainingMatrixV5) -> np.ndarray:
    return matrix.feature("satellite_low_count") + matrix.feature("satellite_nominal_count") + matrix.feature("satellite_high_count")


def news_total(matrix: TrainingMatrixV5) -> np.ndarray:
    return sum(matrix.feature(name) for name in (
        "news_none_count", "news_weak_count", "news_moderate_count", "news_strong_count", "news_unknown_count"))


# ---------------------------------------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NamedSplit:
    name: str
    train: np.ndarray
    validation: np.ndarray


def _as_splits(named: Iterable[tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]]) -> tuple[NamedSplit, ...]:
    return tuple(NamedSplit(name, np.array(train), np.array(validation)) for name, (train, validation) in named)


def environment_splits(matrix: TrainingMatrixV5, n_splits: int = N_ENVIRONMENT_FOLDS) -> tuple[NamedSplit, ...]:
    folds = grouped_environment_folds(matrix.rows, n_splits)
    return _as_splits((f"environment_fold_{i}", fold) for i, fold in enumerate(folds))


def regime_splits(matrix: TrainingMatrixV5) -> tuple[NamedSplit, ...]:
    return _as_splits(leave_one_regime_out(matrix.rows))


def no_fire_subtype_splits(matrix: TrainingMatrixV5) -> tuple[NamedSplit, ...]:
    return _as_splits(leave_one_no_fire_subtype_out(matrix.rows))


def assert_splits_are_grouped(splits: Sequence[NamedSplit], group_values: np.ndarray, partition: bool = True) -> None:
    """No group on both sides of any split; when `partition`, every row is validated exactly once."""
    seen = np.zeros(len(group_values), dtype=int)
    for split in splits:
        if set(group_values[split.train].tolist()) & set(group_values[split.validation].tolist()):
            raise ValueError(f"{split.name}: a group appears in both training and validation.")
        seen[split.validation] += 1
    if partition and not np.all(seen == 1):
        raise ValueError("validation folds must cover every row exactly once.")


def assert_pairs_never_split(matrix: TrainingMatrixV5, splits: Sequence[NamedSplit]) -> None:
    """Both members of every pair are always on the same side of every split."""
    for split in splits:
        side = np.zeros(len(matrix.y), dtype=int)
        side[split.validation] = 1
        for pair_id in np.unique(matrix.pair_ids[matrix.pair_ids >= 0]):
            members = np.flatnonzero(matrix.pair_ids == pair_id)
            if len(set(side[members].tolist())) != 1:
                raise ValueError(f"{split.name}: pair {pair_id} is split across training and validation.")


def assert_folds_represent_regimes_and_labels(matrix: TrainingMatrixV5, splits: Sequence[NamedSplit]) -> None:
    for split in splits:
        for regime in np.unique(matrix.regimes):
            labels = set(matrix.y[split.validation][matrix.regimes[split.validation] == regime].tolist())
            if labels != {0, 1}:
                raise ValueError(f"{split.name}: regime {regime!r} lacks a label in validation.")


# ---------------------------------------------------------------------------------------------------------
# fitting: a "fit-predict" abstraction so every analysis uses the same (possibly calibrated) candidate
# ---------------------------------------------------------------------------------------------------------

CALIBRATION_VARIANTS = ("none", "sigmoid", "isotonic")
_INNER_CALIBRATION_FOLDS = 3


def make_fit_predict(
    build_pipeline: Callable[[], object],
    columns: np.ndarray,
    y: np.ndarray,
    environments: np.ndarray,
    variant: str = "none",
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, object]]:
    """fit_predict(train_idx, predict_idx) -> (P(fire) for predict_idx, fitted model).

    variant "none": the plain pipeline. "sigmoid" / "isotonic": CalibratedClassifierCV whose calibration data
    are the inner GROUPED (by environment) validation folds of the TRAINING rows only - the predicted rows are
    never used to fit either the classifier or its calibrator.
    """
    if variant not in CALIBRATION_VARIANTS:
        raise ValueError(f"variant must be one of {CALIBRATION_VARIANTS}, got {variant!r}")

    def fit_predict(train: np.ndarray, predict: np.ndarray):
        if variant == "none":
            model = build_pipeline()
        else:
            inner = list(GroupKFold(n_splits=_INNER_CALIBRATION_FOLDS).split(columns[train], y[train], environments[train]))
            model = CalibratedClassifierCV(build_pipeline(), method=variant, cv=inner, ensemble=True)
        model.fit(columns[train], y[train])
        return model.predict_proba(columns[predict])[:, 1], model

    return fit_predict


def out_of_fold_scores(
    fit_predict: Callable, n_rows: int, splits: Sequence[NamedSplit]
) -> tuple[np.ndarray, list]:
    """Grouped out-of-fold P(fire) for every row (each row scored by the model that never saw its group)."""
    scores = np.full(n_rows, np.nan)
    fitted = []
    for split in splits:
        fold_scores, model = fit_predict(split.train, split.validation)
        scores[split.validation] = fold_scores
        fitted.append(model)
    if np.isnan(scores).any():
        raise ValueError("out-of-fold splits do not cover every row.")
    return scores, fitted


def per_split_metrics(y: np.ndarray, scores: np.ndarray, splits: Sequence[NamedSplit], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    return [{"split": s.name, **binary_metrics(y[s.validation], scores[s.validation], threshold)} for s in splits]


# ---------------------------------------------------------------------------------------------------------
# operating points
# ---------------------------------------------------------------------------------------------------------


def suspected_threshold(y: np.ndarray, scores: np.ndarray, eligible_positive: np.ndarray, target: float = SUSPECT_RECALL_TARGET) -> float | None:
    """The HIGHEST grid threshold whose recall on `eligible_positive` (non-sparse fires) is >= target; None if unattainable."""
    positives = scores[(y == 1) & eligible_positive]
    for threshold in reversed(THRESHOLD_GRID):
        if float(np.mean(positives >= threshold)) >= target:
            return float(threshold)
    return None


def operating_point_report(matrix: TrainingMatrixV5, scores: np.ndarray, threshold: float) -> dict:
    """Behaviour at one threshold: overall, non-sparse, hard negatives (non-sparse), per regime."""
    y = matrix.y
    ns = non_sparse_mask(matrix)
    hard = hard_negative_mask(matrix) & ns
    predicted = scores >= threshold
    per_regime = {}
    for regime in sorted(set(matrix.regimes.tolist())):
        mask = matrix.regimes == regime
        m = binary_metrics(y[mask], scores[mask], threshold)
        per_regime[regime] = {k: m[k] for k in ("rows", "precision", "recall", "f1", "false_positive_rate", "roc_auc", "brier")}
    negatives_ns = ns & (y == 0)
    return {
        "threshold": float(threshold),
        "all_rows": binary_metrics(y, scores, threshold),
        "non_sparse_rows": binary_metrics(y[ns], scores[ns], threshold),
        "non_sparse_positive_recall": float(np.mean(predicted[(y == 1) & ns])),
        "non_sparse_false_positive_rate": float(np.mean(predicted[negatives_ns])),
        "hard_negative_rows": int(hard.sum()),
        "hard_negative_fpr": float(np.mean(predicted[hard])) if hard.any() else None,
        "hard_negative_fpr_per_fold_note": "see grouped_environment.per_fold_at_suspected_threshold",
        "per_regime": per_regime,
    }


def confirmed_probability_analysis(matrix: TrainingMatrixV5, scores: np.ndarray) -> dict:
    """Probability-only CONFIRMED: is precision >= 0.90 attainable at meaningful recall? Reported, never forced."""
    y = matrix.y
    ns = non_sparse_mask(matrix)
    table = []
    for threshold in THRESHOLD_GRID:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        flagged = int(predicted.sum())
        ns_flagged = int((predicted & ns).sum())
        table.append(
            {
                "threshold": float(threshold),
                "flagged_rows": flagged,
                "precision": tp / flagged if flagged else None,
                "recall": tp / int((y == 1).sum()),
                "non_sparse_precision": (int(np.sum(predicted & ns & (y == 1))) / ns_flagged) if ns_flagged else None,
                "false_positives": flagged - tp,
            }
        )
    meeting = [r for r in table if r["precision"] is not None and r["precision"] >= CONFIRM_PRECISION_TARGET and r["recall"] >= CONFIRM_MIN_RECALL]
    with_recall = [r for r in table if r["precision"] is not None and r["recall"] >= CONFIRM_MIN_RECALL]
    best_precision = max(with_recall, key=lambda r: (r["precision"], r["recall"])) if with_recall else None
    return {
        "precision_target": CONFIRM_PRECISION_TARGET,
        "minimum_meaningful_recall": CONFIRM_MIN_RECALL,
        "precision_target_attainable": bool(meeting),
        "candidate": max(meeting, key=lambda r: r["recall"]) if meeting else None,
        "highest_precision_at_minimum_recall": best_precision,
        "table": [row for row in table if round(row["threshold"] * 100) % 5 == 0],
    }


# ---------------------------------------------------------------------------------------------------------
# paired / sparse / regime analyses
# ---------------------------------------------------------------------------------------------------------


def paired_analysis(matrix: TrainingMatrixV5, scores: np.ndarray) -> dict:
    """Pairwise ranking accuracy P(P_fire_member > P_nofire_member) (ties count 1/2) and mean probability margin."""
    outcomes: dict[str, list[tuple[float, float]]] = {}
    for pair_id in np.unique(matrix.pair_ids[matrix.pair_ids >= 0]):
        members = np.flatnonzero(matrix.pair_ids == pair_id)
        fire = members[matrix.y[members] == 1]
        no_fire = members[matrix.y[members] == 0]
        if len(fire) != 1 or len(no_fire) != 1:
            raise ValueError(f"pair {pair_id} must hold exactly one fire and one no-fire row.")
        outcomes.setdefault(str(matrix.pair_types[fire[0]]), []).append((float(scores[fire[0]]), float(scores[no_fire[0]])))

    def summarize(pairs: list[tuple[float, float]]) -> dict:
        positive = np.array([a for a, _ in pairs])
        negative = np.array([b for _, b in pairs])
        return {
            "pairs": len(pairs),
            "pairwise_ranking_accuracy": float(np.mean((positive > negative) + 0.5 * (positive == negative))),
            "mean_probability_margin": float(np.mean(positive - negative)),
            "mean_p_fire_member": float(positive.mean()),
            "mean_p_no_fire_member": float(negative.mean()),
        }

    every = [pair for pairs in outcomes.values() for pair in pairs]
    return {"overall": summarize(every), "by_pair_type": {name: summarize(pairs) for name, pairs in sorted(outcomes.items())}}


def sparse_evidence_report(matrix: TrainingMatrixV5, scores: np.ndarray) -> dict:
    """Uncertainty behaviour on the intentionally ambiguous regime (calibration, not accuracy)."""
    mask = matrix.regimes == SPARSE_REGIME
    y, p = matrix.y[mask], scores[mask]
    low, high = SPARSE_UNCERTAIN_BAND
    calibration = calibration_report(y, p)
    extreme_high = p > SPARSE_HIGH_EXTREME
    extreme_low = p < SPARSE_LOW_EXTREME
    return {
        "rows": int(mask.sum()),
        "brier_score": calibration["brier_score"],
        "brier_score_of_constant_0_5": 0.25,
        "expected_calibration_error": calibration["expected_calibration_error"],
        "roc_auc": float(roc_auc_score(y, p)),
        "mean_predicted_p": float(p.mean()),
        "fire_share": float(y.mean()),
        "fraction_uncertain_0_35_to_0_65": float(np.mean((p >= low) & (p <= high))),
        "fraction_p_below_0_10": float(np.mean(extreme_low)),
        "fraction_p_above_0_90": float(np.mean(extreme_high)),
        "observed_fire_fraction_when_p_above_0_90": float(y[extreme_high].mean()) if extreme_high.any() else None,
        "observed_fire_fraction_when_p_below_0_10": float(y[extreme_low].mean()) if extreme_low.any() else None,
        "probability_histogram_10_bins": np.histogram(p, bins=10, range=(0.0, 1.0))[0].astype(int).tolist(),
        "reliability_bins": calibration["bins"],
    }


def regime_report(matrix: TrainingMatrixV5, scores: np.ndarray, threshold: float) -> dict:
    return {
        regime: binary_metrics(matrix.y[matrix.regimes == regime], scores[matrix.regimes == regime], threshold)
        for regime in sorted(set(matrix.regimes.tolist()))
    }


# ---------------------------------------------------------------------------------------------------------
# corroboration guardrail candidates (analysis only - nothing is wired into a policy)
# ---------------------------------------------------------------------------------------------------------


def corroboration_analysis(matrix: TrainingMatrixV5, scores: np.ndarray, thresholds: Sequence[float]) -> dict:
    y = matrix.y
    sat, news = satellite_total(matrix), news_total(matrix)
    guardrails = {
        "probability_only": np.ones(len(y), dtype=bool),
        "and_multiple_passes": matrix.feature("satellite_pass_count") >= 2,
        "and_satellite_and_news": (sat > 0) & (news > 0),
        "and_strong_news": matrix.feature("news_strong_count") >= 1,
        "and_multi_pixel": sat >= 2,
        "and_multiple_passes_and_satellite_news": (matrix.feature("satellite_pass_count") >= 2) & (sat > 0) & (news > 0),
    }
    positives = int((y == 1).sum())
    result = {"guardrail_alone": {}, "with_probability": {}}
    for name, mask in guardrails.items():
        flagged = int(mask.sum())
        result["guardrail_alone"][name] = {
            "rows": flagged, "precision": float(y[mask].mean()) if flagged else None, "recall": float(y[mask].sum() / positives),
        }
    for threshold in sorted({round(float(t), 2) for t in thresholds}):
        table = {}
        for name, mask in guardrails.items():
            flagged = (scores >= threshold) & mask
            count = int(flagged.sum())
            table[name] = {
                "flagged_rows": count,
                "precision": float(y[flagged].mean()) if count else None,
                "recall": float(y[flagged].sum() / positives),
                "false_positives": int((y[flagged] == 0).sum()),
            }
        result["with_probability"][f"{threshold:.2f}"] = table
    return result


# ---------------------------------------------------------------------------------------------------------
# rule baseline
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleOutputs:
    predicted_fire: np.ndarray  # SUSPECTED or CONFIRMED
    predicted_confirmed: np.ndarray  # CONFIRMED only
    confidence: np.ndarray  # the rule's own (uncalibrated) confidence, used only as a ranking score


def verify_samples_match_matrix(samples, matrix: TrainingMatrixV5) -> None:
    """The regenerated samples (which carry evidence for the rule calculator) must be exactly the CSV's rows."""
    if len(samples) != len(matrix.rows):
        raise ValueError("Regenerated sample count differs from the CSV.")
    for sample, row in zip(samples, matrix.rows):
        regenerated = row_from_sample(sample)
        if sample.sample_id != row.sample_id or sample.label != row.label or not np.array_equal(
            np.array([np.nan if v is None else v for v in regenerated.features], dtype=float),
            np.array([np.nan if v is None else v for v in row.features], dtype=float),
            equal_nan=True,
        ):
            raise ValueError(f"Regenerated sample {sample.sample_id} does not match the frozen CSV row.")


def rule_baseline_outputs(samples) -> RuleOutputs:
    """The existing deterministic FireDetectionCalculator on each sample's CURRENT candidate (it has no history input)."""
    calculator = FireDetectionCalculator()
    decisions = [calculator.evaluate(sample.candidate.evidence) for sample in samples]
    return RuleOutputs(
        predicted_fire=np.array([d.status is not FireDetectionStatus.NO_EVENT for d in decisions], dtype=float),
        predicted_confirmed=np.array([d.status is FireDetectionStatus.CONFIRMED for d in decisions], dtype=float),
        confidence=np.array([d.confidence for d in decisions], dtype=float),
    )


def _rule_slice(matrix: TrainingMatrixV5, predicted: np.ndarray, mask: np.ndarray) -> dict:
    y = matrix.y[mask]
    p = predicted[mask].astype(bool)
    hard = hard_negative_mask(matrix)[mask]
    tp, fp = int(np.sum(p & (y == 1))), int(np.sum(p & (y == 0)))
    fn, tn = int(np.sum(~p & (y == 1))), int(np.sum(~p & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "rows": int(mask.sum()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "hard_negative_fpr": float(np.mean(p[hard])) if hard.any() else None,
        "predicted_fire_rate": float(p.mean()),
    }


def rule_baseline_report(matrix: TrainingMatrixV5, outputs: RuleOutputs) -> dict:
    ns = non_sparse_mask(matrix)
    everything = np.ones(len(matrix.y), dtype=bool)

    def variant(predicted: np.ndarray) -> dict:
        return {
            "all_rows": _rule_slice(matrix, predicted, everything),
            "non_sparse_rows": _rule_slice(matrix, predicted, ns),
            "per_regime": {r: _rule_slice(matrix, predicted, matrix.regimes == r) for r in sorted(set(matrix.regimes.tolist()))},
            "roc_auc_by_rule_confidence": float(roc_auc_score(matrix.y, outputs.confidence)),
        }

    return {
        "note": "FireDetectionCalculator on each row's CURRENT candidate (no history) against ground-truth labels; "
        "the rule is never trained or tuned on V5.",
        "suspected_or_confirmed_is_fire": variant(outputs.predicted_fire),
        "confirmed_only_is_fire": variant(outputs.predicted_confirmed),
    }


def model_slice_at_threshold(matrix: TrainingMatrixV5, scores: np.ndarray, threshold: float, mask: np.ndarray) -> dict:
    return _rule_slice(matrix, (scores >= threshold).astype(float), mask)


# ---------------------------------------------------------------------------------------------------------
# stress tests
# ---------------------------------------------------------------------------------------------------------


def leave_group_out_report(
    matrix: TrainingMatrixV5,
    splits: Sequence[NamedSplit],
    fit_predict: Callable,
    thresholds: Sequence[float],
) -> dict:
    """Fit on `train`, score the held-out group; metrics at each threshold plus mean P(fire) and high-confidence FPs."""
    report = {}
    for split in splits:
        scores, _ = fit_predict(split.train, split.validation)
        y = matrix.y[split.validation]
        entry = {"rows": int(len(y)), "fire_rows": int(y.sum()), "mean_p_fire": float(scores.mean())}
        two_classes = len(set(y.tolist())) == 2
        entry["roc_auc"] = float(roc_auc_score(y, scores)) if two_classes else None
        negatives = scores[y == 0]
        entry["high_confidence_false_positive_rate"] = (
            float(np.mean(negatives >= HIGH_CONFIDENCE_FALSE_POSITIVE_THRESHOLD)) if len(negatives) else None
        )
        entry["mean_p_fire_on_no_fire_rows"] = float(negatives.mean()) if len(negatives) else None
        entry["at_thresholds"] = {f"{t:.2f}": binary_metrics(y, scores, t) for t in thresholds}
        report[split.name] = entry
    return report


def random_split_diagnostic(
    matrix: TrainingMatrixV5, fit_predict: Callable, grouped_mean_roc_auc: float, seed: int = RANDOM_SEED_V5
) -> dict:
    """DIAGNOSTIC ONLY: one stratified (label x regime) random split. Never used to choose a model."""
    from sklearn.model_selection import train_test_split

    strata = np.array([f"{regime}|{label}" for regime, label in zip(matrix.regimes, matrix.y)])
    train, test = train_test_split(np.arange(len(matrix.y)), test_size=0.2, random_state=seed, stratify=strata)
    scores, _ = fit_predict(np.sort(train), np.sort(test))
    y = matrix.y[np.sort(test)]
    auc = float(roc_auc_score(y, scores))
    shared = len(set(matrix.environments[train].tolist()) & set(matrix.environments[test].tolist()))
    return {
        "label": "DIAGNOSTIC ONLY - never used to select a model",
        "random_split_roc_auc": auc,
        "grouped_environment_mean_roc_auc": grouped_mean_roc_auc,
        "gap_random_minus_grouped": auc - grouped_mean_roc_auc,
        "environments_present_in_both_train_and_test": int(shared),
        "metrics_at_0_50": binary_metrics(y, scores, DEFAULT_THRESHOLD),
    }


# ---------------------------------------------------------------------------------------------------------
# interpretability and audits
# ---------------------------------------------------------------------------------------------------------


def _fold_models(fitted: Sequence) -> list:
    """Underlying pipelines of fitted models (unwrapping CalibratedClassifierCV ensembles)."""
    pipelines = []
    for model in fitted:
        if hasattr(model, "calibrated_classifiers_"):
            pipelines.extend(c.estimator for c in model.calibrated_classifiers_)
        else:
            pipelines.append(model)
    return pipelines


def grouped_permutation_importance(
    fitted: Sequence,
    columns: np.ndarray,
    y: np.ndarray,
    splits: Sequence[NamedSplit],
    feature_names: tuple[str, ...],
    n_repeats: int = PERMUTATION_REPEATS,
    seed: int = RANDOM_SEED_V5,
    blocks: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    """Held-out permutation importance: drop in validation-fold ROC-AUC when one feature (or block) is shuffled."""
    units = {name: [index] for index, name in enumerate(feature_names)}
    block_units = {
        name: [feature_names.index(n) for n in members if n in feature_names]
        for name, members in (blocks or PERMUTATION_BLOCKS_V5).items()
    }
    rng = np.random.default_rng(seed)
    drops = {name: [] for name in (*units, *block_units)}
    for model, split in zip(fitted, splits):
        X_val, y_val = columns[split.validation], y[split.validation]
        baseline = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        for name, cols in {**units, **block_units}.items():
            if not cols:
                continue
            for _ in range(n_repeats):
                permuted = X_val.copy()
                permuted[:, cols] = X_val[rng.permutation(len(X_val))][:, cols]
                drops[name].append(baseline - roc_auc_score(y_val, model.predict_proba(permuted)[:, 1]))
    means = {name: float(np.mean(values)) for name, values in drops.items() if values}
    single = {n: means[n] for n in units if n in means}
    total = sum(max(v, 0.0) for v in single.values())

    def entry(name: str, share_base: dict, share_total: float) -> dict:
        return {
            "mean_auc_drop": means[name],
            "std_auc_drop": float(np.std(drops[name])),
            "share_of_total_positive_drop": (max(means[name], 0.0) / share_total) if share_total and name in share_base else None,
        }

    features = {name: entry(name, single, total) for name in sorted(single, key=single.get, reverse=True)}
    top = next(iter(features))
    return {
        "metric": "drop in held-out (grouped) ROC-AUC when the feature is shuffled; shares are of the total POSITIVE single-feature drop",
        "features": features,
        "blocks": {name: {"mean_auc_drop": means[name], "std_auc_drop": float(np.std(drops[name]))} for name in block_units if name in means},
        "top_feature": top,
        "top_feature_share": features[top]["share_of_total_positive_drop"],
    }


def logistic_coefficient_report(fitted: Sequence, transformed_names: tuple[str, ...]) -> dict | None:
    """Standardized coefficients (features are standardized inside the pipeline), averaged over the fold models."""
    coefficients = []
    for pipeline in _fold_models(fitted):
        classifier = pipeline.named_steps["classifier"]
        if not hasattr(classifier, "coef_"):
            return None
        coefficients.append(classifier.coef_[0])
    mean, std = np.mean(coefficients, axis=0), np.std(coefficients, axis=0)
    order = np.argsort(-np.abs(mean))
    return {
        "note": "positive coefficient = raises P(fire); features are standardized, so magnitudes are comparable",
        "coefficients": [
            {"feature": transformed_names[i], "mean_coefficient": float(mean[i]), "std_across_folds": float(std[i])} for i in order
        ],
    }


def tree_importance_report(fitted: Sequence, transformed_names: tuple[str, ...]) -> dict | None:
    importances = []
    for pipeline in _fold_models(fitted):
        classifier = pipeline.named_steps["classifier"]
        if not hasattr(classifier, "feature_importances_"):
            return None
        importances.append(classifier.feature_importances_)
    mean = np.mean(importances, axis=0)
    order = np.argsort(-mean)
    return {
        "note": "impurity-based importance averaged over the fold models (biased toward continuous features; see permutation importance)",
        "importances": [{"feature": transformed_names[i], "mean_importance": float(mean[i])} for i in order],
    }


NULLABLE_AUDIT_FEATURES = (
    "satellite_night_fraction",
    "news_satellite_lag_minutes",
    "satellite_centroid_stability_km",
    "satellite_frp_trend_per_hour",
    "satellite_brightness_trend_per_hour",
    "satellite_cluster_radius_km",
    "satellite_history_span_minutes",
)


def missingness_association(matrix: TrainingMatrixV5) -> dict:
    """Univariate label association of the MISSINGNESS of each nullable feature, overall and inside each regime."""
    result = {}
    for name in NULLABLE_AUDIT_FEATURES:
        missing = np.isnan(matrix.feature(name))
        entry = {"missing_rate": float(missing.mean())}
        if 0 < missing.sum() < len(missing):
            entry["fire_share_when_missing"] = float(matrix.y[missing].mean())
            entry["fire_share_when_present"] = float(matrix.y[~missing].mean())
            entry["missingness_indicator_auc"] = float(roc_auc_score(matrix.y, missing.astype(float)))
        else:
            entry.update(fire_share_when_missing=None, fire_share_when_present=None, missingness_indicator_auc=None)
        within = {}
        for regime in sorted(set(matrix.regimes.tolist())):
            mask = matrix.regimes == regime
            m = missing[mask]
            if 50 <= m.sum() <= mask.sum() - 50:
                within[regime] = {
                    "missing_rate": float(m.mean()),
                    "missingness_indicator_auc": float(roc_auc_score(matrix.y[mask], m.astype(float))),
                }
        entry["within_regime"] = within
        result[name] = entry
    return result


def missingness_permutation_audit(
    fitted: Sequence, columns: np.ndarray, y: np.ndarray, splits: Sequence[NamedSplit],
    feature_names: tuple[str, ...], seed: int = RANDOM_SEED_V5, n_repeats: int = 3,
) -> dict:
    """Does the model use a nullable feature's VALUES or only WHETHER it is missing?

    For each nullable feature and validation fold, two held-out ROC-AUC drops:
      value_drop        shuffle the observed values among the rows that have one (missingness pattern kept)
      missingness_drop  shuffle WHICH rows are missing (observed values re-dealt; overall missing rate kept)
    A model that imputes (LR / RF) cannot see missingness itself, so its missingness_drop reflects only how the
    median substitution interacts with the values.
    """
    rng = np.random.default_rng(seed)
    result = {}
    for name in NULLABLE_AUDIT_FEATURES:
        if name not in feature_names:
            continue
        column = feature_names.index(name)
        value_drops, mask_drops = [], []
        for model, split in zip(fitted, splits):
            X_val, y_val = columns[split.validation], y[split.validation]
            missing = np.isnan(X_val[:, column])
            if missing.all() or not missing.any():
                continue
            baseline = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            observed_idx = np.flatnonzero(~missing)
            for _ in range(n_repeats):
                shuffled = X_val.copy()
                shuffled[observed_idx, column] = X_val[rng.permutation(observed_idx), column]
                value_drops.append(baseline - roc_auc_score(y_val, model.predict_proba(shuffled)[:, 1]))

                new_missing = missing[rng.permutation(len(missing))]
                pool = X_val[observed_idx, column][rng.permutation(len(observed_idx))]
                moved = X_val.copy()
                moved[:, column] = np.nan
                keep = np.flatnonzero(~new_missing)
                moved[keep, column] = pool[: len(keep)]
                mask_drops.append(baseline - roc_auc_score(y_val, model.predict_proba(moved)[:, 1]))
        if value_drops:
            result[name] = {
                "mean_auc_drop_shuffling_values": float(np.mean(value_drops)),
                "mean_auc_drop_shuffling_missingness": float(np.mean(mask_drops)),
            }
    return result


# ---------------------------------------------------------------------------------------------------------
# ablation helpers
# ---------------------------------------------------------------------------------------------------------


def slice_metrics_at_matched_recall(y: np.ndarray, scores: np.ndarray, target_recall: float = SUSPECT_RECALL_TARGET) -> dict:
    """Within one slice: the highest threshold reaching `target_recall`, and the false-positive rate / precision there."""
    positives = scores[y == 1]
    chosen = None
    for threshold in reversed(THRESHOLD_GRID):
        if float(np.mean(positives >= threshold)) >= target_recall:
            chosen = float(threshold)
            break
    if chosen is None:
        return {"threshold": None, "false_positive_rate": None, "precision": None}
    m = binary_metrics(y, scores, chosen)
    return {"threshold": chosen, "false_positive_rate": m["false_positive_rate"], "precision": m["precision"], "recall": m["recall"]}


def clustered_bootstrap_auc_difference(
    y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, environments: np.ndarray,
    repeats: int = BOOTSTRAP_REPEATS, seed: int = RANDOM_SEED_V5,
) -> dict:
    """AUC(a) - AUC(b) with an environment-clustered bootstrap 95 % interval (resamples whole environments)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(environments)
    index_by_env = {env: np.flatnonzero(environments == env) for env in unique}
    differences = []
    for _ in range(repeats):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_env[env] for env in chosen])
        if len(set(y[idx].tolist())) < 2:
            continue
        differences.append(roc_auc_score(y[idx], scores_a[idx]) - roc_auc_score(y[idx], scores_b[idx]))
    point = roc_auc_score(y, scores_a) - roc_auc_score(y, scores_b)
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "auc_difference": float(point),
        "bootstrap_95_ci": [float(low), float(high)],
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "bootstrap_repeats": len(differences),
    }


# ---------------------------------------------------------------------------------------------------------
# gate and selection
# ---------------------------------------------------------------------------------------------------------


def evaluate_gate(candidate: dict, rule: dict, gate: dict = GATE_V5) -> dict:
    """Apply the predefined gate. `candidate` carries the measured values; every criterion is critical."""
    criteria = {}

    def check(name: str, value, passed: bool | None, requirement: str, note: str | None = None) -> None:
        criteria[name] = {"value": value, "requirement": requirement, "passed": passed, **({"note": note} if note else {})}

    def at_least(value, minimum):
        return None if value is None else bool(value >= minimum)

    def at_most(value, maximum):
        return None if value is None else bool(value <= maximum)

    check("grouped_environment_mean_roc_auc", candidate["grouped_mean_roc_auc"], at_least(candidate["grouped_mean_roc_auc"], gate["grouped_environment_mean_roc_auc_min"]),
          f">= {gate['grouped_environment_mean_roc_auc_min']}")
    check("grouped_environment_worst_fold_roc_auc", candidate["grouped_worst_fold_roc_auc"], at_least(candidate["grouped_worst_fold_roc_auc"], gate["grouped_environment_worst_fold_roc_auc_min"]),
          f">= {gate['grouped_environment_worst_fold_roc_auc_min']}")
    check("non_sparse_positive_recall", candidate["non_sparse_positive_recall"], at_least(candidate["non_sparse_positive_recall"], gate["non_sparse_positive_recall_min"]),
          f">= {gate['non_sparse_positive_recall_min']} at the SUSPECTED operating point",
          None if candidate["suspected_threshold"] is not None else "no threshold reaches the recall target")
    check("hard_negative_fpr", candidate["hard_negative_fpr"], at_most(candidate["hard_negative_fpr"], gate["hard_negative_fpr_max"]),
          f"<= {gate['hard_negative_fpr_max']} at the SUSPECTED operating point (sparse excluded)")
    check("persistent_thermal_roc_auc", candidate["persistent_roc_auc"], at_least(candidate["persistent_roc_auc"], gate["persistent_thermal_roc_auc_min"]),
          f">= {gate['persistent_thermal_roc_auc_min']}")
    brier = candidate["brier_score"]
    check("brier_score", brier, None if brier is None else bool(brier < gate["brier_score_max_exclusive"]), f"< {gate['brier_score_max_exclusive']}")
    check("expected_calibration_error", candidate["expected_calibration_error"], at_most(candidate["expected_calibration_error"], gate["expected_calibration_error_max"]),
          f"<= {gate['expected_calibration_error_max']}")
    limit = gate["top_feature_importance_share_max"] + gate["top_feature_importance_share_tolerance"]
    check("top_feature_importance_share", candidate["top_feature_share"], at_most(candidate["top_feature_share"], limit),
          f"<= {gate['top_feature_importance_share_max']} (+{gate['top_feature_importance_share_tolerance']} tolerance)")

    rule_slice = rule["non_sparse_rows"]
    model_slice = candidate["non_sparse_slice_at_suspected"]
    if model_slice is None or rule_slice["hard_negative_fpr"] in (None, 0):
        check("beats_rules_hard_negative_fpr", None, None, "model hard-negative FPR <= 0.75 x the rule's", "not computable")
        check("beats_rules_recall_not_collapsed", None, None, "model recall >= rule recall - 0.15", "not computable")
        check("beats_rules_f1_not_worse", None, None, "model F1 >= rule F1", "not computable")
    else:
        ratio_limit = gate["vs_rules_hard_negative_fpr_max_ratio"] * rule_slice["hard_negative_fpr"]
        check("beats_rules_hard_negative_fpr", model_slice["hard_negative_fpr"], at_most(model_slice["hard_negative_fpr"], ratio_limit),
              f"<= {gate['vs_rules_hard_negative_fpr_max_ratio']} x rule hard-negative FPR ({rule_slice['hard_negative_fpr']:.3f}) = {ratio_limit:.3f}")
        recall_floor = rule_slice["recall"] - gate["vs_rules_recall_max_drop"]
        check("beats_rules_recall_not_collapsed", model_slice["recall"], at_least(model_slice["recall"], recall_floor),
              f">= rule recall ({rule_slice['recall']:.3f}) - {gate['vs_rules_recall_max_drop']} = {recall_floor:.3f}")
        check("beats_rules_f1_not_worse", model_slice["f1"], at_least(model_slice["f1"], rule_slice["f1"] + gate["vs_rules_f1_min_delta"]),
              f">= rule F1 ({rule_slice['f1']:.3f}) + {gate['vs_rules_f1_min_delta']}")

    failed = [name for name, c in criteria.items() if c["passed"] is not True]
    return {"criteria": criteria, "passed": not failed, "failed_criteria": failed}


def select_model(candidates: Sequence[dict]) -> dict:
    """Among PASSING candidates prefer the simplest within SIMPLICITY_MARGIN_ROC_AUC of the best; else NONE."""
    passing = [c for c in candidates if c["gate"]["passed"]]
    if not passing:
        return {
            "selected_model": None,
            "reason": "no model passed the predefined acceptance gate; no runtime artifact is saved",
            "passing_models": [],
        }
    best = max(c["grouped_mean_roc_auc"] for c in passing)
    close = [c for c in passing if best - c["grouped_mean_roc_auc"] <= SIMPLICITY_MARGIN_ROC_AUC + 1e-9]
    chosen = min(close, key=lambda c: (SIMPLICITY_RANK_V5[c["model_key"]], -c["grouped_mean_roc_auc"]))
    return {
        "selected_model": chosen["model_key"],
        "reason": (
            f"{chosen['model_key']} passed the gate and is within {SIMPLICITY_MARGIN_ROC_AUC} grouped ROC-AUC of the best passing model "
            f"({best:.4f}); the simplest such model is preferred."
        ),
        "passing_models": [c["model_key"] for c in passing],
    }


def to_builtin(value):
    """Recursively convert numpy scalars/arrays to JSON-serialisable builtins (NaN -> None)."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
