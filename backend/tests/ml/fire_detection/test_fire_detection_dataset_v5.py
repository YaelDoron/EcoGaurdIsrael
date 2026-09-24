"""Task 6: the V5 dataset file, train/runtime feature parity, reproducibility and grouped-split support."""
from __future__ import annotations

import csv
from datetime import timedelta
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.ml.fire_detection.fire_detection_dataset_v5 import (
    dataset_sha256,
    environment_fold_assignments,
    extract_features_for_sample,
    grouped_environment_folds,
    leave_one_no_fire_subtype_out,
    leave_one_regime_out,
    load_training_dataset_rows_v5,
    pair_indices,
    row_from_sample,
    write_training_dataset_v5,
)
from src.ml.fire_detection.fire_detection_feature_extractor_v5 import FireDetectionFeatureExtractorV5
from src.ml.fire_detection.fire_detection_features_v5 import (
    FIRE_DETECTION_FEATURE_NAMES_V5,
    NULLABLE_FEATURE_NAMES_V5,
    TRAINING_DATA_CSV_COLUMNS_V5,
    FireDetectionFeaturesV5,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import (
    DEFAULT_TRAINING_DATA_SEED_V5,
    FireDetectionTrainingDataGeneratorV5,
    RegimeV5,
)
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_event_history import FireDetectionEventHistory
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_report import WildfireReport
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService, FireDetectionHistoryService

COMMITTED_CSV = Path(__file__).resolve().parents[3] / "data" / "fire_detection" / "training_v5.csv"


def same(a, b) -> bool:
    return len(a) == len(b) and all(
        (x is None and y is None) or (x is not None and y is not None and (x == y or math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)))
        for x, y in zip(a, b)
    )


# --- CSV round trip ---


def test_csv_has_exactly_the_documented_columns_and_ends_with_the_label(tmp_path, v5_samples):
    path = write_training_dataset_v5(v5_samples[:50], tmp_path / "v5.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == TRAINING_DATA_CSV_COLUMNS_V5
    assert header[-1] == "label" and len(header) == 8 + 25 + 1
    assert not [c for c in header if "danger" in c or "ffwi" in c]


def test_write_then_load_round_trips_metadata_features_and_label(tmp_path, v5_samples):
    subset = v5_samples[:400]
    rows = load_training_dataset_rows_v5(write_training_dataset_v5(subset, tmp_path / "v5.csv"))
    assert len(rows) == len(subset)
    for sample, row in zip(subset, rows):
        expected = row_from_sample(sample)
        assert (row.sample_id, row.seed, row.environment_id, row.regime, row.latent_subtype, row.pair_id, row.pair_type, row.label) == (
            expected.sample_id, expected.seed, expected.environment_id, expected.regime, expected.latent_subtype,
            expected.pair_id, expected.pair_type, expected.label,
        )
        assert row.as_of_utc == sample.as_of
        assert same(row.features, expected.features)
        row.to_features_v5()  # every stored row satisfies the validated contract


def test_missing_values_are_empty_cells_and_load_as_none_never_zero(tmp_path, v5_samples):
    path = write_training_dataset_v5(v5_samples[:600], tmp_path / "v5.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    empties = {name: sum(1 for r in records if r[name] == "") for name in FIRE_DETECTION_FEATURE_NAMES_V5}
    assert all(count == 0 for name, count in empties.items() if name not in NULLABLE_FEATURE_NAMES_V5)
    assert all(count > 0 for name, count in empties.items() if name in NULLABLE_FEATURE_NAMES_V5)
    rows = load_training_dataset_rows_v5(path)
    for row, record in zip(rows, records):
        for name in NULLABLE_FEATURE_NAMES_V5:
            assert (row.feature(name) is None) == (record[name] == "")


def test_loading_rejects_a_wrong_header_and_an_empty_required_cell(tmp_path, v5_samples):
    path = write_training_dataset_v5(v5_samples[:5], tmp_path / "v5.csv")
    text = path.read_text(encoding="utf-8")
    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text(text.replace("satellite_frp_sum", "satellite_frp_total", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected V5 dataset columns"):
        load_training_dataset_rows_v5(bad_header)
    lines = text.splitlines()
    fields = lines[1].split(",")
    fields[8] = ""  # satellite_low_count is not nullable
    (tmp_path / "bad_cell.csv").write_text("\n".join([lines[0], ",".join(fields), *lines[2:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_training_dataset_rows_v5(tmp_path / "bad_cell.csv")


# --- reproducibility ---


def test_regenerating_the_same_seed_gives_a_byte_identical_file(tmp_path):
    first = write_training_dataset_v5(FireDetectionTrainingDataGeneratorV5(seed=11).generate(500), tmp_path / "a.csv")
    second = write_training_dataset_v5(FireDetectionTrainingDataGeneratorV5(seed=11).generate(500), tmp_path / "b.csv")
    other = write_training_dataset_v5(FireDetectionTrainingDataGeneratorV5(seed=12).generate(500), tmp_path / "c.csv")
    assert dataset_sha256(first) == dataset_sha256(second) == hashlib.sha256(first.read_bytes()).hexdigest()
    assert dataset_sha256(first) != dataset_sha256(other)
    assert b"\r" not in first.read_bytes()  # fixed line endings on every OS


@pytest.mark.skipif(not COMMITTED_CSV.exists(), reason="training_v5.csv has not been generated")
def test_the_generated_training_v5_csv_matches_a_fresh_generation_of_seed_42(tmp_path, v5_samples):
    fresh = write_training_dataset_v5(v5_samples, tmp_path / "fresh.csv")
    assert dataset_sha256(COMMITTED_CSV) == dataset_sha256(fresh)
    assert DEFAULT_TRAINING_DATA_SEED_V5 == 42
    assert len(load_training_dataset_rows_v5(COMMITTED_CSV)) == 10000


# --- train / runtime parity ---


def test_training_features_come_from_the_canonical_extractor_for_every_row(v5_samples, v5_rows):
    extractor = FireDetectionFeatureExtractorV5()  # a fresh runtime-style instance
    for sample, row in zip(v5_samples, v5_rows):
        runtime = extractor.extract(sample.candidate, sample.history)
        assert same(row.features, runtime.to_nullable_tuple()), sample.sample_id
        assert same(runtime.to_nullable_tuple(), extract_features_for_sample(sample).to_nullable_tuple())


def test_features_do_not_depend_on_metadata_regime_environment_or_pair(v5_samples):
    """The extractor never sees metadata: the same evidence gives the same features whatever the row says."""
    sample = next(s for s in v5_samples if s.regime is RegimeV5.PERSISTENT_THERMAL)
    relabelled = replace(sample, environment_id=9999, pair_id=None, pair_type=None)
    assert extract_features_for_sample(relabelled).as_tuple() == pytest.approx(extract_features_for_sample(sample).as_tuple(), nan_ok=True)


def test_history_that_repeats_the_candidate_gives_the_same_features_as_history_without_it(v5_samples):
    extractor = FireDetectionFeatureExtractorV5()
    checked = 0
    for sample in (s for s in v5_samples if s.history is not None):
        current = {(e.evidence_type, e.evidence_id) for e in sample.candidate.evidence}
        kept = tuple(e for e in sample.history.evidence if (e.evidence_type, e.evidence_id) not in current)
        with_candidate = extractor.extract(sample.candidate, sample.history)
        history_without = FireDetectionEventHistory(
            fire_event_id=sample.history.fire_event_id,
            as_of=sample.history.as_of,
            window_start=sample.history.window_start,
            evidence=kept,
            satellite_pass_gap_minutes=sample.history.satellite_pass_gap_minutes,
        )
        assert same(with_candidate.to_nullable_tuple(), extractor.extract(sample.candidate, history_without).to_nullable_tuple())
        checked += 1
        if checked == 300:
            break


def _fresh_runtime_stack():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    satellites = SatelliteHotspotRepository(session_factory=factory)
    news = NewsRepository(session_factory=factory)
    events = FireEventRepository(session_factory=factory)
    return engine, satellites, news, events, FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news)


def _persist(sample, satellites, news):
    """Save every evidence item of the sample; return {(type, synthetic id): persisted ref}."""
    refs: dict[tuple, FireEvidenceRef] = {}
    everything = list(sample.candidate.evidence) + (list(sample.history.evidence) if sample.history else [])
    for item in everything:
        key = (item.evidence_type, item.evidence_id)
        if key in refs:
            continue
        if item.evidence_type is FireEvidenceType.SATELLITE:
            satellites.save_hotspot(
                SatelliteHotspot(
                    latitude=item.latitude, longitude=item.longitude, detected_at=item.observed_at,
                    confidence=item.satellite_confidence, frp=item.satellite_frp, brightness=item.satellite_brightness,
                    satellite=item.satellite_name, instrument=item.satellite_instrument, day_night=item.satellite_day_night,
                )
            )
        else:
            news.save_report(
                WildfireReport(
                    source_url=f"https://example.com/{sample.sample_id}/{item.evidence_id}", source_feed="Example",
                    title="Wildfire reported", summary="Smoke.", location_name=None,
                    latitude=item.latitude, longitude=item.longitude, published_at=item.observed_at,
                    fetched_at=item.observed_at, wildfire_signal_strength=item.news_wildfire_signal_strength,
                )
            )
    horizon = sample.as_of + timedelta(minutes=1)
    lookback = 60 * 30
    stored_satellite = {
        (s.hotspot.latitude, s.hotspot.longitude, s.hotspot.detected_at.replace(tzinfo=None)): s.id
        for s in satellites.get_recent_hotspots(as_of=horizon, lookback_minutes=lookback)
    }
    stored_news = {n.report.source_url: n.id for n in news.get_recent_reports(as_of=horizon, lookback_minutes=lookback)}
    for item in everything:
        key = (item.evidence_type, item.evidence_id)
        if item.evidence_type is FireEvidenceType.SATELLITE:
            found = stored_satellite[(item.latitude, item.longitude, item.observed_at.replace(tzinfo=None))]
        else:
            found = stored_news[f"https://example.com/{sample.sample_id}/{item.evidence_id}"]
        refs[key] = FireEvidenceRef(evidence_type=item.evidence_type, evidence_id=found)
    return refs


def test_runtime_path_reproduces_the_training_features_through_the_real_repositories(v5_samples, v5_rows):
    """persisted evidence -> evidence service -> FireEvent -> FireDetectionHistoryService -> extractor == training row."""
    chosen: list[int] = []
    wanted = {regime: 3 for regime in RegimeV5}
    for index, sample in enumerate(v5_samples):
        if wanted[sample.regime] > 0 and sample.label in (0, 1):
            wanted[sample.regime] -= 1
            chosen.append(index)
    assert len(chosen) == 18

    extractor = FireDetectionFeatureExtractorV5()
    for index in chosen:
        sample, row = v5_samples[index], v5_rows[index]
        engine, satellites, news, events, evidence_service = _fresh_runtime_stack()
        try:
            refs = _persist(sample, satellites, news)
            candidate_refs = tuple(refs[(e.evidence_type, e.evidence_id)] for e in sample.candidate.evidence)
            candidate = FireDetectionCandidate(evidence_service.resolve_evidence_refs(candidate_refs))

            history = None
            if sample.history is not None:
                history_refs = tuple(refs[(e.evidence_type, e.evidence_id)] for e in sample.history.evidence)
                stored = events.create_event(
                    FireEvent(
                        latitude=candidate.evidence[0].latitude, longitude=candidate.evidence[0].longitude,
                        detected_at=min(e.observed_at for e in sample.history.evidence), updated_at=sample.as_of,
                        status=FireEventStatus.SUSPECTED, detection_confidence=0.5, methodology="test", methodology_version="1",
                    ),
                    supporting_evidence=history_refs,
                )
                history = FireDetectionHistoryService(evidence_service, events).build_history(stored, sample.as_of)

            runtime = extractor.extract(candidate, history)
            assert same(runtime.to_nullable_tuple(), row.features), (sample.regime, sample.sample_id)
        finally:
            engine.dispose()


# --- grouped-evaluation support (no model is trained) ---


def test_environment_folds_hold_out_whole_environments_and_keep_regime_and_label_representation(v5_rows):
    folds = grouped_environment_folds(v5_rows, 5)
    assert len(folds) == 5
    seen_validation: set[int] = set()
    for train, validation in folds:
        train_envs = {v5_rows[i].environment_id for i in train}
        validation_envs = {v5_rows[i].environment_id for i in validation}
        assert not train_envs & validation_envs  # no environment on both sides
        assert not set(validation) & seen_validation
        seen_validation |= set(validation)
        assert len(train) + len(validation) == len(v5_rows)
        for regime in RegimeV5:
            labels = {v5_rows[i].label for i in validation if v5_rows[i].regime == regime.value}
            assert labels == {0, 1}, regime
    assert seen_validation == set(range(len(v5_rows)))  # every row is validated exactly once
    sizes = [len(v) for _, v in folds]
    assert max(sizes) / min(sizes) < 1.1


def test_paired_rows_always_land_in_the_same_fold(v5_rows):
    assignments = environment_fold_assignments(v5_rows, 5)
    for fire, no_fire in pair_indices(v5_rows).values():
        assert assignments[v5_rows[fire].environment_id] == assignments[v5_rows[no_fire].environment_id]


def test_environment_fold_assignment_is_deterministic_and_validates_n_splits(v5_rows):
    assert environment_fold_assignments(v5_rows, 5) == environment_fold_assignments(v5_rows, 5)
    for bad in (1, 0, True, None, 2.5):
        with pytest.raises(ValueError):
            environment_fold_assignments(v5_rows, bad)


def test_leave_one_regime_out_holds_out_each_regime_with_both_labels(v5_rows):
    splits = leave_one_regime_out(v5_rows)
    assert [name for name, _ in splits] == sorted(r.value for r in RegimeV5)
    for regime, (train, validation) in splits:
        assert not set(train) & set(validation)
        assert len(train) + len(validation) == len(v5_rows)
        assert all(v5_rows[i].regime == regime for i in validation)
        assert all(v5_rows[i].regime != regime for i in train)
        assert {v5_rows[i].label for i in validation} == {0, 1}


def test_leave_one_no_fire_subtype_out_validates_only_the_held_out_false_alarm_type(v5_rows):
    splits = leave_one_no_fire_subtype_out(v5_rows)
    assert [name for name, _ in splits] == [
        "controlled_or_agricultural_burn", "false_or_rumour_report", "industrial_heat_source", "sensor_noise",
    ]
    for subtype, (train, validation) in splits:
        assert validation and all(v5_rows[i].latent_subtype == subtype and v5_rows[i].label == 0 for i in validation)
        assert not any(v5_rows[i].latent_subtype == subtype for i in train)
        assert any(v5_rows[i].label == 1 for i in train)  # fires stay in training


def test_pair_indices_returns_one_fire_and_one_no_fire_per_pair_and_rejects_broken_pairs(v5_rows):
    pairs = pair_indices(v5_rows)
    assert len(pairs) == 2500
    for pair_id, (fire, no_fire) in pairs.items():
        assert v5_rows[fire].label == 1 and v5_rows[no_fire].label == 0
        assert v5_rows[fire].pair_id == v5_rows[no_fire].pair_id == pair_id
    broken = tuple(r for r in v5_rows if r.pair_id is not None)[:1]  # a pair with only one member
    with pytest.raises(ValueError, match="exactly one fire and one no-fire"):
        pair_indices(broken)


def test_metadata_used_for_grouping_is_never_part_of_the_feature_vector(v5_rows):
    row = v5_rows[0]
    assert len(row.features) == 25
    for name in ("environment_id", "regime", "pair_id", "latent_subtype"):
        assert name not in FIRE_DETECTION_FEATURE_NAMES_V5
    assert isinstance(row.to_features_v5(), FireDetectionFeaturesV5)
