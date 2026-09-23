"""Tests for load_training_dataset_rows."""
from __future__ import annotations

from src.ml.fire_detection.fire_detection_dataset import load_training_dataset_rows
from src.ml.fire_detection.fire_detection_features import TRAINING_DATA_CSV_COLUMNS

CSV_CONTENT = (
    ",".join(TRAINING_DATA_CSV_COLUMNS)
    + "\n"
    + "1,fam_a,1,0,0,1,0,5.0,1.0,0\n"
    + "2,fam_b,2,1,0,1,1,10.0,2.0,1\n"
)


def test_load_training_dataset_rows_parses_metadata_and_features(tmp_path):
    csv_path = tmp_path / "training_v1.csv"
    csv_path.write_text(CSV_CONTENT, encoding="utf-8")

    rows = load_training_dataset_rows(csv_path)

    assert len(rows) == 2
    assert rows[0].sample_id == 1
    assert rows[0].scenario_family == "fam_a"
    assert rows[0].features == (1.0, 0.0, 0.0, 1.0, 0.0, 5.0, 1.0)
    assert rows[0].label == 0

    assert rows[1].sample_id == 2
    assert rows[1].scenario_family == "fam_b"
    assert rows[1].label == 1


def test_load_training_dataset_rows_preserves_csv_order(tmp_path):
    csv_path = tmp_path / "training_v1.csv"
    csv_path.write_text(CSV_CONTENT, encoding="utf-8")

    rows = load_training_dataset_rows(csv_path)

    assert [row.sample_id for row in rows] == [1, 2]
