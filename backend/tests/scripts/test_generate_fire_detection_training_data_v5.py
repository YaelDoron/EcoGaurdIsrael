"""Task 6: the V5 generate script validates BEFORE writing and never trains anything."""
from __future__ import annotations

import pytest

from scripts import generate_fire_detection_training_data_v5 as script
from src.ml.fire_detection.fire_detection_dataset_v5 import load_training_dataset_rows_v5


def test_generate_dataset_writes_a_validated_csv(tmp_path):
    output = script.generate_dataset(num_samples=2000, seed=5, output_path=tmp_path / "v5.csv")
    rows = load_training_dataset_rows_v5(output)
    assert len(rows) == 2000 and {r.seed for r in rows} == {5}
    assert sum(r.label for r in rows) == 1000


def test_a_failing_validation_never_writes_the_file(tmp_path, monkeypatch):
    class Failing:
        is_valid = False
        violations = ("forced failure",)

    monkeypatch.setattr(script, "validate_dataset_rows_v5", lambda rows: Failing())
    target = tmp_path / "v5.csv"
    with pytest.raises(ValueError, match="failed validation"):
        script.generate_dataset(num_samples=100, seed=1, output_path=target)
    assert not target.exists()


def test_the_default_output_is_the_v5_file_and_not_the_frozen_v4_one():
    assert script.DEFAULT_OUTPUT_PATH.name == "training_v5.csv"
    assert "v4" not in script.DEFAULT_OUTPUT_PATH.name
