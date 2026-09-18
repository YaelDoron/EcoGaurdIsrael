"""Tests for GlobalResponsePlanChromosome (Stage 4, Tasks 2/5/6)."""
from __future__ import annotations

import pytest

from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome


def test_valid_chromosome():
    chromosome = GlobalResponsePlanChromosome(("R1", "R2", "R3"), ("10#0", None, "20#0"))
    assert chromosome.genes[1] is None


def test_resource_uniqueness_enforced_in_resource_ids():
    with pytest.raises(ValueError):
        GlobalResponsePlanChromosome(("R1", "R1"), (None, None))


def test_slot_uniqueness_enforced_across_genes():
    with pytest.raises(ValueError):
        GlobalResponsePlanChromosome(("R1", "R2"), ("10#0", "10#0"))


def test_same_response_target_id_can_be_referenced_by_different_slot_ids():
    """Task 6: same response_target_id is NOT forbidden at this level -
    only the same slot_id twice is. Stage 5 needs this to remain legal."""
    chromosome = GlobalResponsePlanChromosome(("R1", "R2"), ("10#0", "10#1"))
    assert chromosome.genes == ("10#0", "10#1")


def test_genes_length_must_match_resource_ids_length():
    with pytest.raises(ValueError):
        GlobalResponsePlanChromosome(("R1", "R2"), (None,))


def test_all_idle_chromosome_is_valid():
    chromosome = GlobalResponsePlanChromosome(("R1", "R2"), (None, None))
    assert chromosome.genes == (None, None)


def test_rejects_empty_resource_id():
    with pytest.raises(ValueError):
        GlobalResponsePlanChromosome(("",), (None,))


def test_rejects_non_iterable_resource_ids():
    with pytest.raises(ValueError):
        GlobalResponsePlanChromosome(123, (None,))


def test_two_chromosomes_with_same_content_are_equal():
    a = GlobalResponsePlanChromosome(("R1", "R2"), ("10#0", None))
    b = GlobalResponsePlanChromosome(("R1", "R2"), ("10#0", None))
    assert a == b
    assert hash(a) == hash(b)
