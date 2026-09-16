"""Tests for response-plan chromosome model."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.models import ResponsePlanChromosome


def test_valid_all_none_chromosome():
    assert ResponsePlanChromosome((None, None)).genes == (None, None)


def test_valid_assignments_normalize_string_resource_ids():
    chromosome = ResponsePlanChromosome((" R1 ", None, 2))

    assert chromosome.genes == ("R1", None, 2)


def test_chromosome_is_frozen():
    chromosome = ResponsePlanChromosome((None,))

    with pytest.raises(FrozenInstanceError):
        chromosome.genes = ("R1",)


def test_duplicate_resource_rejected():
    with pytest.raises(ValueError):
        ResponsePlanChromosome(("R1", None, "R1"))


def test_chromosome_deterministic_equality():
    assert ResponsePlanChromosome(("R1", None)) == ResponsePlanChromosome(("R1", None))
    assert ResponsePlanChromosome(("R1", None)) != ResponsePlanChromosome((None, "R1"))


def test_mixed_int_and_string_resource_ids_follow_existing_rules():
    chromosome = ResponsePlanChromosome((1, "1"))

    assert chromosome.genes == (1, "1")


@pytest.mark.parametrize("gene", ["", "   ", 0, True])
def test_invalid_resource_id_gene_rejected(gene):
    with pytest.raises(ValueError):
        ResponsePlanChromosome((gene,))
