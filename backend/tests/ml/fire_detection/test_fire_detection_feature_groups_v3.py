"""Tests for the V3 feature-group ablation definitions."""
from __future__ import annotations

import pytest

from src.ml.fire_detection.fire_detection_feature_groups_v3 import (
    FEATURE_GROUP_BASE,
    FEATURE_GROUP_BASE_PLUS_NEWS_SEMANTIC,
    FEATURE_GROUP_BASE_PLUS_SATELLITE_PHYSICAL,
    FEATURE_GROUP_FULL_V3,
    FEATURE_GROUPS,
    build_feature_matrix,
)
from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3


def test_full_v3_group_equals_the_official_schema():
    assert FEATURE_GROUPS[FEATURE_GROUP_FULL_V3] == FIRE_DETECTION_FEATURE_NAMES_V3


def test_base_group_excludes_physical_and_semantic_features():
    base = FEATURE_GROUPS[FEATURE_GROUP_BASE]

    assert "satellite_frp_mean" not in base
    assert "news_none_count" not in base


def test_base_plus_satellite_physical_excludes_news_semantic_detail():
    group = FEATURE_GROUPS[FEATURE_GROUP_BASE_PLUS_SATELLITE_PHYSICAL]

    assert "satellite_frp_mean" in group
    assert "news_none_count" not in group


def test_base_plus_news_semantic_excludes_satellite_physical():
    group = FEATURE_GROUPS[FEATURE_GROUP_BASE_PLUS_NEWS_SEMANTIC]

    assert "news_none_count" in group
    assert "satellite_frp_mean" not in group


def test_ablation_groups_grow_monotonically_in_size():
    sizes = {name: len(columns) for name, columns in FEATURE_GROUPS.items()}

    assert sizes[FEATURE_GROUP_BASE] < sizes[FEATURE_GROUP_BASE_PLUS_SATELLITE_PHYSICAL]
    assert sizes[FEATURE_GROUP_BASE] < sizes[FEATURE_GROUP_BASE_PLUS_NEWS_SEMANTIC]
    assert sizes[FEATURE_GROUP_BASE_PLUS_SATELLITE_PHYSICAL] < sizes[FEATURE_GROUP_FULL_V3]
    assert sizes[FEATURE_GROUP_BASE_PLUS_NEWS_SEMANTIC] < sizes[FEATURE_GROUP_FULL_V3]


def test_build_feature_matrix_selects_full_v3_columns_unchanged():
    full_row = tuple(float(index) for index in range(len(FIRE_DETECTION_FEATURE_NAMES_V3)))

    matrix = build_feature_matrix([full_row], FEATURE_GROUP_FULL_V3)

    assert matrix == [list(full_row)]


def test_build_feature_matrix_derives_news_total_count_for_base_group():
    index = {name: position for position, name in enumerate(FIRE_DETECTION_FEATURE_NAMES_V3)}
    full_row = [0.0] * len(FIRE_DETECTION_FEATURE_NAMES_V3)
    full_row[index["news_none_count"]] = 1.0
    full_row[index["news_weak_count"]] = 2.0
    full_row[index["news_moderate_count"]] = 0.0
    full_row[index["news_strong_count"]] = 1.0
    full_row[index["news_unknown_count"]] = 1.0

    matrix = build_feature_matrix([tuple(full_row)], FEATURE_GROUP_BASE)

    base_names = FEATURE_GROUPS[FEATURE_GROUP_BASE]
    news_total_index = base_names.index("news_total_count")
    assert matrix[0][news_total_index] == pytest.approx(5.0)  # 1+2+0+1+1


def test_unknown_feature_group_raises():
    with pytest.raises(ValueError):
        build_feature_matrix([(0.0,) * len(FIRE_DETECTION_FEATURE_NAMES_V3)], "not_a_real_group")
