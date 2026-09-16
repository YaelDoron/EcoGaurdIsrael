"""Shared validation helpers for pure response-optimization models."""
from __future__ import annotations

import math
from numbers import Real


def validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def validate_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


def validate_resource_id(field_name: str, value: object) -> None:
    if (
        not isinstance(value, (int, str))
        or isinstance(value, bool)
        or (isinstance(value, int) and value <= 0)
        or (isinstance(value, str) and not value.strip())
    ):
        raise ValueError(f"{field_name} must be a positive int or non-empty string, got {value!r}")


def normalize_resource_id(value: int | str) -> int | str:
    validate_resource_id("resource_id", value)
    if isinstance(value, str):
        return value.strip()
    return value


def resource_sort_key(value: int | str) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def validate_finite_non_negative_number(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number, got {value!r}")


def validate_optional_finite_non_negative_number(field_name: str, value: object) -> None:
    if value is None:
        return
    validate_finite_non_negative_number(field_name, value)


def validate_non_empty_string(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
