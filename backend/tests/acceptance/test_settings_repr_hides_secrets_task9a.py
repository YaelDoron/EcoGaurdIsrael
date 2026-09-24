"""Task 9A secret hygiene: repr(Settings) must never reveal a secret, even when one is set."""
from __future__ import annotations

from dataclasses import fields

from src.config.settings import Settings

SECRET_FIELDS = (
    "IMS_API_TOKEN", "FIRMS_MAP_KEY", "COPERNICUS_CLIENT_ID", "COPERNICUS_CLIENT_SECRET", "DATABASE_URL", "GEMINI_API_KEY",
)


def test_secret_fields_are_excluded_from_repr_and_do_not_leak_a_set_value():
    marker = "not-a-real-secret-marker"
    settings = Settings(**{name: marker for name in SECRET_FIELDS})

    text = repr(settings) + str(settings)

    assert marker not in text
    excluded = {f.name for f in fields(Settings) if not f.repr}
    assert set(SECRET_FIELDS) <= excluded
