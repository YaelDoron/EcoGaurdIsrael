"""Gemini (Google Generative Language API) external integration."""

from src.external.gemini.exceptions import (
    GeminiAuthenticationError,
    GeminiClientError,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiInvalidResponseError,
    GeminiServiceUnavailableError,
)
from src.external.gemini.gemini_client import GeminiClient

__all__ = [
    "GeminiClient",
    "GeminiClientError",
    "GeminiConfigurationError",
    "GeminiAuthenticationError",
    "GeminiServiceUnavailableError",
    "GeminiInvalidResponseError",
    "GeminiEmptyResponseError",
]
