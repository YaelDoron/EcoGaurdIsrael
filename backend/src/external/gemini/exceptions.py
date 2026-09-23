"""Exception hierarchy for the Gemini (Google Generative Language API) HTTP client."""


class GeminiClientError(Exception):
    """Base exception for all Gemini client errors."""


class GeminiConfigurationError(GeminiClientError):
    """Raised when the Gemini client is not configured correctly (e.g. missing API key)."""


class GeminiAuthenticationError(GeminiClientError):
    """Raised when Gemini rejects the request due to invalid/missing credentials (401/403)."""


class GeminiServiceUnavailableError(GeminiClientError):
    """Raised when Gemini is unreachable, rate-limited, or returns a server-side error
    (429/5xx/timeout/connection failure)."""


class GeminiInvalidResponseError(GeminiClientError):
    """Raised when Gemini returns a response that cannot be parsed as JSON."""


class GeminiEmptyResponseError(GeminiClientError):
    """Raised when Gemini returns a successfully-parsed response with no usable generated text."""
