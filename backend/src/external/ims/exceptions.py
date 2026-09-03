"""Exception hierarchy for the IMS (Israeli Meteorological Service) HTTP client."""


class IMSClientError(Exception):
    """Base exception for all IMS client errors."""


class IMSConfigurationError(IMSClientError):
    """Raised when the IMS client is not configured correctly (e.g. missing API token)."""


class IMSAuthenticationError(IMSClientError):
    """Raised when IMS rejects the request due to invalid/missing credentials (401/403)."""


class IMSStationNotFoundError(IMSClientError):
    """Raised when the requested IMS resource (e.g. station) does not exist (404)."""


class IMSServiceUnavailableError(IMSClientError):
    """Raised when IMS is unreachable or returns a server-side error (5xx/timeout/connection failure)."""


class IMSInvalidResponseError(IMSClientError):
    """Raised when IMS returns a response that cannot be parsed as JSON."""
