"""Exception hierarchy for the NASA FIRMS HTTP client."""


class FIRMSClientError(Exception):
    """Base exception for all FIRMS client errors."""


class FIRMSConfigurationError(FIRMSClientError):
    """Raised when the FIRMS client is not configured correctly."""


class FIRMSAuthenticationError(FIRMSClientError):
    """Raised when FIRMS rejects the request due to invalid/missing credentials."""


class FIRMSServiceUnavailableError(FIRMSClientError):
    """Raised when FIRMS is unreachable or returns a server-side error."""


class FIRMSInvalidResponseError(FIRMSClientError):
    """Raised when FIRMS returns a response that cannot be parsed as valid CSV."""
