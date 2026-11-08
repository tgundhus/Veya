"""Failures callers can distinguish without importing the inference engine."""
from typing import Any


class VeyaError(Exception):
    """Base class for client errors."""


class VeyaValidationError(VeyaError, ValueError):
    """An invalid client setting or request, rejected before sending it."""


class VeyaAPIError(VeyaError):
    """A non-2xx HTTP response; detail preserves the server's JSON value."""

    def __init__(self, status_code: int, detail: Any = None):
        super().__init__("Veya returned HTTP %d" % status_code)
        self.status_code = status_code
        self.detail = detail


class VeyaConnectionError(VeyaError):
    """The request or response could not be transported."""


class VeyaTimeoutError(VeyaConnectionError):
    """A blocking I/O operation timed out; server inference may still finish."""


class VeyaResponseError(VeyaError):
    """A successful HTTP response did not satisfy the prediction contract."""

# Legacy API aliases.
LayaError = VeyaError
LayaValidationError = VeyaValidationError
LayaAPIError = VeyaAPIError
LayaConnectionError = VeyaConnectionError
LayaTimeoutError = VeyaTimeoutError
LayaResponseError = VeyaResponseError
