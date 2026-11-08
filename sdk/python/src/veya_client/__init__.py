"""A standalone HTTP client; no local model or inference dependencies."""
from .client import VeyaClient
from .errors import (
    VeyaAPIError, VeyaConnectionError, VeyaError, VeyaResponseError,
    VeyaTimeoutError, VeyaValidationError,
)

__all__ = [
    "VeyaClient", "VeyaError", "VeyaValidationError", "VeyaAPIError",
    "VeyaConnectionError", "VeyaTimeoutError", "VeyaResponseError",
]


def __getattr__(name):
    if name.startswith("Laya"):
        value = globals().get("Veya" + name[4:])
        if value is not None:
            return value
    raise AttributeError(name)
