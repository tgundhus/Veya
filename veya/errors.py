"""Errors shared by the pure-Python runtime and its optional transports."""


class DecisionCacheError(RuntimeError):
    """A required decision cache could not authorize a prediction result."""
