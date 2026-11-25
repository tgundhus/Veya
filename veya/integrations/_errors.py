"""Exceptions shared by the framework integrations."""
from typing import Any, Dict, Optional


class VeyaLowConfidenceError(ValueError):
    """Raised when a routing confidence policy declines to dispatch without a fallback.

    An unevaluated core gate can report no confidence. Custom runners may omit its threshold.
    """

    def __init__(self, message: str, confidence: Optional[float], threshold: Optional[float],
                 raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold
        self.raw_decision = raw_decision

# Legacy API aliases.
LayaLowConfidenceError = VeyaLowConfidenceError
