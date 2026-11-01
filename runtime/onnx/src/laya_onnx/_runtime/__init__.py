"""Canonical inference modules, populated when the distribution is built."""
from .. import __version__

DEFAULT_INFERENCE_BACKEND = "onnx"

from .router import Router
from .presets import email_questions, guard_questions, moderation_questions, router_questions, triage_questions

__all__ = [
    "__version__", "Router", "email_questions", "guard_questions", "moderation_questions",
    "router_questions", "triage_questions",
]
