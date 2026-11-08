"""Third-party agent and framework integrations for Veya."""


def __getattr__(name):
    if name.startswith("Laya"):
        value = globals().get("Veya" + name[4:])
        if value is not None:
            return value
    raise AttributeError(name)

from ._errors import VeyaLowConfidenceError
from .agno import (
    AgnoRouteDecision,
    VeyaAgnoGuardrail,
    VeyaAgnoGuardrailError,
    VeyaAgnoRouter,
)
from .crewai import (
    CrewRouteDecision,
    VeyaCrewRouter,
    VeyaTaskGuard,
    VeyaTaskGuardError,
)
from .langchain import (
    VeyaDecision,
    VeyaEvaluator,
    VeyaGuardrail,
    VeyaGuardrailError,
    VeyaRouter,
    VeyaTriage,
)
from .llamaindex import (
    VeyaMultiSelector,
    VeyaQueryRouter,
    VeyaSingleSelector,
)

__all__ = [
    "VeyaRouter",
    "VeyaGuardrail",
    "VeyaGuardrailError",
    "VeyaTriage",
    "VeyaEvaluator",
    "VeyaDecision",
    "VeyaSingleSelector",
    "VeyaMultiSelector",
    "VeyaQueryRouter",
    "VeyaCrewRouter",
    "VeyaTaskGuard",
    "VeyaTaskGuardError",
    "CrewRouteDecision",
    "VeyaLowConfidenceError",
    "AgnoRouteDecision",
    "VeyaAgnoRouter",
    "VeyaAgnoGuardrail",
    "VeyaAgnoGuardrailError",
]
