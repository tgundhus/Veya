"""Host-side inference budgets for the optional local code interface.

These checks only inspect inputs and resident metadata. They never load a model;
the code runner also verifies runtime eligibility before invoking an operation.
"""
import json
import math

from . import tools
from .device import router_agent
from .tools import ToolError


_DEFAULTS = {
    "max_states": 64, "max_questions": 64, "max_choice_options": 100,
    "max_tournament_options": 255, "max_score_levels": 32,
    "max_total_options": 512, "max_token_budget": 8192,
    "max_batch_tokens": 131072, "max_state_chars": 50000,
    "max_scan_windows": 64,
}


def _fail(message):
    raise ToolError("code_resource_limit", message)


def _limit(limits, name):
    return getattr(limits, name, _DEFAULTS[name])


def _integer(value, name, maximum, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        _fail("%s must be an integer between %d and %d" % (name, minimum, maximum))
    return value


def _state(state, limits):
    checked = tools.validate_state(state)
    # JSON input already has a depth/byte bound in the runner. Include keys and
    # numeric fields here, so every accepted state has a bounded serialized size.
    try:
        text = json.dumps(checked, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise ToolError("invalid_state", "state must contain finite JSON values") from None
    if len(text) > _limit(limits, "max_state_chars"):
        _fail("state exceeds max_state_chars")
    return checked


def _questions(questions, method, limits):
    if isinstance(questions, dict) and len(questions) > _limit(limits, "max_questions"):
        _fail("questions exceeds max_questions")
    checked = tools.validate_questions(questions)
    total = 0
    for question in checked.values():
        kind = question["type"]
        count = len(question["criteria"]) if kind in ("choice", "score") else 2
        cap = ("max_tournament_options" if method in ("shortlist", "tournament") else "max_choice_options")
        if kind == "score":
            cap = "max_score_levels"
        if count > _limit(limits, cap):
            _fail("question option count exceeds %s" % cap)
        total += count
    if total > _limit(limits, "max_total_options"):
        _fail("question options exceed max_total_options")
    return checked, total


def _resident(arguments, state, questions, router):
    name = tools._routed_control(arguments.get("model"), tools.validate_model, router)
    if name != "auto":
        return router_agent(router, name)
    # This private selector is pure; public route() also dispatches user hooks.
    # Unknown/custom routers remain cold for budgeting rather than being invoked.
    from ..router import Router
    if type(router) is not Router or callable(router.lang_guess):
        return None
    controls = {}
    for key, validator in (("task", tools.validate_task), ("lang", tools.validate_lang),
                           ("lang_guess", tools.validate_lang_guess)):
        if key in arguments:
            value = (tools._routed_control(arguments[key], validator, router)
                     if key == "task" else validator(arguments[key]))
            if value is not None:
                controls[key] = value
    decision = router._route(state, questions, **controls)
    return router_agent(router, decision["model"])


def _width(arguments, agent, limits):
    maximum = _limit(limits, "max_token_budget")
    cfg = getattr(agent, "cfg", None) if agent is not None else None
    cfg = cfg if isinstance(cfg, dict) else {}
    widths = {}
    for key in ("max_len", "head_max_len"):
        explicit = tools.validate_budget(arguments.get(key), key)
        default = (512 if key == "max_len" else 192) if agent is not None else maximum
        width = explicit if explicit is not None else cfg.get(key, default)
        widths[key] = _integer(width, key, maximum)
    return widths["max_len"], widths["head_max_len"]


def _schema_questions(schema, limits):
    # Check raw cardinalities before the schema planner constructs score ranges
    # and enum label maps, including definitions reached through local references.
    nodes = 0

    def inspect(value, depth):
        nonlocal nodes
        nodes += 1
        if depth > 32 or nodes > 50000:
            _fail("schema nesting or item count exceeds the code-mode limit")
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict) and len(properties) > _limit(limits, "max_questions"):
                _fail("schema properties exceeds max_questions")
            enum = value.get("enum")
            if isinstance(enum, list) and len(enum) > _limit(limits, "max_choice_options"):
                _fail("schema enum exceeds max_choice_options")
            lower, upper = value.get("minimum"), value.get("maximum")
            if (isinstance(lower, int) and not isinstance(lower, bool)
                    and isinstance(upper, int) and not isinstance(upper, bool)
                    and upper - lower + 1 > _limit(limits, "max_score_levels")):
                _fail("schema integer range exceeds max_score_levels")
            for child in value.values():
                inspect(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                inspect(child, depth + 1)

    inspect(schema, 0)
    from ..structured import SchemaError, questions_from_json_schema
    try:
        return questions_from_json_schema(schema)
    except SchemaError as exc:
        raise ToolError("invalid_schema", str(exc)) from None


def _scan_rows(arguments, state, questions, agent, width, head, limits):
    if agent is None or not hasattr(agent, "tok") or not callable(getattr(agent, "_to_internal", None)):
        _fail("predict_long requires a resident checkpoint tokenizer for bounded scan preflight")
    from ..inference_common import encode_text, serialize_state, window_budget
    window = tools.validate_budget(arguments.get("window"), "window")
    stride = tools.validate_budget(arguments.get("stride"), "stride")
    for name, value in (("window", window), ("stride", stride)):
        if value is not None:
            _integer(value, name, _limit(limits, "max_token_budget"))
    try:
        internal = [agent._to_internal(question) for question in questions.values()]
        size, step, room = window_budget(agent.tok, internal, width, head, window=window, stride=stride)
        text = serialize_state(state).replace(agent.tok.mask_token, " ")
        count = len(encode_text(agent.tok, text, add_special_tokens=False)["input_ids"])
    except (ValueError, TypeError, KeyError):
        raise ToolError("invalid_scan", "scan window or stride does not fit the resident checkpoint") from None
    rows = 1 if count <= (size if window else room) else 1 + max(0, math.ceil((count - size) / step))
    if rows > _limit(limits, "max_scan_windows"):
        _fail("predict_long would exceed max_scan_windows")
    return rows * len(questions)


def _forward_cost(method, arguments, state, questions, router, limits, preflight):
    agent = _resident(arguments, state, questions, router)
    width, head = _width(arguments, agent, limits)
    rows = len(questions)
    extra = 0
    if method == "shortlist":
        k = arguments.get("k")
        k = _integer(20 if k is None else k, "k", _limit(limits, "max_choice_options"))
        for question in questions.values():
            if question["type"] == "choice" and len(question["criteria"]) > k:
                # The existing embedding helper tokenizes one query plus every
                # candidate at max_length=512, once for each narrowed question.
                if _limit(limits, "max_token_budget") < 512:
                    _fail("shortlist embedding width exceeds max_token_budget")
                extra += (len(question["criteria"]) + 1) * 512
    elif method == "tournament":
        size = _integer(arguments.get("group_size", 16), "group_size",
                        _limit(limits, "max_choice_options"), minimum=2)
        counts = [len(question["criteria"]) for question in questions.values() if question["type"] == "choice"]
        while any(count > size for count in counts):
            groups = [(count + size - 1) // size if count > size else 0 for count in counts]
            if sum(groups) > _limit(limits, "max_questions"):
                _fail("tournament round would exceed max_questions")
            rows += sum(groups)
            counts = [groups[i] if groups[i] else count for i, count in enumerate(counts)]
    elif method == "predict_long":
        if agent is None and preflight:
            return None
        rows = _scan_rows(arguments, state, questions, agent, width, head, limits)
    if agent is None and preflight:
        return None
    return rows * width + extra


def check_operation_budget(method, arguments, router, limits, *, preflight=False, replayed=()):
    """Refuse oversized operation work before a model load or forward pass.

    Returns the checked per-state requests, including questions derived from a
    schema or preset. Inputs are not changed. Runtime eligibility and loaded configuration checks
    are enforced by the runner as well; this function does not authorize custom
    hooks or unknown model implementations to rewrite these budgets.

    ``preflight=True`` defers unknown-model forward/scan cost until the runner
    has checked authoritative replay or loaded the requested checkpoint. The
    runner must repeat the full check before invoking an uncached operation.

    ``replayed`` contains host-verified exact-hit indices, never guest input.
    Only plain prediction, batch prediction, schema decisions and presets may
    skip forward cost; all input/cardinality/explicit-budget checks still apply.
    """
    if (not isinstance(replayed, (tuple, list, set, frozenset))
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in replayed)):
        raise ToolError("code_invalid_call", "replayed must contain host-verified nonnegative request indices")
    replayed = frozenset(replayed)
    if replayed and method not in ("predict", "predict_batch", "decide", "preset"):
        raise ToolError("code_invalid_call", "this operation cannot skip work using replayed indices")
    if method == "status":
        return []
    for name in ("max_len", "head_max_len", "window", "stride"):
        if name in arguments and arguments[name] is not None:
            _integer(tools.validate_budget(arguments[name], name), name, _limit(limits, "max_token_budget"))
    if method in ("predict_batch", "route_batch"):
        raw = arguments.get("requests")
        if isinstance(raw, list) and len(raw) > _limit(limits, "max_states"):
            _fail("requests exceeds max_states")
        if method == "predict_batch":
            size = tools._validate_batch_size(arguments.get("batch_size"))
            if size is not None:
                _integer(size, "batch_size", _limit(limits, "max_states"))
        # Cardinality is bounded before validating every nested question.
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    _questions(item.get("questions"), method, limits)
        items = tools._validate_registered_batch_requests(raw, router)
    else:
        state = _state(arguments.get("state"), limits)
        if method == "decide":
            questions = _schema_questions(arguments.get("schema"), limits)
        elif method == "preset":
            from .. import presets
            preset = tools.validate_preset(arguments.get("preset"))
            questions = getattr(presets, tools.PRESETS[preset])()
            field = presets.state_field(questions)
            if field is not None and field not in state and len(state) == 1:
                value, = state.values()
                if isinstance(value, str):
                    state = {field: value}
        else:
            questions = arguments.get("questions")
        items = [{**arguments, "state": state, "questions": questions}]
    if any(index >= len(items) for index in replayed):
        raise ToolError("code_invalid_call", "replayed contains an unknown request index")
    cost, options = 0, 0
    for index, item in enumerate(items):
        state = _state(item["state"], limits)
        questions, count = _questions(item["questions"], method, limits)
        item["state"], item["questions"] = state, questions
        options += count
        if options > _limit(limits, "max_total_options"):
            _fail("operation options exceed max_total_options")
        for key in ("max_len", "head_max_len"):
            value = tools.validate_budget(item.get(key), key)
            if value is not None:
                _integer(value, key, _limit(limits, "max_token_budget"))
        if method not in ("route", "route_batch") and index not in replayed:
            forward = _forward_cost(method, item, state, questions, router, limits, preflight)
            if forward is not None:
                cost += forward
            if cost > _limit(limits, "max_batch_tokens"):
                _fail("operation forward work exceeds max_batch_tokens")
    return items
