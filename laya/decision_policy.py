"""Opt-in, local presentation balancing and content-free contextual correction.

The v1 policy averages every cyclic option rotation for the input and the fixed
content-free probes, then normalizes their probability ratio. It is a different
inference policy, not a fitted calibration. No targets or global prior cache are used.
"""
import copy
from contextvars import ContextVar


DECISION_POLICIES = ("canonical", "context-balanced-v1")
CONTENT_FREE_PROBES = ("", "N/A", "[MASK]")
MAX_BALANCED_OPTIONS = 16
_MAX_VARIANT_ROWS = 64
_FORWARD_TOKEN_BUDGET = ContextVar("veya_policy_forward_token_budget", default=None)


def resolve_decision_policy(value=None, cfg=None):
    """Resolve an explicit policy or its checkpoint default without importing ML runtimes."""
    if value is None:
        value = (cfg or {}).get("decision_policy", "canonical")
    if not isinstance(value, str) or value not in DECISION_POLICIES:
        raise ValueError("decision_policy must be one of %s, got %r" % (DECISION_POLICIES, value))
    return value


def _probabilities(answer, count):
    import numpy as np

    values = np.asarray(answer.get("_decision_probabilities"), dtype=np.float64)
    if (values.shape != (count,) or not np.isfinite(values).all()
            or (values < 0).any() or not np.isclose(values.sum(), 1.0, rtol=1e-7, atol=1e-12)):
        raise ValueError("decision policy received an invalid full-precision probability row")
    action = answer.get("action", {}).get("act_probability")
    if not isinstance(action, (float, int)) or not np.isfinite(action) or not 0 <= action <= 1:
        raise ValueError("decision policy received an invalid action probability")
    return values


def _correct(real, prior):
    import numpy as np

    # A fixed float64 machine floor handles underflow; this is not a fitted smoothing knob.
    tiny = np.finfo(np.float64).tiny
    ratios = np.log(np.maximum(real, tiny)) - np.log(np.maximum(prior, tiny))
    corrected = np.exp(ratios - ratios.max())
    corrected /= corrected.sum()
    if not np.isfinite(corrected).all():
        raise ValueError("decision policy produced invalid corrected probabilities")
    return corrected


def _answer(qdef, probabilities, action):
    import numpy as np
    from .inference_common import confidence_from_probs, render_criterion

    count = len(probabilities)
    kind = qdef["type"]
    answer = {"type": kind, "answer_confidence": round(float(probabilities.max()), 4),
              "confidence": round(float(probabilities.max()) if kind == "noul"
                                  else confidence_from_probs(probabilities, count), 4),
              "action": {"act_probability": round(float(action), 4)},
              "decision_policy": "context-balanced-v1", "confidence_calibrated": False}
    if kind == "choice":
        criteria = qdef["criteria"]
        labels = list(criteria)
        answer["choice"] = labels[int(probabilities.argmax())]
        answer["probabilities"] = {key: round(float(value), 4)
                                   for key, value in zip(labels, probabilities)}
    elif kind == "score":
        answer["score"] = round(float(np.dot(np.arange(count), probabilities)), 4)
        answer["legend"] = {str(i): render_criterion(value) for i, value in enumerate(qdef["criteria"])}
        answer["probabilities"] = {str(i): round(float(value), 4) for i, value in enumerate(probabilities)}
    else:
        answer["noul"] = round(float(probabilities[1]), 4)
    return answer


def _row_cap(agent, max_len):
    budget = _FORWARD_TOKEN_BUDGET.get()
    if budget is None:
        return _MAX_VARIANT_ROWS
    width = agent.cfg.get("max_len", 512) if max_len is None else max_len
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("context-balanced-v1 requires a positive integer max_len")
    cap = min(_MAX_VARIANT_ROWS, budget // width)
    if cap < 1:
        raise ValueError("context-balanced-v1 max_len=%d exceeds the forward token budget=%d" % (width, budget))
    return cap


def infer_balanced(agent, states, questions, *, batch_size=None, lang=None,
                   max_len=None, head_max_len=None, sort_by_length=False):
    """Run v1 through an unhooked canonical backend, with at most 64 variant rows per forward.

    Probe work is shared within this call only and charged to the first result.
    State metadata describes the original state; token totals include all actual work.
    Explicit option orders supply the starting order for the cyclic rotations. This
    balances positions and does not claim invariance to every possible permutation.
    Length sorting retains the canonical eight-batch lookahead with the capped batch.
    """
    import numpy as np
    from .inference import _check_question, _option_count

    if not states:
        return []
    plans = []
    for qid, qdef in questions.items():
        _check_question(qid, qdef)
        count = _option_count(qdef)
        if count > MAX_BALANCED_OPTIONS:
            raise ValueError("context-balanced-v1 supports at most %d options per question; %r has %d"
                             % (MAX_BALANCED_OPTIONS, qid, count))
        initial = list(qdef.get("option_order", range(count)))
        rotations = {index: dict(qdef, option_order=initial[index:] + initial[:index])
                     for index in range(count)}
        plans.append((qid, qdef, count, rotations))
    if not plans:
        return agent._infer_batch(states, questions, batch_size=batch_size, lang=lang,
                                  max_len=max_len, head_max_len=head_max_len,
                                  sort_by_length=sort_by_length)
    row_cap = _row_cap(agent, max_len)
    results = [{"model": "", "answers": {}, "usage": {
        "input_tokens": 0, "output_tokens": 0, "state_tokens": 0,
        "state_tokens_dropped": 0, "truncated": False, "truncated_questions": [],
        "decision_policy": "context-balanced-v1", "policy_rotation_tokens": 0,
        "policy_probe_tokens": 0, "policy_variant_rows": 0, "policy_probe_rows": 0,
    }} for _ in states]
    for qid, qdef, count, rotations in plans:
        group_size = min(count, row_cap)
        groups = [{index: rotations[index] for index in range(start, min(start + group_size, count))}
                  for start in range(0, count, group_size)]
        cap = row_cap // group_size
        if batch_size is not None and batch_size > 0:
            cap = min(cap, batch_size)
        lookahead = cap * 8 if sort_by_length and cap > 1 else cap
        options = {"batch_size": cap, "lang": lang, "max_len": max_len,
                   "head_max_len": head_max_len, "sort_by_length": sort_by_length, "full_precision": True}

        def observe(part):
            # Group only changes scheduling. Keep probe/state and rotation accumulation order
            # fixed even when the HTTP token budget cannot fit every rotation in one forward.
            return zip(*(agent._infer_batch(part, group, **options) for group in groups))

        probe_rows = []
        probe_tokens = 0
        for rows in observe(list(CONTENT_FREE_PROBES)):
            for row in rows:
                probe_rows.extend(_probabilities(answer, count) for answer in row["answers"].values())
                probe_tokens += row["usage"]["input_tokens"]
        prior = np.mean(probe_rows, axis=0)
        for start in range(0, len(states), lookahead):
            for offset, rows in enumerate(observe(states[start:start + lookahead])):
                answers = [answer for row in rows for answer in row["answers"].values()]
                real = np.mean([_probabilities(answer, count) for answer in answers], axis=0)
                action = np.mean([answer["action"]["act_probability"] for answer in answers])
                result = results[start + offset]
                result["model"] = rows[0]["model"]
                result["answers"][qid] = _answer(qdef, _correct(real, prior), action)
                usage = result["usage"]
                originals = [row["usage"] for row in rows]
                tokens = sum(original["input_tokens"] for original in originals)
                usage["input_tokens"] += tokens
                usage["policy_rotation_tokens"] += tokens
                usage["policy_variant_rows"] += count
                usage["state_tokens"] = originals[0].get("state_tokens", 0)
                usage["state_tokens_dropped"] = max(usage["state_tokens_dropped"],
                    max(original.get("state_tokens_dropped", 0) for original in originals))
                if any(original.get("truncated") for original in originals):
                    usage["truncated"] = True
                    usage["truncated_questions"].append(qid)
                collapsed = [stats for original in originals for stats in (original.get("options") or {}).values()]
                if collapsed:
                    usage.setdefault("options", {})[qid] = copy.deepcopy(min(collapsed, key=lambda value: value["distinct"]))
        if results:
            usage = results[0]["usage"]
            usage["input_tokens"] += probe_tokens
            usage["policy_probe_tokens"] += probe_tokens
            usage["policy_probe_rows"] += len(CONTENT_FREE_PROBES) * count
    return results
