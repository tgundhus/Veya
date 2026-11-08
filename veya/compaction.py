"""Experimental, opt-in pruning of explicitly replayable tool history.

The caller owns the transcript and supplies a local Agent, ONNXAgent or Router.
No model is loaded by this module, and original messages are never modified.
"""
import copy
import json
import math
import time

from .router import Router
from .confidence import GATE_PASSED


_QUESTIONS = {
    "keep_call": {"type": "noul", "instructions":
                  "Is knowing this tool call happened still needed for the current goal?"},
    "keep_result": {"type": "noul", "instructions":
                    "Is this tool result still needed verbatim for the current goal?"},
}
_MESSAGE_KEYS = {"role", "text", "toolUses", "toolResults"}


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return value


def _probability(name, value, upper=1.0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= upper):
        raise ValueError("%s must be a finite number between 0 and %s" % (name, upper))
    return value


def _serialized(messages):
    try:
        return json.dumps(messages, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("messages must contain finite JSON values") from exc


def _collect(messages, recent):
    calls, results = {}, {}
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str) or message["role"] not in
                {"system", "developer", "user", "assistant", "tool"}
                or not isinstance(message.get("text", ""), str)):
            raise ValueError("messages must contain role/text dictionaries")
        for key, target in (("toolUses", calls), ("toolResults", results)):
            blocks = message.get(key, [])
            if not isinstance(blocks, list):
                raise ValueError("%s must be a list" % key)
            for block in blocks:
                if not isinstance(block, dict):
                    raise ValueError("tool blocks must be dictionaries")
                identity = block.get("tool_use_id")
                if not isinstance(identity, str) or not identity or identity in target:
                    raise ValueError("tool ids must be nonempty and unique within calls/results")
                if key == "toolUses":
                    if (not isinstance(block.get("tool"), str) or not block["tool"]
                            or not isinstance(block.get("input"), dict)):
                        raise ValueError("tool calls require a tool name and input dictionary")
                elif not isinstance(block.get("text"), str):
                    raise ValueError("tool results require text")
                if "text" in block and not isinstance(block["text"], str):
                    raise ValueError("tool text must be a string")
                for flag in ("pinned", "replayable", "isError"):
                    if flag in block and not isinstance(block[flag], bool):
                        raise ValueError("%s must be a boolean" % flag)
                target[identity] = (index, block)
        if "pinned" in message and not isinstance(message["pinned"], bool):
            raise ValueError("pinned must be a boolean")
    for identity, (index, _) in results.items():
        if identity not in calls or index <= calls[identity][0]:
            raise ValueError("each result must follow its matching tool call")

    def protected(index):
        message = messages[index]
        return (index == 0 or index >= len(messages) - recent or message.get("pinned", False)
                or message["role"] in {"system", "developer"} or "content" in message)

    pairs = []
    for identity, (index, call) in calls.items():
        found = results.get(identity)
        result_index, result = found if found else (None, {})
        if not found:
            reason = "incomplete"
        elif (protected(index) or protected(result_index) or call.get("pinned") or result.get("pinned")
              or "content" in call or "content" in result):
            reason = "pinned"
        elif call.get("isError") or result.get("isError"):
            reason = "error"
        elif call.get("replayable") is not True:
            reason = "not_replayable"
        else:
            reason = "candidate"
        pairs.append({"tool_use_id": identity, "call_index": index, "result_index": result_index,
                      "action": "keep", "reason": reason, "keep_call": None, "keep_result": None})
    return pairs


def _excerpt(text):
    if len(text) <= 600:
        return text
    return text[:300] + "\n[output middle omitted]\n" + text[-300:]


class ContextCompactor:
    """Prune explicitly replayable tool pairs using a caller-supplied local predictor.

    Only calls marked ``replayable=True`` are candidates. This is the caller's
    assertion that the output can be recovered from the same immutable snapshot
    without repeating side effects. Errors, pinned messages, the first message
    and the newest six messages are protected. Uncertain scores retain content.

    This experimental policy does not establish that information is safe to lose.
    Keep the original transcript and evaluate retention on representative tasks.
    """

    def __init__(self, predictor, *, drop_threshold=0.1, preserve_recent_messages=6,
                 batch_size=8, truncate_head_chars=300, min_reduction_ratio=0.1,
                 model=None, max_len=None):
        if not callable(getattr(predictor, "predict_batch", None)):
            raise TypeError("predictor must supply predict_batch")
        self.predictor = predictor
        self.drop_threshold = _probability("drop_threshold", drop_threshold, 0.5)
        self.preserve_recent_messages = _integer("preserve_recent_messages", preserve_recent_messages, 0)
        self.batch_size = _integer("batch_size", batch_size, 1)
        self.truncate_head_chars = _integer("truncate_head_chars", truncate_head_chars, 0)
        self.min_reduction_ratio = _probability("min_reduction_ratio", min_reduction_ratio)
        if model is not None and (not isinstance(model, str) or not model):
            raise ValueError("model must be a nonempty string or None")
        if model is not None and not isinstance(predictor, Router):
            raise ValueError("model selection requires a Router; Agents already select a checkpoint")
        self.model = model
        self.max_len = None if max_len is None else _integer("max_len", max_len, 1)

    def compact(self, messages, *, goal, constraints="", token_counter=None):
        """Return ``messages``, per-call ``decisions`` and ``stats`` without mutating input.

        Messages use ``role``, ``text``, ``toolUses`` and ``toolResults``; tool
        blocks pair by ``tool_use_id``. Extra JSON metadata is preserved. Native
        provider transcripts need an adapter to this schema before compaction.

        ``goal`` and ``constraints`` are supplied in full to every candidate.
        The state includes instruction messages, recent text and output excerpts,
        not the full history. Truncated model inputs are never used for pruning.
        Scoring failures roll back the entire run. Invalid inputs raise ValueError.

        ``token_counter(messages)`` optionally counts the downstream transcript
        tokens. When supplied, the minimum reduction is measured in those tokens;
        otherwise it is measured in serialized JSON characters, not token estimates.
        """
        started = time.perf_counter()
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        if not isinstance(goal, str) or not goal.strip() or not isinstance(constraints, str):
            raise ValueError("goal must be nonempty text and constraints must be text")
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable")
        original = copy.deepcopy(messages)
        decisions = _collect(original, self.preserve_recent_messages)
        chars_before = len(_serialized(original))
        tokens_before = None if token_counter is None else _integer(
            "token count", token_counter(copy.deepcopy(original)), 0)
        candidates = [decision for decision in decisions if decision["reason"] == "candidate"]
        instructions = [m.get("text", "") for i, m in enumerate(original)
                        if i == 0 or m["role"] in {"system", "developer"} or m.get("pinned")]
        # Retain at least the latest turn in scoring even when the protection count is zero.
        recent = [{"role": m["role"], "text": m.get("text", "")}
                  for m in original[-max(1, self.preserve_recent_messages):] if m.get("text")]
        batches = 0
        fallback = None
        try:
            for start in range(0, len(candidates), self.batch_size):
                batch = candidates[start:start + self.batch_size]
                states = []
                for decision in batch:
                    identity = decision["tool_use_id"]
                    call = next(c for c in original[decision["call_index"]]["toolUses"]
                                if c["tool_use_id"] == identity)
                    result = next(r for r in original[decision["result_index"]]["toolResults"]
                                  if r["tool_use_id"] == identity)
                    states.append({"goal": goal, "constraints": constraints, "instructions": instructions,
                                   "recent": recent, "tool": call["tool"], "input": call["input"],
                                   "output_excerpt": _excerpt(result["text"])})
                kwargs = {} if self.max_len is None else {"max_len": self.max_len}
                states = copy.deepcopy(states)
                batches += 1
                if isinstance(self.predictor, Router):
                    requests = [{"state": state, "questions": copy.deepcopy(_QUESTIONS), **kwargs}
                                for state in states]
                    if self.model is not None:
                        for request in requests:
                            request["model"] = self.model
                    results = self.predictor.predict_batch(requests, batch_size=self.batch_size)
                else:
                    results = self.predictor.predict_batch(states, copy.deepcopy(_QUESTIONS),
                                                           batch_size=self.batch_size, **kwargs)
                if not isinstance(results, list) or len(results) != len(batch):
                    raise ValueError("wrong number of prediction results")
                for decision, result in zip(batch, results):
                    answers = [result["answers"][key] for key in _QUESTIONS]
                    scores = [_probability(key, answer["noul"])
                              for key, answer in zip(_QUESTIONS, answers)]
                    decision["keep_call"], decision["keep_result"] = scores
                    usage = result.get("usage", {})
                    if (usage.get("truncated") is not False or type(usage.get("state_tokens_dropped")) is not int
                            or usage["state_tokens_dropped"] != 0
                            or usage.get("truncated_questions")):
                        decision["reason"] = "unverified_context"
                    elif any(answer.get("low_confidence") or answer.get("abstention", GATE_PASSED) != GATE_PASSED
                             for answer in answers):
                        decision["reason"] = "unverified_confidence"
                    elif scores[1] > self.drop_threshold:
                        decision["reason"] = "retained"
                    elif scores[0] <= self.drop_threshold:
                        decision.update(action="drop_pair", reason="low_relevance")
                    else:
                        decision.update(action="truncate_result", reason="replaceable_result")
        except Exception as exc:
            # No exception text: predictor errors can contain private transcript text.
            fallback = "scoring_failed:" + type(exc).__name__

        output = copy.deepcopy(original)
        if fallback is None:
            for decision in decisions:
                if decision["action"] == "keep":
                    continue
                identity = decision["tool_use_id"]
                shortened_any = False
                for index, key in ((decision["call_index"], "toolUses"),
                                   (decision["result_index"], "toolResults")):
                    blocks = output[index][key]
                    if decision["action"] == "drop_pair":
                        output[index][key] = [b for b in blocks if b["tool_use_id"] != identity]
                    else:
                        for block in blocks:
                            if block["tool_use_id"] == identity and "text" in block:
                                text = block["text"]
                                shortened = text[:self.truncate_head_chars] + "\n[veya: replayable output omitted]"
                                if len(shortened) < len(text):
                                    block["text"] = shortened
                                    shortened_any = True
                if decision["action"] == "truncate_result" and not shortened_any:
                    decision.update(action="keep", reason="short_result")
            emptied = {d[key] for d in decisions if d["action"] == "drop_pair"
                       for key in ("call_index", "result_index")}
            output = [m for i, m in enumerate(output) if i not in emptied or m.get("text", "")
                      or m.get("toolUses") or m.get("toolResults") or set(m) - _MESSAGE_KEYS]
        chars_after = len(_serialized(output))
        tokens_after = None if token_counter is None else _integer(
            "token count", token_counter(copy.deepcopy(output)), 0)
        before, after = (chars_before, chars_after) if token_counter is None else (tokens_before, tokens_after)
        reduction = (before - after) / before if before else 0.0
        if candidates and fallback is None and (after >= before or reduction < self.min_reduction_ratio):
            fallback = "insufficient_reduction"
        if fallback is not None:
            output, chars_after, tokens_after, reduction = original, chars_before, tokens_before, 0.0
            for decision in candidates:
                if fallback.startswith("scoring_failed:") or decision["action"] != "keep":
                    decision.update(action="keep", reason=fallback)
        return {"messages": output, "decisions": decisions,
                "stats": {"chars_before": chars_before, "chars_after": chars_after,
                          "messages_before": len(original), "messages_after": len(output),
                          "pairs_dropped": sum(d["action"] == "drop_pair" for d in decisions),
                          "results_shortened": sum(d["action"] == "truncate_result" for d in decisions),
                          "pairs_kept": sum(d["action"] == "keep" for d in decisions),
                          "tokens_before": tokens_before, "tokens_after": tokens_after,
                          "reduction_ratio": reduction, "reduction_unit": "tokens" if token_counter else "chars",
                          "candidates": len(candidates), "batches": batches, "fallback": fallback,
                          "elapsed_seconds": time.perf_counter() - started}}
