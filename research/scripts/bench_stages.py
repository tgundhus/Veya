"""Where a Veya request spends its time, stage by stage, and what changes it.

Every stage a `Router.predict` call goes through is wrapped with a timer: language detection,
the rest of routing, checkpoint lookup, question validation and normalisation, state
tokenization, per-question row building (question and option tokenization), collation, the
encoder, the decision head, the device-to-host copy, answer decoding, and whatever is left
(hooks, context, dict building). Each scenario is also timed with no wrappers installed, which
is the latency to quote; the staged runs say where it goes.

Real checkpoints (downloads on first use; on Apple silicon pass --device mps):

    python research/scripts/bench_stages.py --device mps --out stages_m4.json

Local checkpoints, e.g. synthetic ones with the real architecture:

    python research/scripts/bench_stages.py --checkpoint english=/path/english \\
        --checkpoint multilingual=/path/multilingual --out stages_cpu.json

Variants (--variants, comma separated): threads, bf16, int8, inference_mode, head_cache_off,
mps_fp16, onnx (with --onnx NAME=PATH), compile. Each is timed against the baseline in alternating
runs on the same scenarios; threads, bf16 and int8 apply to CPU only, mps_fp16 to Apple silicon.
"""
import argparse
import contextlib
import functools
import hashlib
from datetime import datetime, timezone
import json
import os
import platform
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import veya  # noqa: E402
import veya.agent as agent_mod  # noqa: E402
import veya.common as common_mod  # noqa: E402
import veya.router as router_mod  # noqa: E402
from veya import DecisionCache, Router  # noqa: E402
from veya.agent import Agent  # noqa: E402

TICKET = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
EMAIL = (
    "Subject: Duplicate charge on invoice INV-2291 and a login problem\n\n"
    "Hello support team,\n\nI am writing about our company account (Acme Logistics, account 88213). "
    "On 3 March we were charged twice for the same monthly subscription: two identical payments of $49 "
    "appear on our card statement, both referencing invoice INV-2291. Our finance department has asked "
    "me to get the duplicate refunded before the end of the month, because it breaks our reconciliation "
    "and we cannot close the books with an unexplained charge.\n\n"
    "Separately, since yesterday two of our dispatchers cannot log in to the web dashboard. They see an "
    "error saying their session expired immediately after entering the one-time code, and clearing the "
    "browser cache did not help. This is slowing down our morning shift, since they have to ask a "
    "colleague to assign routes for them.\n\nCould you please (1) refund the duplicate charge and send "
    "a corrected invoice, and (2) look into the login issue? If the refund cannot be done this week we "
    "will need to reconsider whether to renew our annual plan in April.\n\nThank you,\nMaria Jensen\n"
    "Operations Manager, Acme Logistics\n"
)
TICKET_ES = "Nos cobraron dos veces en marzo. Por favor devuelvan el cargo duplicado hoy o cancelaremos el plan."
EMAIL_ES = (
    "Asunto: Cargo duplicado en la factura INV-2291\n\nHola, equipo de soporte:\n\nEl 3 de marzo nos "
    "cobraron dos veces la misma suscripción mensual: aparecen dos pagos idénticos de 49 dólares en el "
    "extracto de la tarjeta, ambos con la factura INV-2291. Nuestro departamento de finanzas necesita el "
    "reembolso antes de fin de mes para cerrar la contabilidad. Además, desde ayer dos de nuestros "
    "despachadores no pueden iniciar sesión en el panel web: ven un error de sesión caducada justo después "
    "de introducir el código de un solo uso. Si el reembolso no se hace esta semana, tendremos que "
    "reconsiderar la renovación del plan anual en abril.\n\nGracias,\nMaría Jensen\n"
)
TRIAGE = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}
ONE = {"billing": {"type": "noul", "instructions": "Is this about billing?"}}
TEN = dict(TRIAGE, **{
    "refund": {"type": "noul", "instructions": "Does the user ask for a refund?"},
    "login": {"type": "noul", "instructions": "Does the user report a login problem?"},
    "angry": {"type": "noul", "instructions": "Is the tone angry or hostile?"},
    "invoice": {"type": "noul", "instructions": "Does the message reference an invoice number?"},
    "deadline": {"type": "noul", "instructions": "Does the user mention a deadline?"},
    "sentiment": {"type": "choice", "instructions": "Overall sentiment?",
                  "criteria": {"positive": "", "neutral": "", "negative": ""}},
    "language": {"type": "choice", "instructions": "Which language is the message in?",
                 "criteria": {"english": "", "spanish": "", "other": ""}},
})

# name -> (model, state or list of states, questions, extra predict kwargs)
SCENARIOS = {
    "english ticket, 1 question": ("english", TICKET, ONE, {}),
    "english ticket, 3 questions": ("english", TICKET, TRIAGE, {}),
    "english ticket, 10 questions": ("english", TICKET, TEN, {}),
    "english email, 3 questions": ("english", EMAIL, TRIAGE, {}),
    "english batch of 32 tickets, 3 questions": ("english", [TICKET + " #%d" % i for i in range(32)], TRIAGE, {}),
    "multilingual ticket, 3 questions": ("multilingual", TICKET_ES, TRIAGE, {}),
    "multilingual email, 3 questions": ("multilingual", EMAIL_ES, TRIAGE, {}),
}
VARIANT_SCENARIOS = ("english ticket, 3 questions", "english email, 3 questions")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Stages:
    """Installs timers around the functions a request passes through; records seconds per stage."""

    def __init__(self):
        self.t = {}
        self._undo = []

    def add(self, name, dt):
        self.t[name] = self.t.get(name, 0.0) + dt

    def _wrap(self, fn, name, device=None):
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            if device is not None:
                sync(device)
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                if device is not None:
                    sync(device)
                self.add(name, time.perf_counter() - start)
        return timed

    def patch(self, owner, attr, name, device=None, static=False):
        original = owner.__dict__[attr] if isinstance(owner, type) else getattr(owner, attr)
        fn = original.__func__ if static else original
        wrapped = self._wrap(fn, name, device)
        setattr(owner, attr, staticmethod(wrapped) if static else wrapped)
        self._undo.append((owner, attr, original))

    def install(self, agents):
        self.patch(router_mod, "analyse", "routing: language detection")
        self.patch(Router, "route", "routing (total)")
        self.patch(Router, "load", "checkpoint lookup")
        self.patch(Agent, "_check_question", "validate questions", static=True)
        self.patch(Agent, "_to_internal", "normalise questions", static=True)
        self.patch(Agent, "_encode_state", "encode (total)")
        self.patch(agent_mod, "encode_text", "tokenize state")
        self.patch(agent_mod, "build_sequence", "build rows: tokenize questions and options")
        self.patch(agent_mod, "collate_items", "collate tensors")
        self.patch(Agent, "_forward", "forward (total)")
        self.patch(Agent, "_decode_answers", "decode answers")
        for agent in agents:
            if hasattr(agent.model, "encoder"):
                self.patch(agent.model.encoder, "forward", "encoder", device=agent.device)
                self.patch(agent.model, "forward", "model (total)", device=agent.device)

    def uninstall(self):
        for owner, attr, original in reversed(self._undo):
            if isinstance(owner, type) or owner in (router_mod, agent_mod, common_mod):
                setattr(owner, attr, original)
            else:
                delattr(owner, attr)  # instance attribute over the class method
        self._undo.clear()

    def exclusive(self, total):
        t = self.t
        get = t.get
        out = {
            "routing: language detection": get("routing: language detection", 0.0),
            "routing: rest": get("routing (total)", 0.0) - get("routing: language detection", 0.0),
            "checkpoint lookup": get("checkpoint lookup", 0.0),
            "validate questions": get("validate questions", 0.0),
            "normalise questions": get("normalise questions", 0.0),
            "tokenize state": get("tokenize state", 0.0),
            "build rows: tokenize questions and options": get("build rows: tokenize questions and options", 0.0),
            "encode: rest": get("encode (total)", 0.0) - get("tokenize state", 0.0)
            - get("build rows: tokenize questions and options", 0.0),
            "collate tensors": get("collate tensors", 0.0),
            "encoder": get("encoder", 0.0),
            "decision head": get("model (total)", 0.0) - get("encoder", 0.0),
            "host copy and act softmax": get("forward (total)", 0.0) - get("model (total)", 0.0),
            "decode answers": get("decode answers", 0.0),
        }
        out["everything else (hooks, context, results)"] = total - sum(out.values())
        return out


def run(router, scenario, repeat_state=False):
    """One request as a caller makes it: routed by the Router, not pinned to a checkpoint.

    Each run stands for a new request -- a state the Router has not seen, with a question set it
    has -- so the Router's memory of recent language detections is cleared first, and the
    question-head cache is left warm. `repeat_state` keeps both, for a repeated request.
    """
    if not repeat_state and hasattr(router_mod, "_DETECTIONS"):
        router_mod._DETECTIONS.clear()
    _, state, questions, kw = scenario
    if isinstance(state, list):
        return router.predict_batch([{"state": s, "questions": questions} for s in state])
    return router.predict(state, questions, **kw)


def summary(seconds):
    samples = [x * 1e3 for x in seconds]
    ms = sorted(samples)
    return {"samples_ms": [round(x, 6) for x in samples],
            "median": round(statistics.median(ms), 3), "min": round(ms[0], 3),
            "p90": round(ms[min(len(ms) - 1, int(0.9 * len(ms)))], 3)}


def staged_runs(router, agents, scenario, repeats, cm=contextlib.nullcontext):
    stages = []
    for _ in range(repeats):
        prof = Stages()
        with cm():
            prof.install(agents)
            try:
                start = time.perf_counter()
                run(router, scenario)
                total = time.perf_counter() - start
            finally:
                prof.uninstall()
        stages.append(prof.exclusive(total))
    names = list(stages[0])
    med = {n: statistics.median(st[n] for st in stages) * 1e3 for n in names}
    whole = sum(med.values())
    return ({n: round(v, 4) for n, v in med.items()},
            {n: round(v / whole, 4) for n, v in med.items()} if whole > 0 else {})


def measure(router, agents, scenario, repeats, warmup, staged=True):
    for _ in range(warmup):
        run(router, scenario)
    plain = []
    for _ in range(repeats):
        start = time.perf_counter()
        run(router, scenario)
        plain.append(time.perf_counter() - start)
    result = {"latency_ms": summary(plain)}
    if staged:
        result["stages_ms"], result["stages_share"] = staged_runs(router, agents, scenario, repeats)
    return result


def ab(router, agents, scenario, cm, repeats, warmup, staged=False):
    """Baseline and variant runs interleaved A B B A, so drift in the machine cancels out."""
    for _ in range(warmup):
        run(router, scenario)
        with cm():
            run(router, scenario)
    base, var = [], []
    for i in range(repeats):
        for use in ((False, True) if i % 2 == 0 else (True, False)):
            with (cm() if use else contextlib.nullcontext()):
                start = time.perf_counter()
                run(router, scenario)
                (var if use else base).append(time.perf_counter() - start)
    b, v = summary(base), summary(var)
    result = {"baseline_ms": b, "latency_ms": v, "speedup_vs_baseline": round(b["median"] / v["median"], 3)}
    if staged:
        result["stages_ms"], result["stages_share"] = staged_runs(router, agents, scenario, repeats, cm)
    return result


def outputs(router, scenario):
    out = run(router, scenario)
    return out if isinstance(out, list) else [out]


def max_prob_delta(a, b):
    """Largest absolute probability difference between two runs' answers, and label agreement."""
    worst, same, total = 0.0, 0, 0
    for ra, rb in zip(a, b):
        for name, x in ra["answers"].items():
            y = rb["answers"][name]
            px = x.get("probabilities") or {"true": x.get("noul")}
            py = y.get("probabilities") or {"true": y.get("noul")}
            worst = max(worst, max(abs(px[k] - py[k]) for k in px))
            label = lambda v: v.get("choice", v.get("noul", 0) >= 0.5 if "noul" in v else  # noqa: E731
                                     max(v["probabilities"], key=v["probabilities"].get))
            same += label(x) == label(y)
            total += 1
    return {"max_abs_prob_delta": round(worst, 5), "same_decision": "%d/%d" % (same, total)}


def head_cache_off():
    """Variant: `build_sequence` without its per-tokenizer question-head cache (the old path)."""
    @contextlib.contextmanager
    def cm():
        limit = common_mod._QUESTION_HEADS_MAX
        common_mod._QUESTION_HEADS_MAX = 0      # every stored head is evicted at once
        common_mod._QUESTION_HEADS.clear()
        try:
            yield
        finally:
            common_mod._QUESTION_HEADS_MAX = limit
    return cm


def build(args):
    device = args.device
    router = Router(device=device)
    agents = {}
    for name in args.models:
        if name in args.checkpoint:
            agents[name] = router.attach(name, Agent(args.checkpoint[name], device=device))
        else:
            agents[name] = router.load(name)
    return router, agents


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default=None, help="cpu, cuda or mps; default: Veya's own choice")
    parser.add_argument("--models", default="english,multilingual")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help="use a local checkpoint directory for NAME instead of the Hub")
    parser.add_argument("--onnx", action="append", default=[], metavar="NAME=PATH",
                        help="an exported .onnx for NAME (scripts/export_onnx.py); enables the onnx variant")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--scenarios", default="all", help="semicolon-separated scenario names, or all")
    parser.add_argument("--variants", default="threads,bf16,int8,inference_mode,head_cache_off,mps_fp16")
    parser.add_argument("--out", help="write the results here as JSON")
    parser.add_argument("--synthetic-weights", action="store_true", help="label a run that uses synthetic checkpoints")
    args = parser.parse_args()
    args.models = [m for m in args.models.split(",") if m]
    args.checkpoint = dict(item.split("=", 1) for item in args.checkpoint)
    args.onnx = dict(item.split("=", 1) for item in args.onnx)
    variants = [v for v in args.variants.split(",") if v]
    wanted = list(SCENARIOS) if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(";")]
    wanted = [s for s in wanted if SCENARIOS[s][0] in args.models]

    router, agents = build(args)
    first = next(iter(agents.values()))
    device = first.device
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {name: hashlib.sha256(open(os.path.join(ROOT, name), "rb").read()).hexdigest()
                          for name in ("veya/agent.py", "veya/router.py", "veya/common.py",
                                       "veya/inference_common.py", "veya/consistency.py",
                                       "research/scripts/bench_stages.py")},
        "checkpoint_revisions": {n: a.revision for n, a in agents.items()},
        "veya": veya.__version__, "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(),
        "device": str(device), "torch_threads": torch.get_num_threads(),
        "checkpoints": {n: args.checkpoint.get(n, "hub") for n in agents},
        "synthetic_weights": args.synthetic_weights,
        "params_millions": {n: round(sum(p.numel() for p in a.model.parameters()) / 1e6, 1)
                            for n, a in agents.items()},
        "repeats": args.repeats, "warmup": args.warmup,
    }
    results = {"meta": meta, "scenarios": {}, "variants": {}}
    print(json.dumps(meta, indent=2))

    for name in wanted:
        scenario = SCENARIOS[name]
        routed = {o["routing"]["model"] for o in outputs(router, scenario)}
        if routed != {scenario[0]}:
            raise SystemExit("%r routed to %s, expected %s" % (name, sorted(routed), scenario[0]))
        r = measure(router, list(agents.values()), scenario, args.repeats, args.warmup)
        results["scenarios"][name] = r
        print("%-45s %9.2f ms" % (name, r["latency_ms"]["median"]), flush=True)

    # A repeated request through a DecisionCache: routing and the hit are all that is left.
    cached_router = Router(hooks=[DecisionCache()])
    for n, a in agents.items():
        cached_router.attach(n, a)
    for name in [n for n in ("english ticket, 3 questions", "english email, 3 questions") if n in wanted]:
        run(cached_router, SCENARIOS[name])
        hits = []
        for _ in range(max(50, args.repeats)):
            start = time.perf_counter()
            run(cached_router, SCENARIOS[name], repeat_state=True)
            hits.append(time.perf_counter() - start)
        results["scenarios"][name + ", DecisionCache hit"] = {"latency_ms": summary(hits)}
        print("%-45s %9.3f ms" % (name + ", cache hit", statistics.median(hits) * 1e3), flush=True)

    targets = [s for s in VARIANT_SCENARIOS if s in wanted]
    base_out = {s: outputs(router, SCENARIOS[s]) for s in targets}
    all_agents = list(agents.values())

    def attrs(**values):
        """Context manager setting attributes on every agent and restoring them."""
        @contextlib.contextmanager
        def cm():
            saved = [{k: getattr(a, k) for k in values} for a in all_agents]
            for a in all_agents:
                for k, v in values.items():
                    setattr(a, k, v() if callable(v) else v)
            try:
                yield
            finally:
                for a, old in zip(all_agents, saved):
                    for k, v in old.items():
                        setattr(a, k, v)
        return cm

    def threads(n):
        @contextlib.contextmanager
        def cm():
            before = torch.get_num_threads()
            torch.set_num_threads(n)
            try:
                yield
            finally:
                torch.set_num_threads(before)
        return cm

    def swap_models(models):
        @contextlib.contextmanager
        def cm():
            saved = {n: a.model for n, a in agents.items()}
            for n, a in agents.items():
                a.model = models[n]
            try:
                yield
            finally:
                for n, a in agents.items():
                    a.model = saved[n]
        return cm

    plan = []  # (name, context manager factory, staged)
    if "threads" in variants and device.type == "cpu":
        for n in sorted({1, 2} - {torch.get_num_threads()}):
            plan.append(("threads=%d (baseline %d)" % (n, torch.get_num_threads()), threads(n), False))
    if "inference_mode" in variants:
        plan.append(("inference_mode", torch.inference_mode, False))
    if "head_cache_off" in variants and hasattr(common_mod, "_QUESTION_HEADS"):
        plan.append(("question-head cache off", head_cache_off(), True))
    if "bf16" in variants and device.type == "cpu":
        plan.append(("bf16 autocast", attrs(amp_enabled=True, dtype=torch.bfloat16), False))
    if "mps_fp16" in variants and device.type == "mps":
        plan.append(("fp16 autocast from 1 row", attrs(mps_amp_min_rows=1), False))
        plan.append(("fp32 always", attrs(amp_enabled=False, dtype=torch.float32), False))
    if "int8" in variants and device.type == "cpu":
        # The encoder only: quantizing the head's nn.TransformerEncoderLayer breaks the fast-path
        # check in its forward, which reads `.device` off weights that quantization turns into
        # methods. The encoder carries about 94% of the parameters and of the time.
        def encoder_linears(model):
            return {name for name, m in model.named_modules()
                    if name.startswith("encoder.") and isinstance(m, torch.nn.Linear)}
        quantized = {n: torch.ao.quantization.quantize_dynamic(a.model, encoder_linears(a.model),
                                                               dtype=torch.qint8)
                     for n, a in agents.items()}
        plan.append(("int8 dynamic quantization", swap_models(quantized), True))
    if "onnx" in variants and args.onnx:
        from veya.onnx_agent import ONNXAgent

        onnx_agents = {n: ONNXAgent(args.checkpoint.get(n, Router().models[n]), onnx_path=path)
                       for n, path in args.onnx.items() if n in agents}

        def use_onnx():
            @contextlib.contextmanager
            def cm():
                for n, a in onnx_agents.items():
                    router.attach(n, a)
                try:
                    yield
                finally:
                    for n in onnx_agents:
                        router.attach(n, agents[n])
            return cm
        plan.append(("ONNX Runtime (ONNXAgent)", use_onnx(), False))
    if "compile" in variants:
        compiled = {n: torch.compile(a.model, dynamic=True) for n, a in agents.items()}
        plan.append(("torch.compile", swap_models(compiled), False))

    for variant, cm, staged in plan:
        for s in targets:
            if variant == "torch.compile":
                with cm():
                    start = time.perf_counter()
                    run(router, SCENARIOS[s])
                    first_ms = (time.perf_counter() - start) * 1e3
            r = ab(router, all_agents, SCENARIOS[s], cm, args.repeats, args.warmup, staged=staged)
            if variant == "torch.compile":
                r["first_call_ms_incl_compile"] = round(first_ms, 1)
            with cm():
                r["vs_base_outputs"] = max_prob_delta(base_out[s], outputs(router, SCENARIOS[s]))
            results["variants"].setdefault(variant, {})[s] = r
            print("  %-34s %-34s %9.2f ms vs %9.2f ms  x%.3f  %s" % (
                variant, s, r["latency_ms"]["median"], r["baseline_ms"]["median"], r["speedup_vs_baseline"],
                r["vs_base_outputs"]), flush=True)

    text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
