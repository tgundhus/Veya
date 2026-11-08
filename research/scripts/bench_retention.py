"""Does DecisionCache keep decisions for exactly as long as it is configured to, and what does that buy?

No model is needed. Requests go through the cache's own hooks, as a Router drives them, over an
agent that answers instantly with a numbered payload, so a replay can be told from a recompute.
The cache's clock (`veya.consistency._now`) is simulated, so a month runs in seconds.

Sections:

    exactness   every hit and miss is checked against a reference model of the documented rules:
                a fixed ttl, renew_on_hit (a replay moves the expiry out, written only once it would
                move by more than ttl/16), a ttl function of the question set and the answer
                (0 keeps nothing, None keeps for ever), and the maxsize bound. Counts disagreements.
    simulate    a month of traffic, recurring requests with Zipf popularity plus one-off ones,
                through each retention setting: the share of requests replayed, the share of
                repeats that got the decision their request got before, forward passes, decisions
                held, and store writes; and the memory store restarted every day against SQLite.
    restart     real process restarts over one SQLite file: what a fresh process replays, byte for
                byte; nothing after a model upgrade (a new fingerprint); nothing past its ttl.

    python research/scripts/bench_retention.py --out retention.json
"""
import argparse
import bisect
import contextlib
import json
import math
import os
import platform
import random
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import veya  # noqa: E402
from veya import DecisionCache, decision_margins  # noqa: E402
from veya import consistency  # noqa: E402
from veya.hooks import PredictContext  # noqa: E402

HOUR, DAY = 3600.0, 86400.0
ARCHIVE = os.path.join(ROOT, "research", "benchmarks", "feishu_zh", "results", "v1", "laya", "raw.jsonl")
DECISION = {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text",
            "detection": {"script": "latin", "language": "en"}, "workflow": None}
QUESTION_SETS = {
    "billing": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    "urgency": {"urgency": {"type": "score", "instructions": "How urgent is this?",
                            "criteria": ["not urgent", "soon", "blocking"]}},
    "spam": {"spam": {"type": "noul", "instructions": "Is this spam?"}},
    "policy": {"policy": {"type": "noul", "instructions": "Does this break the posting policy?"}},
}


class Clock:
    def __init__(self, t=1_750_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeAgent:
    """Instant answers carrying a serial number; the identity attributes a real Agent has."""

    model_id = "convaiinnovations/laya"
    revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    cfg = {"max_len": 512, "head_max_len": 192}
    temperature = None
    temperature_by_options = None
    lang_temperatures = {}


def base_payload():
    for line in open(ARCHIVE, encoding="utf-8"):
        row = json.loads(line)
        if row["mode"] == "four_noul":
            return row["response"]
    raise SystemExit("no four_noul row in " + ARCHIVE)


class Driver:
    """Sends one-state requests through a cache the way a Router does, counting forward passes."""

    def __init__(self, cache, payload, p_yes=None):
        self.cache = cache
        self.payload = payload
        self.agent = FakeAgent()
        self.router = object()
        self.computed = 0
        self.p_yes = p_yes  # key -> P(yes) of the computed answer, to give answers margins

    def request(self, state, questions):
        ctx = PredictContext(states=[state], questions=questions, decision=DECISION, model="english",
                             agent=self.agent, router=self.router)
        self.cache.on_predict_start(ctx)
        if ctx.results is not None and len(ctx.states) == 1 and ctx.results[0] is not None:
            self.cache.on_predict_end(ctx)
            return ctx.results[0], True
        self.computed += 1
        result = dict(self.payload, serial=self.computed)
        if self.p_yes is not None:
            p = self.p_yes(state)
            result["answers"] = {q: {"type": "noul", "noul": p, "confidence": p} for q in questions}
        ctx.results = [result]
        self.cache.on_predict_end(ctx)
        return ctx.results[0], False


def count_renewals(cache):
    """Wrap the store's `renew` to count writes per key."""
    store = cache._store
    counts = {}
    inner = store.renew

    def renew(key, expires_at, now):
        counts[key] = counts.get(key, 0) + 1
        return inner(key, expires_at, now)
    store.renew = renew
    return counts


# ------------------------------------------------------------------------------------ exactness

def exactness_ttl(make_cache, clock, payload, ttl, renew, n_keys=2000, n_events=40_000, hot=0.3, seed=1):
    """Random requests over 3 ttl; each outcome against the documented expiry and renewal rules.

    A `hot` share of requests go to one request, hit far more often than every ttl/16, so its
    renewal writes show the ceiling of 16 per ttl.
    """
    rng = random.Random(seed)
    cache = make_cache(ttl=ttl, renew_on_hit=renew, maxsize=None)
    renewals = count_renewals(cache)
    drv = Driver(cache, payload)
    questions = QUESTION_SETS["billing"]
    times = sorted(rng.uniform(0, 3 * ttl) for _ in range(n_events))
    shadow = {}  # state -> [serial, expires_at]
    start = clock.t
    wrong_outcome = wrong_value = hits = 0
    for t in times:
        clock.t = start + t
        now = clock.t
        state = "hot request" if rng.random() < hot else "request %d" % rng.randrange(n_keys)
        result, hit = drv.request(state, questions)
        entry = shadow.get(state)
        expect_hit = entry is not None and entry[1] > now
        if hit != expect_hit:
            wrong_outcome += 1
        if hit:
            hits += 1
            if result["serial"] != entry[0]:
                wrong_value += 1
            if renew and (now + ttl) - entry[1] > ttl * consistency._RENEW_FRACTION:
                entry[1] = now + ttl
        else:
            shadow[state] = [result["serial"], now + ttl]
    # Renewal writes per key per ttl of time: the documented ceiling is 16.
    worst = max(renewals.values()) / 3.0 if renewals else 0.0
    cache.close()
    return {"events": n_events, "hits": hits, "outcome_disagreements": wrong_outcome,
            "replayed_other_decision": wrong_value, "renewal_writes": sum(renewals.values()),
            "renewal_writes_per_key_per_ttl_max": round(worst, 2),
            "store_writes_per_hit": round((sum(renewals.values())) / hits, 4) if hits else None}


def exactness_policy(make_cache, clock, payload, n_events=40_000, seed=2):
    """A ttl function: per question set, and shorter for borderline answers."""
    def ttl(questions, result):
        name = next(iter(questions))
        if name == "spam":
            return 0            # never kept
        if name == "policy":
            return None         # kept for ever
        if name == "billing":
            return 7 * DAY
        margin = min(decision_margins(result).values() or [1.0])
        return 5 * 60 if margin < 0.1 else HOUR   # urgency: borderline answers expire in 5 minutes

    rng = random.Random(seed)
    p_of = {}

    def p_yes(state):
        return p_of.setdefault(state, round(rng.choice([0.51, 0.53, 0.9, 0.97, 0.2]), 4))
    cache = make_cache(ttl=ttl, renew_on_hit=False, maxsize=None)
    drv = Driver(cache, payload, p_yes=p_yes)
    start = clock.t
    times = sorted(rng.uniform(0, 14 * DAY) for _ in range(n_events))
    shadow = {}
    wrong = hits = 0
    per_set = {name: {"requests": 0, "hits": 0} for name in QUESTION_SETS}
    for t in times:
        clock.t = start + t
        name = rng.choice(list(QUESTION_SETS))
        state = "item %d" % rng.randrange(300)
        result, hit = drv.request(state, QUESTION_SETS[name])
        key = (name, state)
        entry = shadow.get(key)
        expect_hit = entry is not None and entry > clock.t
        wrong += hit != expect_hit
        hits += hit
        per_set[name]["requests"] += 1
        per_set[name]["hits"] += hit
        if not hit:
            life = ttl(QUESTION_SETS[name], result)
            if life is None:
                shadow[key] = math.inf
            elif life > 0:
                shadow[key] = clock.t + life
    cache.close()
    return {"events": n_events, "hits": hits, "outcome_disagreements": wrong, "per_question_set": per_set}


def exactness_maxsize(make_cache, clock, payload, maxsize=1000, n_events=20_000, seed=3):
    """No expiry, a size bound: evicted decisions are always the least recently stored or renewed."""
    rng = random.Random(seed)
    cache = make_cache(ttl=None, renew_on_hit=False, maxsize=maxsize)
    drv = Driver(cache, payload)
    order = []          # states in the order they were stored, oldest first (no renewals here)
    held_max = 0
    wrong_evictions = 0
    for i in range(n_events):
        clock.t += 1.0
        # Mostly new requests, some repeats of recent ones: repeats hit and must not reorder.
        if order and rng.random() < 0.3:
            state = order[-1 - rng.randrange(min(len(order), maxsize // 2))]
        else:
            state = "fresh %d" % i
        _, hit = drv.request(state, QUESTION_SETS["billing"])
        if not hit:
            order.append(state)
        held_max = max(held_max, len(cache._store))
    # The keys still held must be the newest ones stored.
    keys = cache._keys(PredictContext(states=order, questions=QUESTION_SETS["billing"], decision=DECISION,
                                      model="english", agent=drv.agent, router=drv.router), order, False)
    held = [cache._store.get(k, clock.t) is not None for k in keys]
    n_held = sum(held)
    wrong_evictions = sum(1 for i, h in enumerate(held) if h != (i >= len(order) - n_held))
    cache.close()
    return {"maxsize": maxsize, "stored": len(order), "held_at_end": n_held, "held_max_seen": held_max,
            "held_not_newest": wrong_evictions}


def run_exactness(tmp, payload):
    clock = Clock()
    consistency._now = clock
    out = {}
    stores = {
        "memory": lambda **kw: DecisionCache(**kw),
        "sqlite": lambda **kw: DecisionCache(os.path.join(tmp, "exact_%d.sqlite" % time.perf_counter_ns()), **kw),
    }
    for name, make in stores.items():
        out[name] = {
            "fixed ttl 1 h": exactness_ttl(make, clock, payload, HOUR, renew=False),
            "fixed ttl 30 days": exactness_ttl(make, clock, payload, 30 * DAY, renew=False),
            "ttl 1 h, renew_on_hit": exactness_ttl(make, clock, payload, HOUR, renew=True),
            "ttl function (per question set, 5 min for borderline answers)": exactness_policy(make, clock, payload),
            "maxsize 1,000, no ttl": exactness_maxsize(make, clock, payload),
        }
    consistency._now = time.time
    return out


# ------------------------------------------------------------------------------------- simulate

def zipf_sampler(n, s, rng):
    weights = [1.0 / (k ** s) for k in range(1, n + 1)]
    cum, total = [], 0.0
    for w in weights:
        total += w
        cum.append(total)
    return lambda: bisect.bisect_left(cum, rng.random() * total)


def trace(n_requests, days, universe, one_off, s, seed):
    rng = random.Random(seed)
    draw = zipf_sampler(universe, s, rng)
    times = sorted(rng.uniform(0, days * DAY) for _ in range(n_requests))
    out = []
    for i, t in enumerate(times):
        out.append((t, "one-off %d" % i if rng.random() < one_off else "recurring %d" % draw()))
    return out


def simulate(events, make_cache, clock, payload, restart_every=None):
    cache = make_cache()
    drv = Driver(cache, payload)
    questions = QUESTION_SETS["billing"]
    start = clock.t
    seen = set()
    repeats = repeats_replayed = 0
    held_max = 0
    next_restart = restart_every
    adds = hits = 0
    renewals = count_renewals(cache)
    for n, (t, state) in enumerate(events):
        clock.t = start + t
        if next_restart is not None and t >= next_restart:
            cache.cache_clear()          # a restart: an in-memory store starts empty
            next_restart += restart_every
        _, hit = drv.request(state, questions)
        hits += hit
        if state in seen:
            repeats += 1
            repeats_replayed += hit
        else:
            seen.add(state)
        adds += not hit
        if n % 5000 == 0:
            held_max = max(held_max, len(cache._store))
    held_end = len(cache._store)
    cache.close()
    return {"requests": len(events), "replayed": hits, "replayed_share": round(hits / len(events), 4),
            "repeats": repeats, "repeats_given_earlier_decision": repeats_replayed,
            "repeats_given_earlier_decision_share": round(repeats_replayed / repeats, 4) if repeats else None,
            "forward_passes": drv.computed, "re_decisions": repeats - repeats_replayed,
            "held_max": max(held_max, held_end), "held_end": held_end,
            "store_writes": adds + sum(renewals.values()), "renewal_writes": sum(renewals.values())}


def run_simulation(tmp, payload, n_requests, days, universe, one_off, s):
    events = trace(n_requests, days, universe, one_off, s, seed=7)
    clock = Clock()
    consistency._now = clock
    out = {"trace": {"requests": n_requests, "days": days, "recurring_universe": universe, "zipf_s": s,
                     "one_off_share": one_off, "distinct": len({e[1] for e in events})}, "runs": {}}
    configs = []
    for ttl_name, ttl in (("1 h", HOUR), ("1 day", DAY), ("7 days", 7 * DAY), ("30 days", 30 * DAY), ("none", None)):
        for renew in (False, True):
            if ttl is None and renew:
                continue
            configs.append(("memory, ttl %s%s" % (ttl_name, ", renew_on_hit" if renew else ""),
                            dict(ttl=ttl, renew_on_hit=renew), None, None))
    for maxsize in (10_000, 30_000, None):
        configs.append(("memory, ttl 7 days, renew_on_hit, maxsize %s" % (maxsize,),
                        dict(ttl=7 * DAY, renew_on_hit=True, maxsize=maxsize), None, None))
    configs.append(("memory, ttl 7 days, renew_on_hit, restarted daily",
                    dict(ttl=7 * DAY, renew_on_hit=True), None, DAY))
    configs.append(("sqlite, ttl 7 days, renew_on_hit (survives restarts)",
                    dict(ttl=7 * DAY, renew_on_hit=True), "sqlite", None))
    for name, kw, store, restart in configs:
        def make(kw=kw, store=store):
            if store == "sqlite":
                return DecisionCache(os.path.join(tmp, "sim_%d.sqlite" % time.perf_counter_ns()), **kw)
            return DecisionCache(**kw)
        t = time.perf_counter()
        out["runs"][name] = simulate(events, make, clock, payload, restart_every=restart)
        out["runs"][name]["wall_s"] = round(time.perf_counter() - t, 1)
        r = out["runs"][name]
        kept = r["repeats_given_earlier_decision_share"]  # None when the trace repeats nothing
        print("  %-58s replayed %5.1f%%  repeats given earlier decision %s  held max %7d  writes %7d"
              % (name, 100 * r["replayed_share"], "   n/a" if kept is None else "%5.1f%%" % (100 * kept),
                 r["held_max"], r["store_writes"]), flush=True)
    consistency._now = time.time
    return out


# -------------------------------------------------------------------------------------- restart

CHILD = r"""
import json, os, sys, time
sys.path.insert(0, %(root)r)
sys.path.insert(0, %(here)r)
from bench_retention import Driver, QUESTION_SETS, base_payload
from veya import DecisionCache, consistency
job = json.loads(sys.argv[1])
if job.get("advance"):
    consistency._now = lambda: time.time() + job["advance"]
t = time.perf_counter()
cache = DecisionCache(job["path"], ttl=job["ttl"], fingerprint=job["fingerprint"])
open_s = time.perf_counter() - t
drv = Driver(cache, base_payload())
first = None
hits, identical, lat = 0, 0, []
for i in range(job["n"]):
    state = "persisted request %%d" %% i
    t = time.perf_counter()
    result, hit = drv.request(state, QUESTION_SETS["billing"])
    lat.append(time.perf_counter() - t)
    if first is None:
        first = lat[-1]
    hits += hit
    identical += hit and result.get("serial") == i + 1
lat.sort()
print(json.dumps({"open_ms": round(open_s * 1e3, 3), "first_request_ms": round(first * 1e3, 3),
                  "median_request_us": round(lat[len(lat) // 2] * 1e6, 2),
                  "p99_request_us": round(lat[int(0.99 * len(lat))] * 1e6, 2),
                  "replayed": hits, "replayed_first_decision": identical, "computed": drv.computed,
                  "held": len(cache._store)}))
cache.close()
"""


def run_restart(tmp, n):
    path = os.path.join(tmp, "restart.sqlite")
    script = CHILD % {"root": ROOT, "here": HERE}

    def child(**job):
        job = dict({"path": path, "n": n, "ttl": 30 * DAY, "fingerprint": "model-v1"}, **job)
        done = subprocess.run([sys.executable, "-c", script, json.dumps(job)], check=True,
                              capture_output=True, text=True)
        return json.loads(done.stdout.strip().splitlines()[-1])
    out = {"decisions": n}
    out["1. first process computes and stores"] = child()
    out["2. a fresh process, same model"] = child()
    out["3. a fresh process after a model upgrade (new fingerprint)"] = child(fingerprint="model-v2", ttl=None)
    out["4. a fresh process 31 days later, same model"] = child(advance=31 * DAY)
    with contextlib.closing(sqlite3.connect(path)) as db:
        out["file_bytes"] = os.path.getsize(path)
        out["rows"] = db.execute("SELECT COUNT(*) FROM laya_decisions").fetchone()[0]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sections", default="exactness,simulate,restart")
    parser.add_argument("--requests", type=int, default=300_000, help="simulated requests")
    parser.add_argument("--days", type=float, default=30)
    parser.add_argument("--universe", type=int, default=60_000, help="distinct recurring requests")
    parser.add_argument("--one-off", type=float, default=0.3, help="share of requests never seen again")
    parser.add_argument("--zipf", type=float, default=1.0)
    parser.add_argument("--restart-decisions", type=int, default=20_000)
    parser.add_argument("--out")
    args = parser.parse_args()
    sections = [s for s in args.sections.split(",") if s]
    payload = base_payload()
    results = {"meta": {"laya": veya.__version__, "python": platform.python_version(), "machine": platform.machine(),
                        "platform": platform.platform(), "sqlite": sqlite3.sqlite_version,
                        "renew_fraction": consistency._RENEW_FRACTION}}
    with tempfile.TemporaryDirectory() as tmp:
        if "exactness" in sections:
            print("exactness", flush=True)
            results["exactness"] = run_exactness(tmp, payload)
            print(json.dumps(results["exactness"], indent=1))
        if "simulate" in sections:
            print("simulate", flush=True)
            results["simulate"] = run_simulation(tmp, payload, args.requests, args.days, args.universe,
                                                 args.one_off, args.zipf)
        if "restart" in sections:
            print("restart", flush=True)
            results["restart"] = run_restart(tmp, args.restart_decisions)
            print(json.dumps(results["restart"], indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
