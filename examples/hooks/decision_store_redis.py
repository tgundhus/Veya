"""Share DecisionCache decisions between machines through Redis.

`RedisDecisionStore` implements `veya.DecisionStore`: each decision lives under
`veya:decision:<key>` with Redis's own expiry, so every process and machine pointed at the same
Redis replays the same decision, and the first one stored wins (`SET ... NX`). Redis drops
expired decisions itself; bound its size with a `maxmemory` policy.

    import redis
    from veya import DecisionCache, Router

    store = RedisDecisionStore(redis.Redis(host="cache.internal"))
    router = Router(hooks=[DecisionCache(store=store, ttl=30 * 24 * 3600, renew_on_hit=True)])

Run this file to check the store against a real Redis (REDIS_URL) or, without one, against
fakeredis (`pip install redis fakeredis`). The check stands two Routers, as two machines, over a
stand-in model, so it needs no checkpoint download.

    python examples/hooks/decision_store_redis.py
"""
import math
import os


class RedisDecisionStore:
    """A `veya.DecisionStore` kept in Redis. `client` is a `redis.Redis` returning bytes."""

    def __init__(self, client, prefix: bytes = b"veya:decision:"):
        self.client = client
        self.prefix = prefix

    def _name(self, key: bytes) -> bytes:
        return self.prefix + key.hex().encode("ascii")

    @staticmethod
    def _ms(expires_at: float, now: float):
        return None if math.isinf(expires_at) else max(1, int((expires_at - now) * 1000))

    def get(self, key, now):
        pipe = self.client.pipeline()
        pipe.get(self._name(key))
        pipe.pttl(self._name(key))
        value, pttl = pipe.execute()
        if value is None or pttl == -2:
            return None
        return value, (math.inf if pttl == -1 else now + pttl / 1000.0)

    def add(self, items, now):
        held = []
        for key, value, expires_at in items:
            name = self._name(key)
            if self.client.set(name, value, nx=True, px=self._ms(expires_at, now)):
                held.append(value)
                continue
            first = self.client.get(name)
            if first is None:  # it expired between the two calls: store ours after all
                self.client.set(name, value, nx=True, px=self._ms(expires_at, now))
                first = self.client.get(name) or value
            held.append(first)
        return held

    def renew(self, key, expires_at, now):
        name = self._name(key)
        if math.isinf(expires_at):
            self.client.persist(name)
        else:
            self.client.pexpire(name, self._ms(expires_at, now))

    def prune(self, now):
        return 0  # Redis expires keys itself

    def __len__(self):
        return sum(1 for _ in self.client.scan_iter(match=self.prefix + b"*", count=1000))

    def clear(self):
        for name in self.client.scan_iter(match=self.prefix + b"*", count=1000):
            self.client.delete(name)

    def close(self):
        self.client.close()


def _client():
    url = os.environ.get("REDIS_URL")
    if url:
        import redis
        return redis.Redis.from_url(url)
    import fakeredis
    return fakeredis.FakeRedis()


class _StandIn:
    """Answers like a Veya checkpoint would, with its own P(yes): one per 'machine'."""

    def __init__(self, p):
        self.p, self.calls = p, 0

    def system_one(self, state, questions, **kw):
        self.calls += 1
        return {"model": "veya-rl-agent", "usage": {"input_tokens": 12, "output_tokens": 0},
                "answers": {q: {"type": "noul", "noul": self.p} for q in questions}}


if __name__ == "__main__":
    from veya import DecisionCache, Router

    client = _client()
    store = RedisDecisionStore(client, prefix=b"veya:example:")
    store.clear()
    machines = []
    for p in (0.51, 0.49):   # two machines whose arithmetic lands either side of a close call
        router = Router(hooks=[DecisionCache(store=RedisDecisionStore(client, prefix=b"veya:example:"),
                                             ttl=3600, renew_on_hit=True)])
        model = router.attach("english", _StandIn(p))
        machines.append((router, model))
    questions = {"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}}
    state = "Hi, we were billed twice for March. Please refund the duplicate today."

    first = machines[0][0].predict(state, questions)["answers"]["refund"]["noul"]
    second = machines[1][0].predict(state, questions)["answers"]["refund"]["noul"]
    assert first == second == 0.51, (first, second)
    assert machines[1][1].calls == 0, "the second machine should replay, not run its model"
    assert len(store) == 1
    name = next(iter(client.scan_iter(match=b"veya:example:*")))
    assert 0 < client.pttl(name) <= 3600 * 1000, "the decision should carry Redis's own expiry"
    print("both machines answered %.2f; the second never ran its model; one decision in Redis" % second)

    # The store's own rules: the first value stored under a key wins, and renew moves the expiry.
    store.clear()
    key, now = b"\x01" * 16, 1_000_000.0
    assert store.add([(key, b"first", math.inf)], now) == [b"first"]
    assert store.add([(key, b"second", math.inf)], now) == [b"first"], "SET NX must keep the first"
    assert store.get(key, now) == (b"first", math.inf)
    store.renew(key, now + 10, now)
    assert 0 < client.pttl(store._name(key)) <= 10_000
    store.renew(key, math.inf, now)
    assert client.pttl(store._name(key)) == -1
    print("first write wins under SET NX; renew sets and clears the expiry")
    store.clear()
