"""The Router's memory of recent language detections must answer exactly what `analyse` does.

Routing a state runs `analyse` on it, which reads the whole state; the Router keeps recent
results under a hash of the state so a repeated request skips that. A remembered detection must
equal a fresh one for every state, must be a copy a caller can change without changing the next
answer, must stay bounded, and a state that JSON cannot write exactly is detected afresh rather
than keyed on a lossy form.

Run: python tests/test_route_memo.py
"""
import ast
import glob
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from veya import router as router_mod  # noqa: E402
from veya.lang import analyse  # noqa: E402
from veya.router import Router  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


texts = []
for path in glob.glob(os.path.join(ROOT, "tests", "test_*lang*.py")) + [os.path.join(ROOT, "tests", "test_router.py")]:
    with open(path, encoding="utf-8") as f:
        for node in ast.walk(ast.parse(f.read())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                texts.append(node.value)
rng = random.Random(11)
pools = [range(0x20, 0x7F), range(0xC0, 0x250), range(0x400, 0x500), range(0x900, 0x980), range(0x4E00, 0x4F00)]
for _ in range(2000):
    pool = rng.choice(pools)
    texts.append("".join(chr(rng.choice(rng.choice([pool, pools[0]]))) for _ in range(rng.randint(0, 120))))
states = texts + [None, b"bytes state", ["Hi, I need help", "Nos cobraron dos veces"],
                  {"subject": "Refund", "body": "Nos cobraron dos veces en marzo, por favor devuelvan"},
                  {"a": {"b": ["deep", {"c": "Deutsch ist hier und dort und so weiter und so fort"}]}}]
states += [{"subject": rng.choice(texts), "body": rng.choice(texts)} for _ in range(300)]

router_mod._DETECTIONS.clear()
cold = warm = 0
for state in states:
    want = analyse(state)
    cold += router_mod._detect(state) != want
    warm += router_mod._detect(state) != want
check("memo/cold detections equal analyse (%d states)" % len(states), cold, 0)
check("memo/remembered detections equal analyse", warm, 0)

state = "Влади́мир called about the café résumé twice this week, please call back"
first = router_mod._detect(state)
first["language"] = "xx"
first["script_profile"]["latin"] = -1.0
again = router_mod._detect(state)
check("memo/a caller's change does not reach the next answer", again, analyse(state))
check("memo/each caller gets its own copy", again is router_mod._detect(state), False)

before = len(router_mod._DETECTIONS)
odd = {"body": "An English message about a refund", "when": object()}
check("memo/a state JSON cannot write exactly is detected afresh", router_mod._detect(odd), analyse(odd))
check("memo/and is not remembered", len(router_mod._DETECTIONS), before)
check("memo/str and bytes of the same text are different states",
      router_mod._detection_key("same") != router_mod._detection_key(b"same"), True)

for i in range(router_mod._DETECTIONS_MAX + 50):
    router_mod._detect("state number %d" % i)
check("memo/bounded", len(router_mod._DETECTIONS), router_mod._DETECTIONS_MAX)

r = Router()
one = r.route("Nos cobraron dos veces en marzo. Por favor devuelvan el cargo duplicado hoy.")
two = r.route("Nos cobraron dos veces en marzo. Por favor devuelvan el cargo duplicado hoy.")
check("router/a repeated state routes the same way", dict(one), dict(two))
check("router/routes do not share a detection", one["detection"] is two["detection"], False)
check("router/and it is the multilingual checkpoint", one["model"], "multilingual")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all route memo tests passed")
sys.exit(1 if FAIL else 0)
