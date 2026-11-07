"""Render README comparisons from recorded results, without running models or APIs.

Run: python research/scripts/make_readme_comparisons.py
Requires Matplotlib for authoring only; it is not an inference dependency.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "research/results"
ARCHIVE = "research/benchmarks/feishu_zh/results/v1"
MATCHED_VEYA = "research/results/matched_feishu_veya_cpu_20261107"
PAPER, INK, MUTED, GRID = "#f5f7fb", "#152238", "#516278", "#dfe5ef"
JEV, LAYA, VEYA = "#b55b20", "#336bc0", "#087f68"


def recorded_facts():
    """Recompute recorded counts and check the inputs behind every plotted number."""
    sources, facts, hashes = {}, {}, {}

    def load(relative, lines=False):
        path = ROOT / relative
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines() if line.strip()] if lines else json.loads(raw)

    for backend in ("jev", "laya", "veya"):
        folder = MATCHED_VEYA if backend == "veya" else ARCHIVE + "/" + backend
        rows = load(folder + "/raw.jsonl", lines=True)
        groups = defaultdict(list)
        for row in rows:
            assert row["status"] == "ok", "Failed requests must not disappear from the denominator"
            groups[row["mode"], row["id"]].append(row)
        assert len(groups) == 128 and all({r["repeat"] for r in g} == {0, 1, 2}
                                          and len(g) == 3 for g in groups.values())
        first = [row for row in rows if row["mode"] == "choice" and row["repeat"] == 0]
        binary = [row for row in rows if row["mode"] == "four_noul" and row["repeat"] == 0]
        assert len(first) == len(binary) == 64
        facts[backend] = {
            "choice_correct": sum(row["predicted"] == row["expected"] for row in first),
            "choice_cases": len(first),
            "four_noul_correct": sum(row["predicted"] == row["expected"] for row in binary),
            "four_noul_cases": len(binary),
            "identical_groups": sum(all(row["response"]["answers"] == group[0]["response"]["answers"]
                                        for row in group) for group in groups.values()),
            "repeat_groups": len(groups),
        }
        hashes[backend] = {(row["mode"], row["id"], row["repeat"]): row["request_sha256"] for row in rows}
    assert hashes["jev"] == hashes["laya"] == hashes["veya"], "The comparison must use identical requests"
    quality = load("research/results/matched_quality_comparison_20261107.json")
    for name, digest in quality["source_sha256"].items():
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert sources[name] == digest, "Matched quality source changed: " + name
    for backend in facts:
        assert quality["models"][backend]["choice"]["correct"] == facts[backend]["choice_correct"]
        assert quality["models"][backend]["four_noul"]["correct"] == facts[backend]["four_noul_correct"]

    refresh = load("research/results/veya_refresh_cpu_20261107.json")
    for name, digest in refresh["artifacts"].items():
        load("research/results/" + name)
        assert sources["research/results/" + name] == digest, "CPU source artifact changed: " + name
    assert refresh["answers_identical"] and refresh["answers_compared"] == 17
    facts["veya"].update(answers_identical=refresh["answers_compared"], latency=[])
    latency = {row["workload"]: row["after_median_ms"] for row in refresh["latency"]}
    for name, label in (("english ticket, 3 questions", "Ticket\n3 questions"),
                        ("english email, 3 questions", "Email\n3 questions")):
        computed, cached = latency[name], latency[name + ", DecisionCache hit"]
        assert 0 < cached < computed
        facts["veya"]["latency"].append({"label": label, "computed_ms": computed, "cached_ms": cached})

    consistency = load("research/results/backend_consistency_m4max_20260928.json")
    facts["precision"] = []
    for setup, label in (("mps-default-batched", "Apple GPU batch\nFP16"),
                         ("cpu-int8", "CPU INT8"), ("onnx-int8", "ONNX INT8\nEnglish only")):
        models = [m for m in consistency["models"].values() if setup in m["vs_cpu_fp32"]]
        changed = total = requests = 0
        for model in models:
            computed = model["vs_cpu_fp32"][setup]
            replay = model["decision_cache_replay"]["setups"][setup]
            assert computed["same_decision"] + computed["changed_decisions"] == computed["answers"]
            assert replay["identical_to_first_decision"] == replay["requests"] == computed["requests"]
            changed += computed["changed_decisions"]
            total += computed["answers"]
            requests += replay["requests"]
        facts["precision"].append({"label": label, "changed_answers": changed, "answers": total,
                                   "replayed_requests": requests, "replay_changed_answers": 0})
    eviction = load("research/results/cache_eviction_production_20261107.json")
    assert eviction["parameters"]["production_lru"] and eviction["parameters"]["repeats"] >= 7
    facts["eviction"] = [case for case in eviction["cases"] if case["capacity"] == 1024]
    assert len(facts["eviction"]) == 4
    workflow = load("research/results/cache_workflow_real_20261107.json")
    assert workflow["reference_metadata_sha256"] == sources[MATCHED_VEYA + "/metadata.json"]
    assert workflow["reference_raw_sha256"] == sources[MATCHED_VEYA + "/raw.jsonl"]
    for policy in workflow["policies"].values():
        assert len(policy["trials"]) >= 3 and all(t["changed_answer_payloads"] == 0 for t in policy["trials"])
        assert len({t["forward_passes"] for t in policy["trials"]}) == 1
    facts["workflow"] = workflow
    return {"sources_sha256": sources, "facts": facts,
            "scope": "Jev/Laya: September Chinese diagnostic. Veya: November, same frozen prompts and scoring, "
                     "same pinned weights as Laya. CPU/cache measurements have separate dates and hardware. "
                     "No three-way latency ranking; synthetic cache hit rates are separate from real model timing."}


def text(fig, x, y, value, size=11, color=INK, weight="normal", **kwargs):
    return fig.text(x, y, value, fontsize=size, color=color, weight=weight, **kwargs)


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=160, facecolor=PAPER, metadata={"Software": "Veya benchmark figures"})
    plt.close(fig)
    print("wrote assets/" + name)


def overview(facts):
    fig = plt.figure(figsize=(12, 7), facecolor=PAPER)
    text(fig, .04, .95, "VEYA  /  RECORDED COMPARISONS", 10, MUTED, "bold")
    text(fig, .04, .875, "Jev, Laya and Veya", 32, weight="bold")
    text(fig, .04, .827, "Matched questions. Same scoring rules. Measured quality and repeatability.", 12, MUTED)
    cards = [(.04, "Jev", "API  /  1.13.0  /  SEP 2026", JEV, "jev"),
             (.353, "Laya", "LOCAL  /  MULTILINGUAL  /  SEP 2026", LAYA, "laya"),
             (.666, "Veya", "LOCAL  /  MULTILINGUAL  /  NOV 2026", VEYA, "veya")]
    for x, name, badge, color, key in cards:
        card = FancyBboxPatch((x, .225), .294, .54, boxstyle="round,pad=0.007,rounding_size=0.016",
                              transform=fig.transFigure, facecolor="white", edgecolor=GRID, linewidth=1)
        fig.add_artist(card)
        text(fig, x + .018, .705, name, 23, color, "bold")
        text(fig, x + .018, .668, badge, 8.1, MUTED, "bold")
        fig.add_artist(Line2D([x + .018, x + .276], [.635, .635], transform=fig.transFigure,
                             color=GRID, linewidth=1))
        row = facts[key]
        score = 100 * row["choice_correct"] / row["choice_cases"]
        binary = 100 * row["four_noul_correct"] / row["four_noul_cases"]
        text(fig, x + .018, .553, "%g%%" % score, 33, color, "bold")
        text(fig, x + .018, .508, "Choice labels matched: %d / %d" % (
            row["choice_correct"], row["choice_cases"]), 10.5)
        text(fig, x + .018, .407, "%g%%" % binary, 27, color, "bold")
        text(fig, x + .018, .363, "Four binary questions → label: %d / %d" % (
            row["four_noul_correct"], row["four_noul_cases"]), 9.3)
        text(fig, x + .018, .277, "Exact 3-run answer groups: %d / %d" % (
            row["identical_groups"], row["repeat_groups"]), 9.5, MUTED)
    text(fig, .04, .158, "64 synthetic Chinese cases × 2 modes × 3 repeats. Quality uses the fixed first repeat; no failed requests.", 10, MUTED)
    text(fig, .04, .112, "Laya / Veya use identical pinned weights. Jev / Laya: 21 Sep; Veya: 7 Nov. Hardware differs; timings are not ranked.", 9.5, MUTED)
    text(fig, .04, .05, "Sources: frozen Feishu protocol and audited raw responses. File paths and SHA256: companion chart manifest.", 8.6, MUTED)
    save(fig, "decision_engine_comparison.png")


def replay_chart(facts):
    fig = plt.figure(figsize=(12, 7.7), facecolor=PAPER)
    text(fig, .04, .95, "VEYA  /  INFERENCE AND REPLAY", 10, MUTED, "bold")
    text(fig, .04, .878, "Fast repeats. Stable stored answers.", 28, weight="bold")
    fig.legend([Line2D([], [], color=LAYA, linewidth=5), Line2D([], [], color=VEYA, linewidth=5)],
               ["New inference", "Stored answer replay"], loc="center left", bbox_to_anchor=(.03, .814),
               ncol=2, frameon=False, fontsize=11, labelcolor=INK, columnspacing=2.4)
    text(fig, .04, .733, "Current Veya: compute vs cache", 13, weight="bold")
    text(fig, .04, .698, "Same English checkpoint  /  7 Nov 2026", 9.5, MUTED)
    text(fig, .575, .733, "Laya compute vs Veya replay", 13, weight="bold")
    text(fig, .575, .698, "Changed labels vs CPU FP32  /  28 Sep 2026", 9.5, MUTED)
    left = fig.add_axes([.125, .32, .325, .33], facecolor=PAPER)
    for i, row in enumerate(facts["veya"]["latency"]):
        y = 1 - i
        for delta, key, color in ((.16, "computed_ms", LAYA), (-.16, "cached_ms", VEYA)):
            value = row[key]
            left.hlines(y + delta, .01, value, color=color, linewidth=5)
            left.scatter(value, y + delta, color=color, s=40, zorder=3)
            left.text(value * 1.25, y + delta, "%.3f ms" % value, va="center", fontsize=9.5, color=INK)
    left.set_xscale("log")
    left.set_xlim(.01, 20000)
    left.set_ylim(-.5, 1.5)
    left.set_yticks([1, 0], [r["label"] for r in facts["veya"]["latency"]], fontsize=10)
    left.set_xticks([.01, 1, 100, 10000], ["0.01", "1", "100", "10,000"])
    left.set_xlabel("Median milliseconds  /  logarithmic scale", fontsize=9.5, labelpad=10)

    right = fig.add_axes([.723, .32, .215, .33], facecolor=PAPER)
    for i, row in enumerate(facts["precision"]):
        y = 2 - i
        percent = 100 * row["changed_answers"] / row["answers"]
        right.barh(y + .13, percent, height=.2, color=LAYA)
        right.plot([0], [y - .13], "D", color=VEYA, markersize=5, clip_on=False)
        right.text(percent + 1, y + .13, "%.1f%%" % percent, va="center", fontsize=9.5, color=LAYA)
        right.text(1, y - .13, "0%", va="center", fontsize=9.5, color=VEYA)
    right.set_xlim(0, 39)
    right.set_ylim(-.5, 2.5)
    right.set_yticks([2, 1, 0], [r["label"] for r in facts["precision"]], fontsize=10)
    right.set_xticks([0, 10, 20, 30], ["0%", "10%", "20%", "30%"])
    right.set_xlabel("Answer labels changed", fontsize=9.5, labelpad=10)
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.tick_params(axis="both", length=0, labelcolor=MUTED, labelsize=9)
        for side in ax.spines.values():
            side.set_visible(False)

    text(fig, .04, .217, "Windows i9-13900KF · eager FP32 · 2 threads.\n15 warm inference samples; 50 primed memory-cache hits.",
         9, MUTED, linespacing=1.6)
    text(fig, .575, .217, "M4 Max measurements: 841 requests; ONNX INT8: 306.\nReference payloads replayed exactly in each tested setup.",
         9, MUTED, linespacing=1.6)
    text(fig, .04, .109, "Jev has no matching cache or precision measurements in these artifacts; it is not plotted as zero.", 9.6, MUTED)
    text(fig, .04, .056, "A cache hit preserves a stored answer. New inference can still vary, and a stored answer can still be wrong.", 9.6, MUTED)
    save(fig, "veya_inference_and_replay.png")


def eviction_chart(facts):
    fig = plt.figure(figsize=(12, 7.3), facecolor=PAPER)
    text(fig, .04, .95, "VEYA  /  BOUNDED MEMORY CACHE", 10, MUTED, "bold")
    text(fig, .04, .875, "Keep useful decisions. Avoid more inference.", 27, weight="bold")
    text(fig, .04, .827, "Implemented: opt-in LRU. FIFO remains the default; expiry and first-writer rules stay intact.", 11, MUTED)
    colors = (LAYA, VEYA, "#8994a7")
    fig.legend([Line2D([], [], color=color, linewidth=6) for color in colors],
               ["FIFO (default)", "LRU (implemented)", "LFU (research candidate)"],
               loc="center left", bbox_to_anchor=(.03, .765), ncol=3, frameon=False,
               fontsize=10, labelcolor=INK, columnspacing=1.8)
    text(fig, .04, .69, "Seeded traffic: cache hit rate", 13, weight="bold")
    text(fig, .61, .69, "Real checkpoint: forward passes", 13, weight="bold")
    left = fig.add_axes([.08, .335, .43, .31], facecolor=PAPER)
    labels = {"zipf": "Skewed\npopularity", "hot_plus_scan": "Hot set +\none-off traffic",
              "phase_shift": "Popularity\nshift", "all_unique": "All\nunique"}
    cases = facts["eviction"]
    for offset, (policy, color) in enumerate(zip(("fifo", "lru", "lfu"), colors)):
        xs = [i + (offset - 1) * .24 for i in range(len(cases))]
        values = [100 * case["policies"][policy]["hit_rate"] for case in cases]
        left.bar(xs, values, width=.21, color=color)
        for x, value in zip(xs, values):
            left.text(x, value + 2, "%.1f" % value, ha="center", fontsize=8, color=INK)
    left.set_xticks(range(len(cases)), [labels[c["trace"]] for c in cases], fontsize=9)
    left.set_ylim(0, 105)
    left.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    left.grid(axis="y", color=GRID)
    right = fig.add_axes([.675, .335, .24, .31], facecolor=PAPER)
    workflow = facts["workflow"]
    runs = workflow["policies"]
    forwards = [runs[p]["forward_passes_per_trial"] for p in ("fifo", "lru")]
    right.bar([0, 1], forwards, color=colors[:2], width=.55)
    for x, value in enumerate(forwards):
        right.text(x, value + 2, str(value), ha="center", fontsize=17, weight="bold")
    right.set_xticks([0, 1], ["FIFO", "LRU"])
    right.set_ylim(0, max(forwards) * 1.3)
    right.set_ylabel("Forward passes / %d requests" % workflow["parameters"]["requests"], fontsize=9)
    right.grid(axis="y", color=GRID)
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelcolor=MUTED, labelsize=9)
        for side in ax.spines.values():
            side.set_visible(False)
    text(fig, .04, .22, "60,000 requests · 1,024 entries · 7 timing repeats.\nLFU helps stable popularity but loses hits after the shift.",
         9.5, MUTED, linespacing=1.5)
    fifo, lru = (runs[p]["median_wall_seconds"] for p in ("fifo", "lru"))
    saved = 100 * (1 - forwards[1] / forwards[0])
    text(fig, .61, .22, "%.1f%% fewer forwards; median %.2f → %.2f seconds.\n3 alternating trials; all answer payloads unchanged." % (saved, fifo, lru),
         9.5, MUTED, linespacing=1.5)
    text(fig, .04, .105, "Left: synthetic traces, no model. Right: real multilingual weights, 16 entries, two CPU threads, warmed model.", 9.5, MUTED)
    text(fig, .04, .055, "Benefits depend on request reuse. Unique traffic gains no hits. Timings are from one workstation, not a universal speedup.", 9.5, MUTED)
    save(fig, "veya_cache_eviction.png")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": MUTED, "savefig.facecolor": PAPER})
    manifest = recorded_facts()
    overview(manifest["facts"])
    replay_chart(manifest["facts"])
    eviction_chart(manifest["facts"])
    (RESULTS / "readme_comparison_sources_20261107.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS: request equality, repeat completeness, choice denominators and replay counts")


if __name__ == "__main__":
    main()
