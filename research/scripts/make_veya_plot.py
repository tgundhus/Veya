"""Render the Veya against Laya against TypeSafe Jev figure used in the README.

    python research/scripts/make_veya_plot.py

Laya and Veya figures are read from the committed results in research/results/ (Apple M4 Max
runs on the published checkpoints, *_m4max_20260928.json). Jev figures are third-party published
(AbdelStark/jev-benchmarks, nibzard/decision-model-benchmark), except the repeat test, which reads
the archived raw responses in research/benchmarks/feishu_zh; they are constants below, as in
make_plots.py. Writes assets/veya_vs_laya_vs_jev.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

RESEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(RESEARCH)
RESULTS = os.path.join(RESEARCH, "results")
ASSETS = os.path.join(REPO, "assets")

# Validated categorical slots 1 and 3 (light mode) for Laya and Veya; Jev is the de-emphasised
# comparator in neutral gray, as in laya_vs_jev_full.png. Aqua and gray sit below 3:1 on the
# surface, so every mark carries a visible value label.
BLUE, AQUA, GRAY = "#2a78d6", "#1baf7a", "#9a9a92"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0,
})

# Jev: third-party published, never run here, except `JEV_REPEAT`, recomputed by tests/test_consistency.py
# from the archived Feishu responses (29 of 128 identical across three runs).
JEV_REPEAT = (29, 128)
JEV_P50_MS = (236, 276)
ACCURACY = [  # (task, Jev 1.13.0 published, Laya routed; Veya gives Laya's answers)
    ("typed-decisions\n2,000 decisions", 0.727, 0.766),
    ("AG News\n4 labels", 0.910, 0.950),
    ("DAIR Emotion\n6 labels", 0.480, 0.595),
]


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def strip(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def title(ax, text, sub=None, lines=None):
    """A bold title with an optional muted subtitle between it and the plot; `lines` reserves room."""
    lines = lines if lines is not None else (sub.count("\n") + 1 if sub else 0)
    ax.set_title(text, fontsize=13, fontweight="bold", color=INK, loc="left", pad=14 + 14 * lines)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK3, va="bottom", linespacing=1.35)


def bar_labels(ax, bars, fmt, pad):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt(h), ha="center", va="bottom", fontsize=8.5, color=INK2)


def facts():
    cons = load("backend_consistency_m4max_20260928.json")["models"]
    setups = {}
    for model in cons.values():
        for name, c in model["vs_cpu_fp32"].items():
            s = setups.setdefault(name, [0, 0])
            s[0] += c["changed_decisions"]
            s[1] += c["answers"]
    replayed = sum(r["identical_to_first_decision"] for m in cons.values()
                   for r in m["decision_cache_replay"]["setups"].values())
    replay_total = sum(r["requests"] for m in cons.values() for r in m["decision_cache_replay"]["setups"].values())
    stages = {dev: load("stages_%s_m4max_20260928.json" % dev)["scenarios"] for dev in ("mps", "cpu")}
    ticket = "english ticket, 3 questions"
    retention = load("retention_m4max_20260928.json")["simulate"]["runs"]
    trace = load("cache_serving_trace_mps_m4max_20260928.json")["trace"]
    return {
        "setups": setups, "replayed": (replayed, replay_total),
        "computed_ms": {d: stages[d][ticket]["latency_ms"]["median"] for d in stages},
        "hit_ms": {d: stages[d][ticket + ", DecisionCache hit"]["latency_ms"]["median"] for d in stages},
        "retention": retention,
        "rps": (trace["no cache (default path)"]["requests_per_s"], trace["memory DecisionCache"]["requests_per_s"]),
        "trace": trace,
    }


def main():
    f = facts()
    fig = plt.figure(figsize=(16, 13.2))
    gs = GridSpec(3, 3, figure=fig, left=0.095, right=0.975, top=0.80, bottom=0.05,
                  hspace=0.72, wspace=0.34, width_ratios=[1, 1, 1.08])

    fig.text(0.012, 0.975, "Veya   vs   Laya   vs   TypeSafe Jev", fontsize=26, fontweight="bold", color=INK,
             va="top")
    fig.text(0.012, 0.935, "Veya is Laya with a memory for its decisions: until a decision is replayed, its "
             "answers are Laya's, byte for byte. Laya and Veya are measured here on the published checkpoints "
             "(Apple M4 Max).", fontsize=10.5, color=INK2, va="top")
    fig.text(0.012, 0.915, "Jev figures are third-party published (AbdelStark/jev-benchmarks, "
             "nibzard/decision-model-benchmark), except the repeat test, which reads archived raw responses; Jev "
             "was never run in this project.", fontsize=10.5, color=INK2, va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GRAY, BLUE, AQUA)]
    fig.legend(handles, ["TypeSafe Jev 1.13.0", "Laya", "Veya"], loc="upper left", ncol=3, frameon=False,
               bbox_to_anchor=(0.008, 0.905), fontsize=11, labelcolor=INK2, handlelength=1.2, columnspacing=1.8)

    # A. Same request, same answer
    ax = fig.add_subplot(gs[0, 0:2])
    groups = ["the same request\nsent three times", "fp16 instead of fp32\n(a batch on the Apple GPU)",
              "int8 instead of fp32\n(PyTorch on the CPU)"]
    fp16, int8 = f["setups"]["mps-default-batched"], f["setups"]["cpu-int8"]
    jev = [100 * JEV_REPEAT[0] / JEV_REPEAT[1], None, None]
    laya = [100.0, 100 * (1 - fp16[0] / fp16[1]), 100 * (1 - int8[0] / int8[1])]
    pro = [100.0, 100.0, 100.0]
    w = 0.26
    for k, (vals, col) in enumerate(((jev, GRAY), (laya, BLUE), (pro, AQUA))):
        xs = [i + (k - 1) * w for i in range(len(groups))]
        for x, v in zip(xs, vals):
            if v is None:
                ax.text(x, 3, "not\npublished", ha="center", va="bottom", fontsize=8, color=INK3)
                continue
            ax.bar(x, v, w - 0.03, color=col)
            ax.text(x, v + 1.5, ("%.1f%%" % v).replace(".0%", "%"), ha="center", va="bottom", fontsize=9,
                    color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    title(ax, "Same request, same answer",
          "First: responses identical across three runs of 64 requests (128 responses).\nThen: decisions unchanged "
          "when the precision changes (%s answers on 841 requests).\nVeya replays the first decision: %s of %s "
          "replays identical across CPU, Apple GPU, ONNX and int8." % (format(fp16[1], ","),
                                                                      *(format(n, ",") for n in f["replayed"])))

    # B. Where Veya leads
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    title(ax, "Where Veya leads", lines=3)  # level with the panel beside it
    hit_lo, hit_hi = min(f["hit_ms"].values()), max(f["hit_ms"].values())
    low = min(f["computed_ms"]["mps"] / f["hit_ms"]["mps"], f["computed_ms"]["cpu"] / f["hit_ms"]["cpu"])
    renew7 = f["retention"]["memory, ttl 7 days, renew_on_hit"]["repeats_given_earlier_decision_share"]
    renew30 = f["retention"]["memory, ttl 30 days, renew_on_hit"]["repeats_given_earlier_decision_share"]
    rps_base, rps_pro = f["rps"]
    tiles = [
        ("%s×" % format(round(low, -1), ",.0f"), "faster on a repeat",
         "%.2f–%.2f ms vs %.0f–%.0f ms computed" % (hit_lo, hit_hi, f["computed_ms"]["mps"], f["computed_ms"]["cpu"])),
        ("0", "decisions moved by the setup", "vs up to %.0f%% (int8) without it" % (100 * int8[0] / int8[1])),
        ("%.0f%%" % (100 * renew7), "of repeats keep their decision",
         "7-day retention; %.1f%% at 30 days" % (100 * renew30)),
        ("%.1f×" % (rps_pro / rps_base), "requests served",
         "at 70%% repeats: %.1f vs %.1f a second" % (rps_pro, rps_base)),
    ]
    for i, (big, head, small) in enumerate(tiles):
        y = 1.02 - i * 0.24
        ax.text(0.0, y, big, fontsize=25, fontweight="bold", color=AQUA, va="center", transform=ax.transAxes)
        ax.text(0.37, y + 0.04, head, fontsize=10.5, fontweight="bold", color=INK, va="center", transform=ax.transAxes)
        ax.text(0.37, y - 0.045, small, fontsize=8.5, color=INK3, va="center", transform=ax.transAxes)
    ax.text(0.0, 0.0, "Over Jev, Laya and Veya alike: 7.8× faster,\n3× better calibrated, 45 of 51 languages "
            "usable,\n$0 self-hosted under Apache 2.0.", fontsize=9, color=INK2, transform=ax.transAxes, va="top",
            linespacing=1.4)

    # C. A repeated request, log scale
    ax = fig.add_subplot(gs[1, 0])
    rows = [("Jev, hosted API", None, GRAY), ("Laya, CPU", f["computed_ms"]["cpu"], BLUE),
            ("Laya, Apple GPU", f["computed_ms"]["mps"], BLUE), ("Veya, CPU", f["hit_ms"]["cpu"], AQUA),
            ("Veya, Apple GPU", f["hit_ms"]["mps"], AQUA)]
    for y, (name, v, col) in enumerate(reversed(rows)):
        if v is None:
            ax.plot(JEV_P50_MS, [y, y], color=col, linewidth=2, solid_capstyle="round")
            ax.plot(JEV_P50_MS, [y, y], "o", color=col, markersize=8)
            ax.text(JEV_P50_MS[0] / 1.35, y, "%d–%d ms" % JEV_P50_MS, ha="right", va="center", fontsize=9, color=INK2)
        else:
            ax.plot([v], [y], "o", color=col, markersize=9)
            ax.text(v * 1.45, y, ("%.3f ms" if v < 1 else "%.1f ms") % v, ha="left", va="center", fontsize=9,
                    color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 3000)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5)
    ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1,000 ms"], fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    title(ax, "A repeated request", "log scale · Laya and Veya: a 3-question ticket,\nApple M4 Max · Jev: "
          "published p50 for one question")

    # D. Accuracy where Jev numbers exist
    ax = fig.add_subplot(gs[1, 1])
    for k, col in enumerate((GRAY, BLUE, AQUA)):
        xs = [i + (k - 1) * w for i in range(len(ACCURACY))]
        vals = [a[1] if k == 0 else a[2] for a in ACCURACY]
        bars = ax.bar(xs, vals, w - 0.03, color=col)
        if k == 0:
            bar_labels(ax, bars, lambda h: "%.3f" % h, 0.015)
    for i, (_, _, v) in enumerate(ACCURACY):  # Laya and Veya answer alike: one label over both
        ax.text(i + w / 2, v + 0.015, "%.3f" % v, ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(ACCURACY)))
    ax.set_xticklabels([a[0] for a in ACCURACY], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    title(ax, "Accuracy where Jev numbers exist", "Veya gives Laya's answers, so its accuracy is Laya's\n"
          "Jev: third-party published")

    # E. Serving with repeats
    ax = fig.add_subplot(gs[1, 2])
    bars = ax.bar([0, 1], [rps_base, rps_pro], 0.5, color=[BLUE, AQUA])
    bar_labels(ax, bars, lambda h: "%.1f" % h, 1.0)
    ax.text(0.5, rps_pro * 0.62, "%.1f×" % (rps_pro / rps_base), ha="center", va="center", fontsize=16,
            fontweight="bold", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Laya", "Veya"], fontsize=10)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, rps_pro * 1.18)
    ax.set_ylabel("requests a second", fontsize=9.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    t = f["trace"]
    title(ax, "Serving a stream with 70% repeats", "%d requests, %d distinct, one after another\non the Apple GPU · "
          "Jev: not measured (hosted)" % (t["requests"], t["distinct_in_trace"]))

    # F. Decision memory by retention period
    ax = fig.add_subplot(gs[2, 0])
    ttls = [("1 hour", "1 h"), ("1 day", "1 day"), ("7 days", "7 days"), ("30 days", "30 days")]
    for renew, style, label in ((True, "-", "renewed on use"), (False, "--", "fixed")):
        ys = [100 * f["retention"]["memory, ttl %s%s" % (key, ", renew_on_hit" if renew else "")]
              ["repeats_given_earlier_decision_share"] for _, key in ttls]
        ax.plot(range(len(ttls)), ys, style, color=AQUA, linewidth=2, marker="o", markersize=7)
        ax.text(len(ttls) - 1 + 0.15, ys[-1] + (4 if renew else -5), "%s  %.1f%%" % (label, ys[-1]),
                va="center", fontsize=9, color=INK2)
    ax.text(0, 3, "Laya and Jev keep no decisions: every repeat is recomputed", fontsize=9, color=INK3,
            va="bottom")
    ax.set_xticks(range(len(ttls)))
    ax.set_xticklabels([t[0] for t in ttls], fontsize=9.5)
    ax.set_xlim(-0.3, len(ttls) + 0.7)
    ax.set_ylim(-3, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    tr = load("retention_m4max_20260928.json")["simulate"]["trace"]
    title(ax, "Decision memory by retention period", "Veya: repeats served their earlier decision · a simulated "
          "month,\n%s requests, %s distinct" % (format(tr["requests"], ","), format(tr["distinct"], ",")))

    # G. Decisions changed by the setup
    ax = fig.add_subplot(gs[2, 1:3])
    other = [f["setups"][n] for n in ("cpu-fp32-batched", "mps-fp32", "onnx-fp32")]
    rows = [("fp32 on the Apple GPU, ONNX, or in a batch", sum(o[0] for o in other), sum(o[1] for o in other)),
            ("fp16: a batch on the Apple GPU (Laya's default there)", *f["setups"]["mps-default-batched"]),
            ("int8, ONNX (English checkpoint)", *f["setups"]["onnx-int8"]),
            ("int8, PyTorch", *f["setups"]["cpu-int8"])]
    for y, (name, changed, total) in enumerate(reversed(rows)):
        pct = 100 * changed / total
        ax.text(0, y + 0.36, name, va="bottom", fontsize=9.5, color=INK)
        ax.barh(y + 0.16, max(pct, 0.08), 0.26, color=BLUE)
        ax.text(max(pct, 0.08) + 0.4, y + 0.16, "Laya %.1f%%  (%d of %s)" % (pct, changed, format(total, ",")),
                va="center", fontsize=9, color=INK2)
        ax.barh(y - 0.14, 0.08, 0.26, color=AQUA)
        ax.text(0.5, y - 0.14, "Veya 0", va="center", fontsize=9, color=INK2)
    ax.set_yticks([])
    ax.set_ylim(-0.45, len(rows) - 0.2)
    ax.set_xlim(0, 36)
    ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("% of decisions changed", fontsize=9.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    strip(ax, keep=())
    title(ax, "Decisions changed when the setup changes", "Against fp32 on the CPU, each request answered on its "
          "own · Veya: one DecisionCache in front of\nevery setup replays the first decision, so none change")

    out = os.path.join(ASSETS, "veya_vs_laya_vs_jev.png")
    fig.savefig(out, dpi=125)
    print("wrote", out)


if __name__ == "__main__":
    main()
