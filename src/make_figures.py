"""Figures in docs/, drawn from the result files (no numbers typed in by hand).

usage: python src/make_figures.py

fig1_drift.png      pedalboard / Rubber Band drift at Beat This!'s settings: measured on
                    click trains vs predicted from the engine's own hop log
fig2_magnitude.png  genre-probe accuracy vs shift size per tool (FMA, PANNs and AST)
fig3_truevar.png    NSynth: real +-2 semitone recordings vs tool shifts (probe accuracy)
fig4_piano.png      Kong onset-F1 loss per tool, split into timing and content (30 pieces)
fig5_training.png   Beat This! retraining: per-seed beat-F difference to the drifted labels,
                    two training scales x two test sets
Palette: reference categorical slots 1-4 (validated light mode); text in ink tokens,
identity never by colour alone (direct labels or legends).
"""

import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{ROOT}/results"
OUT = f"{ROOT}/docs"
os.makedirs(OUT, exist_ok=True)

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})  # fmt: skip


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---- fig 1: drift, measured vs predicted from the engine's hop log
def fig1():
    from rb_mechanism import predicted_curve

    cur = dict(np.load(f"{RESULTS}/bt_lagcurves.npz"))
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, (8, -8, 12, 16)):
        t, lag, *_ = predicted_curve(44100, 1 / (1 + s / 100), 1.0, 240)
        ax.plot(
            cur[f"ts{s}_t"],
            cur[f"ts{s}_lag"],
            color=C["blue"],
            label="measured (click train)",
        )
        ax.plot(
            t,
            lag,
            color=C["orange"],
            linestyle=(0, (4, 3)),
            label="predicted from hop log",
        )
        ax.axhspan(-70, 70, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(f"tempo {s:+d}%", loc="left")
        ax.set_xlim(0, 230)
    for ax in axes[:, 0]:
        ax.set_ylabel("audio vs label (ms)")
    for ax in axes[1]:
        ax.set_xlabel("time in stretched file (s)")
    axes[0, 0].legend(loc="lower right", fontsize=9)
    axes[0, 0].text(
        225, 62, "beat F tolerance ±70 ms", ha="right", va="top", color=INK2, fontsize=8
    )
    fig.suptitle(
        "pedalboard.time_stretch at 44.1 kHz: content drifts away from the scaled labels",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    save(fig, "fig1_drift.png")


# ---- fig 2: accuracy vs shift size (FMA genre probe)
def fig2():
    rows = json.load(open(f"{R}/timbre2.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    tools = (
        ("vocoder", "phase vocoder (librosa)", C["blue"]),
        ("signalsmith", "signalsmith", C["orange"]),
        ("resample", "resample", C["aqua"]),
    )
    for ax, model in zip(axes, ("panns", "ast")):
        clean = np.mean([r[f"{model}|clean"]["correct"] for r in rows])
        xs = ["pitch10", "pitch50", "pitch200"]
        for key, lab, col in tools:
            ys = [
                np.mean([r[f"{model}|{x}_{key}"]["correct"] for r in rows]) for x in xs
            ]
            ax.plot([10, 50, 200], ys, color=col, marker="o", ms=6, label=lab)
            ax.annotate(
                lab,
                (200, ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK2,
            )
        ax.axhline(clean, color=MUTED, lw=1, linestyle=":")
        ax.text(10, clean + 0.004, "no shift", color=INK2, fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks([10, 50, 200], ["10", "50", "200"])
        ax.set_xlabel("pitch shift (cents)")
        ax.set_title(model.upper(), loc="left")
        ax.set_xlim(8, 900)
    axes[0].set_ylabel("genre accuracy (400 FMA tracks)")
    fig.suptitle(
        "The vocoder's loss is already there at 10 cents and barely grows with size",
        x=0.01,
        y=1.04,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    save(fig, "fig2_magnitude.png")


# ---- fig 3: true variation (NSynth)
def fig3():
    rows = [json.loads(l) for l in open(f"{R}/truevar.jsonl")]
    vers = [("real", "real ±2 semitone recording"), ("signalsmith", "signalsmith"), ("resample", "resample"),
            ("rubberband", "Rubber Band"), ("librosa", "librosa"), ("torchaudio", "torchaudio")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, (m, col) in enumerate((("panns", C["blue"]), ("ast", C["orange"]))):
        acc = [np.mean([r[v]["pred"][m] == r["family"] for r in rows]) for v, _ in vers]
        y = np.arange(len(vers)) + (i - 0.5) * 0.22
        ax.scatter(
            acc,
            y,
            color=col,
            s=46,
            zorder=3,
            label=m.upper(),
            edgecolor=SURF,
            linewidth=2,
        )
    ax.set_yticks(range(len(vers)), [l for _, l in vers])
    ax.invert_yaxis()
    ax.set_xlabel("instrument-family probe accuracy (800 NSynth items)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        "A real pitch change costs nothing; most tools' simulation of it does",
        loc="left",
    )
    save(fig, "fig3_truevar.png")


# ---- fig 4: piano loss split into timing and content
def fig4():
    txt = open(f"{R}/summary_piano_all_rescore.txt").read()
    v = {
        m.group(1): (float(m.group(2)), float(m.group(3)))
        for m in re.finditer(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+[+-]", txt, re.M)
    }
    clean = v["clean"][0]
    tools = [("signalsmith", "signalsmith"), ("rbsmall", "Rubber Band, small blocks"), ("rubberband", "Rubber Band via pedalboard"),
             ("torchaudio", "torchaudio"), ("librosa", "librosa")]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, cond in zip(axes, ("pitch10", "tempo10")):
        for i, (k, _) in enumerate(tools):
            raw, aligned = v[f"{cond}_{k}"]
            timing, content = max(aligned - raw, 0), max(clean - aligned, 0)
            ax.barh(
                i,
                content,
                color=C["blue"],
                height=0.6,
                label="content" if i == 0 else None,
            )
            ax.barh(
                i,
                timing,
                left=content + 0.004,
                color=C["orange"],
                height=0.6,
                label="timing" if i == 0 else None,
            )
            ax.text(
                content + timing + 0.012,
                i,
                f"{clean - raw:.2f}",
                va="center",
                fontsize=8,
                color=INK2,
            )
        ax.set_yticks(range(len(tools)), [l for _, l in tools])
        ax.invert_yaxis()
        ax.set_title(
            {"pitch10": "pitch ±10 cents", "tempo10": "tempo ±10%"}[cond], loc="left"
        )
        ax.set_xlabel("onset-F1 lost vs unprocessed audio")
        ax.set_xlim(0, 0.72)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Kong piano transcription, 30 MAESTRO pieces: what time alignment recovers (timing) and what it cannot (content)",
        x=0.01,
        y=1.06,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    save(fig, "fig4_piano.png")


# ---- fig 5: retraining, per-seed differences, both scales and both test sets
def fig5():
    def load(f):
        by = {}
        for l in open(f"{R}/{f}"):
            r = json.loads(l)
            c, s_, _ = r["name"].split("_")
            by.setdefault(c, {})[int(s_[1:])] = r
        return by

    panels = (("bt_eval.jsonl", "4 training sets, 5 seeds — GTZAN"), ("bt_eval_val.jsonl", "4 training sets — validation"),
              ("bt_eval_big.jsonl", "11 training sets, 3 seeds — GTZAN"), ("bt_eval_big_val.jsonl", "11 training sets — validation"))  # fmt: skip
    conds = (
        ("fixed", "labels corrected", C["blue"]),
        ("spectempo", "spectrogram-domain tempo", C["aqua"]),
        ("notempo", "no tempo augmentation", C["orange"]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 4.6), sharex=True)
    for ax, (f, title) in zip(axes.flat, panels):
        by = load(f)
        # each condition keeps its row in every panel; spectrogram-domain tempo ran only at 11 sets
        for i, (c, _, col) in enumerate(conds):
            if c not in by:
                ax.text(0.0005, i, "not run", va="center", fontsize=8, color=MUTED)
                continue
            d = [
                by[c][s_]["beat_f70"] - by["asis"][s_]["beat_f70"]
                for s_ in sorted(by["asis"])
                if s_ in by[c]
            ]
            ax.scatter(
                d, [i] * len(d), color=col, s=40, zorder=3, edgecolor=SURF, linewidth=2
            )
            ax.plot([np.mean(d)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.set_yticks(range(len(conds)), [n for _, n, _ in conds], fontsize=9)
        ax.set_ylim(len(conds) - 0.5, -0.5)
        ax.set_xlim(-0.014, 0.042)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(title, loc="left", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("beat F (70 ms) minus training on the drifted labels")
    fig.suptitle(
        "Beat This! (small) retrained: drifted tempo augmentation is below every alternative (dots = seeds, bar = mean)",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "fig5_training.png")


if __name__ == "__main__":
    for f in (fig2, fig3, fig4, fig5, fig1):
        f()
