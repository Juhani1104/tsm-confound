"""Summarise results/truevar.jsonl: is a tool shift as easy to classify as a real pitch change?

usage: python src/analyze_truevar.py [RESULTS.jsonl]

Per version (real target, unshifted source, each tool's shift): probe accuracy per model,
mean embedding cosine distance to the real target, and mean log-mel distance to it.
Accuracy differences are paired against the real target with a 95% bootstrap interval over
pairs (both directions of one pair are resampled together). Then the same per instrument
source (acoustic / electronic / synthetic).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else f"{RESULTS}/truevar.jsonl"
rows = [json.loads(l) for l in open(path)]
VERS = [
    v
    for v in (
        "real",
        "orig",
        "librosa",
        "torchaudio",
        "signalsmith",
        "rubberband",
        "rbsmall",
        "resample",
    )
    if v in rows[0]
]
MODELS = list(rows[0]["real"]["pred"])
rng = np.random.default_rng(0)


def table(rs, title):
    """Print accuracy, embedding distance and log-mel distance per version for rows rs."""
    pair = np.array([min(r["src"], r["tgt"]) for r in rs])
    ids = np.unique(pair)
    idx = [np.flatnonzero(pair == i) for i in ids]
    print(f"\n{title}  items {len(rs)}  pairs {len(ids)}")
    print(
        f"{'version':12s} "
        + " ".join(f"{'acc ' + m:>22s}" for m in MODELS)
        + " "
        + " ".join(f"{'cos ' + m:>10s}" for m in MODELS)
        + f" {'mel dB':>7s}"
    )
    for v in VERS:
        cells = []
        for m in MODELS:
            ok = np.array([r[v]["pred"][m] == r["family"] for r in rs], float)
            ref = np.array([r["real"]["pred"][m] == r["family"] for r in rs], float)
            d = ok - ref
            b = [
                np.concatenate(
                    [d[idx[j]] for j in rng.integers(len(idx), size=len(idx))]
                ).mean()
                for _ in range(2000)
            ]
            cells.append(
                f"{ok.mean():.3f} ({d.mean():+.3f} [{np.quantile(b, 0.025):+.3f},{np.quantile(b, 0.975):+.3f}])"
                if v != "real"
                else f"{ok.mean():.3f}{'':16s}"
            )
        cos = [np.mean([r[v]["cos"][m] for r in rs]) for m in MODELS]
        mel = np.mean([r[v]["mel"] for r in rs])
        print(
            f"{v:12s} "
            + " ".join(f"{c:>22s}" for c in cells)
            + " "
            + " ".join(f"{c:10.4f}" for c in cos)
            + f" {mel:7.2f}"
        )


table(rows, path)
for s in sorted({r["source"] for r in rows}):
    table([r for r in rows if r["source"] == s], f"source {s}")
