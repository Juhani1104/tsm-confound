"""Summarise results/nsynth.jsonl: zero-shot instrument-family accuracy per condition and model.

usage: python src/analyze_nsynth.py > results/summary_nsynth.txt
Paired differences against signalsmith use a bootstrap over notes.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [json.loads(l) for l in open(f"{ROOT}/results/nsynth.jsonl")]
rng = np.random.default_rng(0)
conds = [k for k in rows[0] if k not in ("note", "family", "pitch")]


def ci(d):
    """Mean of d with a 95% bootstrap interval over notes."""
    d = np.asarray(d, float)
    b = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(4000)]
    return (
        f"{d.mean():+.3f} [{np.quantile(b, 0.025):+.3f},{np.quantile(b, 0.975):+.3f}]"
    )


print("notes", len(rows), "families", len({r["family"] for r in rows}))
for m in ("panns", "ast"):
    acc = {c: np.array([r[c][m] == r["family"] for r in rows]) for c in conds}
    print(f"\n{m}: clean accuracy {acc['clean'].mean():.3f}")
    for c in conds[1:]:
        if c.endswith("_signalsmith") or c == "clean":
            continue
        ref = c.rsplit("_", 1)[0] + "_signalsmith"
        print(
            f"  {c:22s} acc {acc[c].mean():.3f}   minus signalsmith {ci(acc[c].astype(float) - acc[ref].astype(float))}"
        )
    for c in conds[1:]:
        if c.endswith("_signalsmith"):
            print(f"  {c:22s} acc {acc[c].mean():.3f}")
