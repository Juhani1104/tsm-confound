"""Paired comparison of the Beat This! training conditions over seeds and GTZAN items.

usage: python src/bt_compare.py [EVAL.jsonl]   (default results/bt_eval.jsonl; bt_eval_val.jsonl for the val split)

From results/bt_eval.jsonl: for each condition, mean over seeds of every metric; then for
calibrated beat and downbeat F20, the per-item score averaged over the seeds a condition
has, and the paired difference fixed - asis and notempo - asis with a 95% bootstrap
interval over the 993 GTZAN items. Seeds are also compared one by one (sign of the
per-seed difference), since seed-to-seed variation is the main uncertainty.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS  # noqa: E402

rows = [
    json.loads(l)
    for l in open(sys.argv[1] if len(sys.argv) > 1 else f"{RESULTS}/bt_eval.jsonl")
]
by = defaultdict(dict)
for r in rows:
    cond, seed, _ = r["name"].split("_")
    by[cond][int(seed[1:])] = r
metrics = [
    "beat_f70",
    "beat_f30",
    "beat_f20",
    "beat_f30_cal",
    "beat_f20_cal",
    "beat_bias_ms",
    "downbeat_f70",
    "downbeat_f20_cal",
]
print(f"{'cond':8s} {'seeds':6s} " + " ".join(f"{m:>16s}" for m in metrics))
for c in ("asis", "fixed", "notempo", "spectempo"):
    s = by.get(c, {})
    if s:
        print(
            f"{c:8s} {len(s):<6d} "
            + " ".join(f"{np.mean([v[m] for v in s.values()]):16.3f}" for m in metrics)
        )

rng = np.random.default_rng(0)
for k in ("beat", "downbeat"):
    key = f"{k}_per_item_f20_cal"
    common = sorted(
        set(by["asis"]) & set(by.get("fixed", {})) & set(by.get("notempo", {}))
    )  # spectempo: compared on its own seeds below
    if not common:
        break
    A = np.mean([by["asis"][s][key] for s in common], 0)
    for c in [c for c in ("fixed", "notempo", "spectempo") if c in by]:
        B = np.mean([by[c][s][key] for s in common], 0)
        d = B - A
        bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(4000)]
        per_seed = [
            np.mean(by[c][s][key]) - np.mean(by["asis"][s][key]) for s in common
        ]
        print(
            f"{k} F20_cal {c} - asis over seeds {common}: {d.mean():+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]  per seed "
            + " ".join(f"{v:+.3f}" for v in per_seed)
        )
