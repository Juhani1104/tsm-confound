"""Summarise any task's jsonl: mean score per condition and paired differences to signalsmith.

usage: python src/analyze.py RESULTS.jsonl METRIC [LABEL_FIELD]

METRIC picks the score inside each condition entry:
  a key of a dict entry, e.g. note_f1 or beat_f            (task_guitar)
  'value' when the entry itself is a number                 (task_piano)
  'model:NAME' when the entry maps model -> predicted label, scored as correct against
  LABEL_FIELD of the row                                    (task_nsynth_probe)
Differences to signalsmith at the same kind and magnitude are paired per item, with a 95%
bootstrap interval over items.
"""

import json
import sys

import numpy as np

path, metric = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else None
rows = [json.loads(l) for l in open(path)]
conds = [
    k
    for k, v in rows[0].items()
    if k == "clean" or "_" in k and isinstance(v, (dict, float, int)) and k[0] in "pt"
]
rng = np.random.default_rng(0)


def score(row, c):
    """The item's score under condition c, as METRIC defines it."""
    v = row[c]
    if metric == "value":
        return float(v)
    if metric.startswith("model:"):
        return float(v[metric[6:]] == row[label])
    return np.nan if v.get(metric) is None else float(v[metric])


def ci(d):
    """Mean of the paired differences d with a 95% bootstrap interval over items."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    b = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(4000)]
    return (
        f"{d.mean():+.3f} [{np.quantile(b, 0.025):+.3f},{np.quantile(b, 0.975):+.3f}]"
    )


S = {c: np.array([score(r, c) for r in rows]) for c in conds}
print(f"{path}  metric {metric}  items {len(rows)}")
print(f"clean {np.nanmean(S['clean']):.3f}")
groups = sorted(
    {c.rsplit("_", 1)[0] for c in conds if c != "clean"},
    key=lambda g: (g[0], int("".join(ch for ch in g if ch.isdigit()))),
)
for g in groups:
    ref = S.get(f"{g}_signalsmith")
    parts = []
    for tool in ("librosa", "torchaudio", "rubberband", "rbsmall", "resample"):
        c = f"{g}_{tool}"
        if c in S:
            parts.append(
                f"{tool} {np.nanmean(S[c]):.3f} ({ci(S[c] - ref) if ref is not None else ''})"
            )
    print(f"{g:9s} signalsmith {np.nanmean(ref):.3f} | " + " | ".join(parts))
