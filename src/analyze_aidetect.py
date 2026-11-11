"""Summarise results/aidetect.jsonl per detector and condition.

usage: python src/analyze_aidetect.py [RESULTS.jsonl]

For each detector, condition by condition: share of real clips flagged as AI (false positive
rate at the 0.5 threshold), share of EnCodec clips flagged (detection rate), and the AUC
between real and fake scores. Deezer outputs P(real), so 'flagged' is P(real) < 0.5;
ArtifactNet outputs P(AI), flagged is P(AI) > 0.5.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else f"{RESULTS}/aidetect.jsonl"
rows = [json.loads(l) for l in open(path)]
conds = [k for k in rows[0] if k != "track"]


def auc(pos, neg):
    """P(score of a fake > score of a real), ties count half; scores oriented so higher = AI."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


for det, ai in (("deezer", lambda v: 1 - v), ("anet", lambda v: v)):
    print(f"\n{det}  ({len(rows)} tracks)   real flagged | fake flagged | AUC")
    for c in conds:
        r = np.array(
            [ai(row[c]["real"][det]) for row in rows if row[c]["real"][det] is not None]
        )
        f = np.array(
            [ai(row[c]["fake"][det]) for row in rows if row[c]["fake"][det] is not None]
        )
        print(
            f"{c:22s} {np.mean(r > 0.5):6.2f} | {np.mean(f > 0.5):6.2f} | {auc(f, r):.3f}"
        )
