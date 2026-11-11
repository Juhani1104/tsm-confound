"""Split each tool's transcription loss into timing and content: rescore after time alignment.

usage: python src/piano_rescore.py [NOTES.jsonl]   (default results/piano_rb_notes.jsonl)

For every piece and condition, Kong's notes are matched to the reference notes of the
same pitch within 150 ms, and a straight line t_ref = a + b * t_est (offset a, drift b) is
fitted to the matched pairs with a Theil-Sen estimator. The notes are moved by that line and
scored again (onset F1, 50 ms, pitch exact after the whole-semitone label shift). What the
alignment recovers is loss from timing (constant delay or drift); what remains is loss from
the content (missed, extra or wrong notes). The clean condition is aligned the same way as
a control for how much the fit itself adds.
"""

import json
import os
import sys

import librosa
import mir_eval
import numpy as np
import pretty_midi
from scipy.stats import theilslopes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import MAESTRO, MAESTRO_PICK, RESULTS  # noqa: E402

SEC = 60
path = sys.argv[1] if len(sys.argv) > 1 else f"{RESULTS}/piano_rb_notes.jsonl"
_all = json.load(open(MAESTRO_PICK))
pick = _all[::3] + [
    r for i, r in enumerate(_all) if i % 3
]  # same order as task_piano.py
rng = np.random.default_rng(0)


def f1(ref_on, ref_pi, on, pi):
    """Onset F1 (50 ms, exact pitch) of estimated notes against reference notes."""
    ok = on < ref_on.max() + 1 if len(on) else on.astype(bool)
    if not ok.any():
        return 0.0
    return mir_eval.transcription.precision_recall_f1_overlap(
        np.c_[ref_on, ref_on + 0.1], librosa.midi_to_hz(ref_pi),
        np.c_[on[ok], on[ok] + 0.1], librosa.midi_to_hz(pi[ok]), offset_ratio=None, onset_tolerance=0.05,
    )[2]  # fmt: skip


def align(ref_on, ref_pi, on, pi, win=0.15):
    """(a, b) of t_ref = a + b * t_est from same-pitch matches within win seconds."""
    xs, ys = [], []
    for t, p in zip(on, pi):
        c = ref_on[ref_pi == p]
        if len(c):
            j = np.argmin(np.abs(c - t))
            if abs(c[j] - t) < win:
                xs.append(t)
                ys.append(c[j])
    if len(xs) < 10:
        return 0.0, 1.0
    xs, ys = np.array(xs), np.array(ys)
    b = theilslopes(ys, xs)[0]
    # intercept as the median residual; theilslopes' own median(y) - b * median(x) can be off by
    # a whole inter-onset gap when the matched pairs are not in the same order
    return float(np.median(ys - b * xs)), b


rows = [json.loads(l) for l in open(path)]
res = {}
for row in rows:
    r = pick[row["piece"]]
    n = sorted(
        pretty_midi.PrettyMIDI(f"{MAESTRO}/{r['midi_filename']}").instruments[0].notes,
        key=lambda v: v.start,
    )
    on0 = np.array([v.start for v in n])
    pi0 = np.array([v.pitch for v in n])
    keep = on0 < SEC - 1
    for cond, d in row.items():
        if cond == "piece" or cond.startswith("tempo") and cond.endswith("_resample"):
            continue
        ref_on, ref_pi = on0[keep] * d["scale"], pi0[keep] + round(d["cents"] / 100)
        e = np.array(d["notes"]).reshape(-1, 2)
        on, pi = e[:, 0], e[:, 1].astype(int)
        a, b = align(ref_on, ref_pi, on, pi)
        res.setdefault(cond, []).append(
            (
                f1(ref_on, ref_pi, on, pi),
                f1(ref_on, ref_pi, np.maximum(a + b * on, 0), pi),
                a * 1000,
                (b - 1) * 1000,
            )
        )

print(f"{path}  pieces {len(rows)}")
print(
    f"{'condition':24s} {'raw':>6s} {'aligned':>8s} {'gain':>6s}   offset ms   drift ms/s"
)
for cond in sorted(res, key=lambda c: (c != "clean", c)):
    v = np.array(res[cond])
    print(
        f"{cond:24s} {v[:, 0].mean():6.3f} {v[:, 1].mean():8.3f} {v[:, 1].mean() - v[:, 0].mean():+6.3f}   {np.median(v[:, 2]):+7.1f}   {np.median(v[:, 3]):+6.2f}"
    )
