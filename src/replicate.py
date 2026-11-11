"""Replicate Edwards et al.'s pitch-shift test protocol on two datasets, with three shifters.

usage: python src/replicate.py > results/replicate.txt

Edwards et al. (arXiv 2402.01424, Table V) report Kong's onset F1 under a random
+-10 cent pitch shift dropping 96.8 -> 85.2 (MAESTRO) and 95.7 -> 76.5 (Studio
MAESTRO). Here each excerpt gets one uniform random shift in [-10, 10] cents, as their
audiomentations PitchShift draws it, made three ways: librosa phase vocoder (the
audiomentations default before 0.39), signalsmith-stretch (the default since 0.39),
and a pure resample that changes no timbre. If only the vocoder reproduces both
reported drops, the reported sensitivity is the vocoder's.
"""

import json
import os
import sys

import librosa
import mir_eval
import numpy as np
import pretty_midi
import python_stretch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import MAESTRO, MAESTRO_PICK, STUDIO_LAGS, STUDIO_MAESTRO, STUDIO_PICK  # noqa: E402
from piano_models import load  # noqa: E402

SR = 16000
DUR = 90
kong = load("kong")
pick = json.load(open(MAESTRO_PICK))
studio = json.load(open(STUDIO_PICK))
# Studio MAESTRO audio starts slightly off the MIDI; each recording's offset (s) was found by
# aligning Kong's clean transcription to the MIDI (index/studio_lags.json)
lags = {int(k): v for k, v in json.load(open(STUDIO_LAGS)).items()}
rng = np.random.default_rng(0)


def shift(x, how, c):
    """x shifted by c cents; also the factor for GT onset times."""
    if how == "vocoder":
        return librosa.effects.pitch_shift(
            x, sr=SR, n_steps=c / 100, res_type="soxr_hq"
        ), 1.0
    if how == "signalsmith":
        s = python_stretch.Signalsmith.Stretch()
        s.preset(1, SR)
        s.setTransposeSemitones(c / 100)
        return s.process(x[None])[0].astype("float32"), 1.0
    r = 2 ** (c / 1200)
    return librosa.resample(x, orig_sr=SR * r, target_sr=SR, res_type="soxr_hq"), 1 / r


def f1(x, ref, t, lag):
    """Onset F1 (%) of Kong on x against ref onsets scaled by t, predictions moved by lag."""
    e = kong.transcribe(x.astype("float32"), None)["est_note_events"]
    on = np.array([n["onset_time"] for n in e]) - lag
    pi = np.array([n["midi_note"] for n in e])
    ok = (on >= 0) & (on < DUR - 1)
    ro, rp = ref[0] * t, ref[1]
    keep = ro < DUR - 1
    return mir_eval.transcription.precision_recall_f1_overlap(
        np.c_[ro[keep], ro[keep] + 0.1], librosa.midi_to_hz(rp[keep]),
        np.c_[on[ok], on[ok] + 0.1], librosa.midi_to_hz(pi[ok]),
        offset_ratio=None, onset_tolerance=0.05,
    )[2] * 100  # fmt: skip


res = {}
for k, r in enumerate(pick):
    n = sorted(
        pretty_midi.PrettyMIDI(f"{MAESTRO}/{r['midi_filename']}").instruments[0].notes,
        key=lambda v: v.start,
    )
    ref = (np.array([v.start for v in n]), np.array([v.pitch for v in n]))
    c = float(rng.uniform(-10, 10))
    for ds, path, lag in (
        ("maestro", f"{MAESTRO}/{r['audio_filename']}", 0.0),
        ("studio", f"{STUDIO_MAESTRO}/{os.path.basename(studio[k])}", lags[k]),
    ):
        x, _ = librosa.load(path, sr=SR, mono=True, duration=DUR + 3)
        row = {"clean": f1(x, ref, 1.0, lag)}
        for how in ("vocoder", "signalsmith", "resample"):
            y, t = shift(x, how, c)
            row[how] = f1(y, ref, t, lag * t)
        res.setdefault(ds, []).append(row)
        print(
            ds,
            k,
            r["year"],
            f"shift {c:+.1f}",
            {h: round(v, 2) for h, v in row.items()},
            flush=True,
        )

print(
    "\nreported by Edwards et al.: maestro 96.8 -> 85.2 (-11.6), studio 95.7 -> 76.5 (-19.2)"
)
for ds, rows in res.items():
    m = {h: np.mean([r[h] for r in rows]) for h in rows[0]}
    print(
        ds,
        "n",
        len(rows),
        " ".join(f"{h} {v:.2f} ({v - m['clean']:+.2f})" for h, v in m.items()),
    )
