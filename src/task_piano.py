"""Piano transcription (Kong et al.) under every tool and condition, to test the onset prediction.

usage: python src/task_piano.py N SECONDS

Prediction from artifacts2.py: torchaudio's phase-vocoder pitch shift (n_fft 512) keeps
onsets, librosa's (n_fft 2048) smears them, while both damage the spectrum equally. If
onset smearing drives the transcription drop, torchaudio pitch shifts should hurt Kong far
less than librosa ones; tempo changes, where both use long windows, should hurt both.
N MAESTRO test pieces (data/pick.json: every third first, then the rest; 30 at most),
first SECONDS of each. Onset-only F1
at 50 ms and 50 cents, labels moved by whole semitones and times by the tempo change.
Rows go to results/piano{TAG}.jsonl; Kong's notes for every condition, with the time scale
and cents applied, go to results/piano{TAG}_notes.jsonl for rescoring offline (e.g. after
correcting a tool's time offset).
"""

import json
import os
import sys

import librosa
import mir_eval
import numpy as np
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from paths import MAESTRO, MAESTRO_PICK  # noqa: E402
from piano_models import load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEC = int(sys.argv[2]) if len(sys.argv) > 2 else 60
# PIANO_MODEL=edwards switches to Edwards et al.'s augmented model (same interface); its rows
# go to a file with the model name in it
MODEL = os.environ.get("PIANO_MODEL", "kong")
kong = load(MODEL)
OUT = f"{ROOT}/results/piano{harness.TAG}{'' if MODEL == 'kong' else '_' + MODEL}.jsonl"
# every third piece first (the original 10, so piece numbers stay comparable), then the rest
_all = json.load(open(MAESTRO_PICK))
pick = (_all[::3] + [r for i, r in enumerate(_all) if i % 3])[:N]
COND = harness.conditions(pitch_amounts=harness.PITCH_LABELLED)
rng = np.random.default_rng(0)


def f1(x, ref_on, ref_pi, keep_notes=None):
    """Onset F1 of Kong's transcription of x; appends (onset, pitch) pairs to keep_notes."""
    e = kong.transcribe(x.astype("float32"), None)["est_note_events"]
    on = np.array([n["onset_time"] for n in e])
    pi = np.array([n["midi_note"] for n in e])
    if keep_notes is not None:
        keep_notes += [[round(float(a), 4), int(b)] for a, b in zip(on, pi)]
    ok = on < ref_on.max() + 1 if len(on) else on.astype(bool)
    return mir_eval.transcription.precision_recall_f1_overlap(
        np.c_[ref_on, ref_on + 0.1], librosa.midi_to_hz(ref_pi),
        np.c_[on[ok], on[ok] + 0.1], librosa.midi_to_hz(pi[ok]),
        offset_ratio=None, onset_tolerance=0.05,
    )[2] if ok.any() else 0.0  # fmt: skip


done = {json.loads(l)["piece"] for l in open(OUT)} if os.path.exists(OUT) else set()
with open(OUT, "a") as fout, open(OUT.replace(".jsonl", "_notes.jsonl"), "a") as fnotes:
    for k, r in enumerate(pick):
        if k in done:
            continue
        x, _ = librosa.load(
            f"{MAESTRO}/{r['audio_filename']}", sr=SR, mono=True, duration=SEC + 2
        )
        n = sorted(
            pretty_midi.PrettyMIDI(f"{MAESTRO}/{r['midi_filename']}")
            .instruments[0]
            .notes,
            key=lambda v: v.start,
        )
        on = np.array([v.start for v in n])
        pi = np.array([v.pitch for v in n])
        keep = on < SEC - 1
        sg = harness.signs(rng, pitch_amounts=harness.PITCH_LABELLED)
        row = {"piece": k, "year": r["year"]}
        notes = {"piece": k}
        for cond in COND:
            y, scale, cents = harness.apply(x, SR, cond, sg.get((cond[0], cond[1]), 1))
            # cut the audio to the scored span so every condition sees the same notes
            y = y[: int((SEC - 1) * scale * SR) + SR]
            est = []
            row[harness.key(cond)] = f1(
                y, on[keep] * scale, pi[keep] + round(cents / 100), est
            )
            notes[harness.key(cond)] = {"scale": scale, "cents": cents, "notes": est}
        fnotes.write(json.dumps(notes) + "\n")
        fnotes.flush()
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(
            k,
            {c: round(v, 3) for c, v in row.items() if "pitch10" in c or c == "clean"},
            flush=True,
        )
print("done", flush=True)
