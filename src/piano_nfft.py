"""Causal test for torchaudio's piano damage: vary only the vocoder window.

usage: python src/piano_nfft.py

The torchaudio pitch-shift recipe (phase vocoder, then soxr resample) at n_fft 256..4096,
+200 cents, on the first 30 s of 5 MAESTRO test pieces (same pick as task_piano.py),
transcribed by Kong. Reports onset F1, precision, recall, notes per reference note, and the
median timing offset of matched notes. If short windows add false notes through poor
frequency resolution, precision should rise with n_fft while the timing offset (onset
smearing) grows.
"""

import json
import os
import sys

import librosa
import mir_eval
import numpy as np
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms  # noqa: E402
from paths import MAESTRO, MAESTRO_PICK  # noqa: E402
from piano_models import load  # noqa: E402

SR, SEC = 16000, 30
kong = load("kong")
pick = json.load(open(MAESTRO_PICK))[::3][:5]
res = {}
for r in pick:
    x, _ = librosa.load(
        f"{MAESTRO}/{r['audio_filename']}", sr=SR, mono=True, duration=SEC + 2
    )
    n = pretty_midi.PrettyMIDI(f"{MAESTRO}/{r['midi_filename']}").instruments[0].notes
    on = np.array([v.start for v in n])
    pi = np.array([v.pitch for v in n])
    k = on < SEC - 1
    for n_fft in (0, 256, 512, 1024, 2048, 4096):
        y = x if n_fft == 0 else transforms._torchaudio_pitch(x, SR, 200, n_fft=n_fft)
        y = y[: (SEC - 1) * SR + SR]
        e = kong.transcribe(y.astype("float32"), None)["est_note_events"]
        eo = np.array([v["onset_time"] for v in e])
        ep = np.array([v["midi_note"] for v in e])
        ok = eo < on[k].max() + 1
        ro, rp = on[k], pi[k] + (0 if n_fft == 0 else 2)
        p, rc, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            np.c_[ro, ro + 0.1], librosa.midi_to_hz(rp), np.c_[eo[ok], eo[ok] + 0.1], librosa.midi_to_hz(ep[ok]),
            offset_ratio=None, onset_tolerance=0.05)  # fmt: skip
        d = [
            eo[ok][j] - t
            for t, q in zip(ro, rp)
            for j in np.flatnonzero((ep[ok] == q) & (np.abs(eo[ok] - t) < 0.05))[:1]
        ]
        res.setdefault(n_fft, []).append(
            (f, p, rc, ok.sum() / len(ro), np.median(d) * 1000)
        )
print(
    f"{'n_fft':>6s} {'F1':>6s} {'prec':>6s} {'recall':>7s} {'est/ref':>8s} {'offset ms':>10s}"
)
for n_fft, v in res.items():
    v = np.array(v)
    print(
        f"{'clean' if n_fft == 0 else n_fft:>6} {v[:, 0].mean():6.3f} {v[:, 1].mean():6.3f} {v[:, 2].mean():7.3f} {v[:, 3].mean():8.2f} {np.median(v[:, 4]):+10.1f}"
    )
