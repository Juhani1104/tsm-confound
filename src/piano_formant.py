"""Does Rubber Band's formant preservation explain its +200 cent piano loss?

usage: python src/piano_formant.py

pedalboard.time_stretch defaults to preserve_formants=True, and rbsmall copies that option.
Formant preservation keeps the spectral envelope in place while the partials move, so the
relative strengths of a piano note's partials change, which a transcriber may read as other
notes. Same 5 pieces x 30 s, +200 cents, Kong: Rubber Band (small blocks) with formants
preserved vs shifted, and signalsmith as the reference.
"""

import json
import os
import sys

import librosa
import mir_eval
import numpy as np
import pretty_midi
import pylibrb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms  # noqa: E402
from paths import MAESTRO, MAESTRO_PICK  # noqa: E402
from piano_models import load  # noqa: E402

SR, SEC = 16000, 30
kong = load("kong")
pick = json.load(open(MAESTRO_PICK))[::3][:5]
O = pylibrb.Option


def rb(x, formant):
    """+200 cents by Rubber Band R3 in 4096-sample blocks, formants preserved or shifted."""
    opts = (
        O.PROCESS_OFFLINE
        | O.THREADING_NEVER
        | O.CHANNELS_TOGETHER
        | O.PitchHighQuality
        | O.ENGINE_FINER
        | formant
    )
    st = pylibrb.RubberBandStretcher(SR, 1, opts, 1.0, 2 ** (200 / 1200))
    st.set_logging_level(0)
    a = x[None]
    st.set_expected_input_duration(len(x))
    st.set_max_process_size(4096)
    for i in range(0, len(x), 4096):
        st.study(np.ascontiguousarray(a[:, i : i + 4096]), final=i + 4096 >= len(x))
    out = []
    for i in range(0, len(x), 4096):
        st.process(np.ascontiguousarray(a[:, i : i + 4096]), final=i + 4096 >= len(x))
        out.append(st.retrieve_available()[0])
    while not st.is_done():
        out.append(st.retrieve_available()[0])
    return np.concatenate(out).astype(np.float32)


tools = {"signalsmith": lambda x: transforms.pitch(x, SR, 200, "signalsmith")[0],
         "rubberband formant preserved": lambda x: rb(x, O.FORMANT_PRESERVED),
         "rubberband formant shifted": lambda x: rb(x, O.FORMANT_SHIFTED)}  # fmt: skip
res = {}
for r in pick:
    x, _ = librosa.load(
        f"{MAESTRO}/{r['audio_filename']}", sr=SR, mono=True, duration=SEC + 2
    )
    n = pretty_midi.PrettyMIDI(f"{MAESTRO}/{r['midi_filename']}").instruments[0].notes
    on = np.array([v.start for v in n])
    pi = np.array([v.pitch for v in n])
    k = on < SEC - 1
    for name, f in tools.items():
        y = f(x)[: (SEC - 1) * SR + SR]
        e = kong.transcribe(y.astype("float32"), None)["est_note_events"]
        eo = np.array([v["onset_time"] for v in e])
        ep = np.array([v["midi_note"] for v in e])
        ok = eo < on[k].max() + 1
        p, rc, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            np.c_[on[k], on[k] + 0.1], librosa.midi_to_hz(pi[k] + 2), np.c_[eo[ok], eo[ok] + 0.1], librosa.midi_to_hz(ep[ok]),
            offset_ratio=None, onset_tolerance=0.05)  # fmt: skip
        res.setdefault(name, []).append((f1, p, rc, ok.sum() / k.sum()))
print(f"{'tool':30s} {'F1':>6s} {'prec':>6s} {'recall':>7s} {'est/ref':>8s}")
for name, v in res.items():
    v = np.array(v)
    print(
        f"{name:30s} {v[:, 0].mean():6.3f} {v[:, 1].mean():6.3f} {v[:, 2].mean():7.3f} {v[:, 3].mean():8.2f}"
    )
