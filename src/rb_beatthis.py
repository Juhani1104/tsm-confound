"""How far do pedalboard time_stretch outputs drift from their scaled labels, at the settings
Beat This! (Foscarin et al. 2024) uses to precompute its tempo augmentation?

usage: python src/rb_beatthis.py

launch_scripts/preprocess_audio.py in CPJKU/beat_this calls
pedalboard.time_stretch(waveform, 44100, stretch_factor=1 + s/100) for s in -20..20 step 4
(defaults otherwise, so high_quality=True), and beat_this/dataset/augment.py divides the
beat times by the same factor. A click train stands in for the beats: bursts every 0.5 s for
240 s (a typical track length); for each burst, the lag between its scaled time and the
onset found in the stretched audio. Same for the drift-free rbsmall tool as a reference.
Output: results/rb_beatthis.txt
"""

import os
import sys

import numpy as np
import pedalboard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms  # noqa: E402
from paths import MAESTRO, MAESTRO_PICK  # noqa: E402
from rb_offset import burst_lags, bursts  # noqa: E402

SR = 44100
x, t0 = bursts(SR, period=0.5, dur=240.0)
at = [10, 30, 60, 120, 180, 230]
print(
    f"lag ms at {at} s; beat F tolerance 70 ms; share of bursts off by more than 70 ms"
)
for s in range(-20, 21, 4):
    if s == 0:
        continue
    f = 1 + s / 100
    for name, y in (("pedalboard", pedalboard.time_stretch(x[None], SR, stretch_factor=f, pitch_shift_in_semitones=0.0)[0]),
                    ("rbsmall", transforms.tempo(x, SR, f, "rbsmall")[0])):  # fmt: skip
        l = burst_lags(y, SR, t0 / f)
        i = np.searchsorted(t0, at)
        print(
            f"{s:+3d}% {name:10s} "
            + " ".join(f"{v:+6.0f}" for v in l[i])
            + f"   >70ms {np.mean(np.abs(l) > 70):.2f}  max {np.nanmax(np.abs(l)):.0f}",
            flush=True,
        )

# Real music check: 2 MAESTRO pieces, first 240 s at 44.1 kHz. pedalboard's output is compared
# with rbsmall's (same stretch, no drift) by windowed onset-envelope cross-correlation, so no
# labels are needed; a lag here is pedalboard's drift relative to a correct stretch.
import json  # noqa: E402

import librosa  # noqa: E402

from rb_offset import window_lags  # noqa: E402

pick = json.load(open(MAESTRO_PICK))[::3][:2]
print("\nreal piano, lag of pedalboard vs rbsmall, ms (3 s windows, 3.6 ms resolution)")
for r in pick:
    x, _ = librosa.load(
        f"{MAESTRO}/{r['audio_filename']}", sr=SR, mono=True, duration=240
    )
    for s in (-12, -8, 8, 12):
        f = 1 + s / 100
        a = pedalboard.time_stretch(x[None], SR, stretch_factor=f)[0]
        b = transforms.tempo(x, SR, f, "rbsmall")[0]
        n = min(len(a), len(b))
        l = window_lags(b[:n], a[:n], SR)
        q = np.linspace(0, len(l) - 1, 6).astype(int)
        print(
            f"{r['audio_filename'][-30:]} {s:+3d}%  "
            + " ".join(f"{l[k]:+5.0f}" for k in q)
            + f"   >70ms {np.mean(np.abs(l) > 70):.2f}",
            flush=True,
        )
