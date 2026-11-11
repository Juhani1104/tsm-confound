"""Why do phase-vocoder outputs look ~20 ms early? Onset detection vs energy centre, by n_fft.

usage: python src/vocoder_offset.py

Hypothesis: the vocoder does not move the sound; its analysis window smears energy
symmetrically around each onset (pre-echo), so any detector that fires on the rise
(half-peak threshold, onset envelopes, a transcriber's onset head) fires early, and more so
for longer windows. Test on the click train of rb_offset.py (16 kHz): for librosa
pitch_shift and time_stretch at n_fft 512..4096, the lag of (a) the half-peak onset and (b)
the energy centroid of each burst, against the clean burst's own onset and centroid.
If (b) stays near 0 while (a) scales with n_fft, the offset is smearing, not a shift.
"""

import os
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rb_offset import burst_lags, bursts  # noqa: E402

SR = 16000
x, t0 = bursts(SR, dur=20.0)


def centroids(y, t0, scale=1.0):
    """Energy centroid (s) of each burst inside [t-150 ms, t+250 ms] of its expected time."""
    out = []
    for t in t0 * scale:
        a, b = int((t - 0.15) * SR), int((t + 0.25) * SR)
        e = y[max(a, 0) : b] ** 2
        out.append((max(a, 0) + (np.arange(len(e)) * e).sum() / e.sum()) / SR)
    return np.array(out)


ref_on = burst_lags(x, SR, t0)
ref_c = centroids(x, t0)
print(f"{'transform':28s} {'n_fft':>6s} {'onset lag ms':>13s} {'centroid lag ms':>16s}")
for n_fft in (512, 1024, 2048, 4096):
    for name, f in (("pitch_shift +50 cents", lambda z, n_fft=n_fft: (librosa.effects.pitch_shift(z, sr=SR, n_steps=0.5, n_fft=n_fft), 1.0)),
                    ("time_stretch x1.10", lambda z, n_fft=n_fft: (librosa.effects.time_stretch(z, rate=1.1, n_fft=n_fft), 1 / 1.1))):  # fmt: skip
        y, s = f(x)
        on = np.median(burst_lags(y, SR, t0 * s) - ref_on * s)
        ce = np.median((centroids(y, t0, s) - ref_c * s) * 1000)
        print(f"{name:28s} {n_fft:6d} {on:+13.1f} {ce:+16.1f}", flush=True)

# Second test: Gaussian-windowed 440 Hz tones (sigma 50 ms, every 0.6 s). Their envelope is
# smooth and symmetric, so symmetric smearing leaves the energy centroid in place; any
# centroid move here is a real time shift of the sound.
t0g = np.arange(0.5, 19.0, 0.6)
tt = np.arange(int(20 * SR)) / SR
g = np.zeros_like(tt, dtype=np.float32)
for t in t0g:
    g += (np.sin(2 * np.pi * 440 * tt) * np.exp(-0.5 * ((tt - t) / 0.05) ** 2)).astype(
        np.float32
    )
ref_g = centroids(g, t0g)
print(f"\n{'Gaussian tones':28s} {'n_fft':>6s} {'centroid lag ms':>16s}")
for n_fft in (512, 1024, 2048, 4096):
    for name, f in (("pitch_shift +50 cents", lambda z, n_fft=n_fft: (librosa.effects.pitch_shift(z, sr=SR, n_steps=0.5, n_fft=n_fft), 1.0)),
                    ("pitch_shift +10 cents", lambda z, n_fft=n_fft: (librosa.effects.pitch_shift(z, sr=SR, n_steps=0.1, n_fft=n_fft), 1.0)),
                    ("time_stretch x1.10", lambda z, n_fft=n_fft: (librosa.effects.time_stretch(z, rate=1.1, n_fft=n_fft), 1 / 1.1))):  # fmt: skip
        y, s = f(g)
        ce = np.median((centroids(y, t0g, s) - ref_g * s) * 1000)
        print(f"{name:28s} {n_fft:6d} {ce:+16.2f}", flush=True)
