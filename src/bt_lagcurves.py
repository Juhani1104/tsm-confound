"""The exact time error of each Beat This! tempo augmentation, by simulation.

usage: python src/bt_lagcurves.py [NPZ ...]

pedalboard's drift depends only on the stretch factor and the time since the start of the
file (not on the content or the file length; checked for 40, 100 and 300 s inputs), so one
click train per factor gives the error curve for every track. For each tempo percentage s
in -20..20 step 4 the click train (bursts every 0.5 s, 900 s, 44.1 kHz as in Beat This!'s
preprocess_audio.py) is stretched with pedalboard.time_stretch(stretch_factor=1+s/100), and
the lag of every burst against its scaled time is measured.
Saved to results/bt_lagcurves.npz: 'ts{s}_t' (time in the stretched file, s) and 'ts{s}_lag' (ms,
positive = content later than the scaled labels).

With NPZ arguments (Beat This! spectrogram bundles), the curves are checked against the
lags measured on the released spectrograms (as in bt_spect_check.py): the residual after
subtracting the curve should be near zero if the curve explains the data.
"""

import os
import sys

import numpy as np
import pedalboard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS  # noqa: E402
from rb_offset import burst_lags, bursts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{RESULTS}/bt_lagcurves.npz"
SR = 44100
PCTS = [s for s in range(-20, 21, 4) if s]

if not os.path.exists(OUT):
    x, t0 = bursts(SR, period=0.5, dur=900.0)
    curves = {}
    for s in PCTS:
        f = 1 + s / 100
        y = pedalboard.time_stretch(x[None], SR, stretch_factor=f)[0]
        lag = burst_lags(y, SR, t0 / f)
        curves[f"ts{s}_t"] = t0 / f
        curves[f"ts{s}_lag"] = lag
        print(
            f"ts{s:+d}: lag at 30/60/300/800 s "
            + " ".join(f"{np.interp(t, t0 / f, lag):+.0f}" for t in (30, 60, 300, 800))
            + f"  max |lag| {np.nanmax(np.abs(lag)):.0f}",
            flush=True,
        )
    np.savez(OUT, **curves)
curves = dict(np.load(OUT))


def lag_at(s, t):
    """Simulated lag (ms) of tempo augmentation s at stretched-file times t (s)."""
    return np.interp(t, curves[f"ts{s}_t"], curves[f"ts{s}_lag"])


if len(sys.argv) > 1:
    import bt_spect_check as chk  # noqa: E402

    for path in sys.argv[1:]:
        res = chk.measure(path)
        print(f"\n{path}: measured vs simulated lag, ms")
        print(
            f"{'aug':6s} {'measured median':>16s} {'residual median':>16s} {'residual p5..p95':>18s} {'|res|>20ms':>11s}"
        )
        for s in PCTS:
            v = np.array(res.get(f"ts{s}", []))
            if not len(v):
                continue
            r = v[:, 1] - lag_at(s, v[:, 0])
            print(
                f"ts{s:+d}".ljust(6)
                + f" {np.median(v[:, 1]):+16.1f} {np.median(r):+16.1f} {np.percentile(r, 5):+8.1f}..{np.percentile(r, 95):+7.1f} {np.mean(np.abs(r) > 20):11.2f}",
                flush=True,
            )
