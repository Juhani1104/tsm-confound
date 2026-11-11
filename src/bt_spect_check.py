"""Check Beat This!'s released training spectrograms for the pedalboard drift.

usage: python src/bt_spect_check.py NPZ

NPZ is a per-dataset bundle from Zenodo record 13922116 (CC BY 4.0), e.g. guitarset.npz,
holding for every track the log-mel spectrogram (50 frames/s) of the original audio
('.../track') and of each precomputed augmentation ('.../track_ts8' for +8% tempo,
'.../track_ps3' for +3 semitones). Training divides the beat labels of a ts version by
1 + s/100, so its content should sit exactly at the original times divided by that factor.

For each track and augmentation, a spectral-flux onset envelope of the original is
resampled onto the augmented timeline and cross-correlated with the augmented one in 3 s
windows (lag search +-300 ms, parabolic refinement). Pitch-shifted versions (made with
pedalboard's PitchShift plugin, which does not drift) are the control.
Output: printed table and results/bt_spect_check_<dataset>.json.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS  # noqa: E402

FPS = 50


def flux(S):
    """Spectral-flux onset envelope (one value per frame) of a log-mel spectrogram."""
    S = np.asarray(S, np.float32)
    if S.shape[0] < S.shape[1]:  # (mels, frames) -> (frames, mels)
        S = S.T
    d = np.maximum(np.diff(S, axis=0), 0).sum(1)
    return np.r_[0, d]


def lags(e0, e1, win=3.0, step=1.0, maxlag=0.3):
    """[(window centre s, lag ms)]: where e1's content sits relative to e0, per window."""
    W, S, L = int(win * FPS), int(step * FPS), int(maxlag * FPS)
    out = []
    for s in range(L, min(len(e0), len(e1)) - W - L, S):
        a = e1[s : s + W] - e1[s : s + W].mean()
        c = np.array(
            [
                np.dot(a, e0[s + k : s + k + W] - e0[s + k : s + k + W].mean())
                for k in range(-L, L + 1)
            ]
        )
        i = int(np.argmax(c))
        off = 0.0
        if 0 < i < len(c) - 1 and (c[i - 1] - 2 * c[i] + c[i + 1]) != 0:
            off = 0.5 * (c[i - 1] - c[i + 1]) / (c[i - 1] - 2 * c[i] + c[i + 1])
        # positive: augmented content comes later than the scaled labels say
        out.append(((s + W / 2) / FPS, -(i - L + off) / FPS * 1000))
    return out


def measure(path):
    """{aug name: [(window centre s, lag ms), ...]} over all tracks of a bundle."""
    z = np.load(path, allow_pickle=False)
    keys = list(z.keys())
    tracks = sorted({k.rsplit("/", 1)[0] for k in keys if k.endswith("/track")})
    print(len(keys), "arrays,", len(tracks), "tracks; e.g.", keys[:3], flush=True)
    res = defaultdict(list)
    for t in tracks:
        e0 = flux(z[t + "/track"])
        t0 = np.arange(len(e0)) / FPS
        for k in keys:
            if not k.startswith(t + "/track_"):
                continue
            aug = k.rsplit("_", 1)[1]
            if aug.startswith("ts"):
                f = 1 + int(aug[2:]) / 100
            elif aug.startswith("ps"):
                f = 1.0
            else:
                continue
            e1 = flux(z[k])
            # original envelope on the augmented timeline: content at t in the original is at t/f
            e0s = np.interp(np.arange(len(e1)) / FPS * f, t0, e0)
            res[aug] += lags(e0s, e1)
    return res


if __name__ == "__main__":
    path = sys.argv[1]
    OUT = f"{RESULTS}/bt_spect_check_" + path.rsplit("/", 1)[-1].replace(
        ".npz", ".json"
    )
    res = measure(path)
    summary = {}
    print(
        f"{'aug':6s} {'windows':>8s} {'median':>8s} {'|lag|>70ms':>11s}   median lag by time (0-10, 10-30, 30-60, 60-120, 120+ s)"
    )
    for aug in sorted(res, key=lambda a: (a[:2], int(a[2:]))):
        v = np.array(res[aug])
        tm, lg = v[:, 0], v[:, 1]
        bins = [
            np.median(lg[(tm >= a) & (tm < b)])
            if ((tm >= a) & (tm < b)).any()
            else np.nan
            for a, b in ((0, 10), (10, 30), (30, 60), (60, 120), (120, 1e9))
        ]
        summary[aug] = {
            "n": len(lg),
            "median": float(np.median(lg)),
            "frac_over_70": float(np.mean(np.abs(lg) > 70)),
            "by_time": [float(b) for b in bins],
        }
        print(
            f"{aug:6s} {len(lg):8d} {np.median(lg):+8.1f} {np.mean(np.abs(lg) > 70):11.2f}   "
            + " ".join(f"{b:+7.1f}" for b in bins),
            flush=True,
        )
    json.dump(summary, open(OUT, "w"), indent=1)
