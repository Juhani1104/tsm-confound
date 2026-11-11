"""Where do a tool's notes land in time? Per-event lag of every tool's output, over time.

usage: python src/rb_offset.py

Rubber Band (pedalboard) at +-10 cents cut note and beat scores far more than at +-200
cents. Hypothesis: a time offset that depends on the shift ratio and the signal, larger
than the 50-70 ms scoring tolerance. Two measurements, both model-free and CPU only:

1. Click train: short decaying 440 Hz tone bursts every 0.37 s for 30 s. For each burst,
   the lag between its true start and the onset found in the processed audio (energy
   envelope crossing half its local peak). Gives lag at a known time, so drift shows up as
   a slope.
2. Real piano: first 30 s of 6 MAESTRO pieces. Lag that maximises the cross-correlation of
   onset-strength envelopes in 3 s windows every 1 s (search +-250 ms).

Output: results/rb_offset.json and a printed table.
"""

import json
import os
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms  # noqa: E402
from paths import MAESTRO, MAESTRO_PICK  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
TOOLS = ("librosa", "torchaudio", "signalsmith", "rubberband", "rbsmall", "resample")
CENTS = (-200, -50, -10, 10, 50, 200)


def bursts(sr, period=0.37, dur=30.0):
    """Click train of short decaying 440 Hz bursts, and the burst start times."""
    t0 = np.arange(0.2, dur - 0.5, period)
    x = np.zeros(int(dur * sr), np.float32)
    n = np.arange(int(0.12 * sr))
    b = (np.sin(2 * np.pi * 440 * n / sr) * np.exp(-n / (0.03 * sr))).astype(np.float32)
    for t in t0:
        i = int(round(t * sr))
        x[i : i + len(b)] += 0.5 * b
    return x, t0


def burst_lags(y, sr, t0):
    """Lag (ms) of each burst onset in y: first sample whose 1 ms envelope reaches half the
    peak inside [t-150 ms, t+250 ms]."""
    env = np.convolve(np.abs(y), np.ones(int(0.001 * sr)) / (0.001 * sr), "same")
    out = []
    for t in t0:
        a, b = int((t - 0.15) * sr), int((t + 0.25) * sr)
        seg = env[max(a, 0) : b]
        if seg.max() <= 0:
            out.append(np.nan)
            continue
        i = np.argmax(seg >= 0.5 * seg.max())
        out.append(((max(a, 0) + i) / sr - t) * 1000)
    return np.array(out)


def window_lags(x, y, sr, win=3.0, step=1.0, maxlag=0.25):
    """Lag (ms) of y against x in sliding windows, from onset-envelope cross-correlation."""
    hop = 160
    ox = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop)
    oy = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    fps = sr / hop
    L, W, S = int(maxlag * fps), int(win * fps), int(step * fps)
    lags = []
    for s in range(L, min(len(ox), len(oy)) - W - L, S):
        a = ox[s : s + W] - ox[s : s + W].mean()
        c = [
            np.dot(a, oy[s + k : s + k + W] - oy[s + k : s + k + W].mean())
            for k in range(-L, L + 1)
        ]
        lags.append((np.argmax(c) - L) / fps * 1000)
    return np.array(lags)


def summary(l):
    """Median, 5th/95th percentile, share beyond 50 ms and slope of a lag series."""
    l = l[~np.isnan(l)]
    slope = np.polyfit(np.arange(len(l)), l, 1)[0] if len(l) > 2 else np.nan
    return {"median": float(np.median(l)), "p5": float(np.percentile(l, 5)), "p95": float(np.percentile(l, 95)),
            "frac_over_50ms": float(np.mean(np.abs(l) > 50)), "slope_per_event": float(slope)}  # fmt: skip


if __name__ == "__main__":
    res = {"bursts": {}, "piano": {}}
    x, t0 = bursts(SR)
    print("bursts: lag ms median [p5,p95] frac>50ms slope/event")
    for tool in TOOLS:
        for c in CENTS:
            y, scale = transforms.pitch(x, SR, c, tool)
            s = summary(burst_lags(y, SR, t0 * scale))
            s["len_ratio"] = len(y) / len(x)
            res["bursts"][f"{tool}_{c}"] = s
            print(
                f"{tool:11s} {c:+5d}  {s['median']:+7.1f} [{s['p5']:+7.1f},{s['p95']:+7.1f}]  {s['frac_over_50ms']:.2f}  {s['slope_per_event']:+.2f}  len {s['len_ratio']:.3f}",
                flush=True,
            )

    pick = json.load(open(MAESTRO_PICK))[::3][:6]
    # resample changes tempo on purpose (labels are rescaled), so it is left out here
    print("\npiano windows: lag ms median [p5,p95] frac>50ms (pooled over 6 pieces)")
    xs = [
        librosa.load(f"{MAESTRO}/{r['audio_filename']}", sr=SR, mono=True, duration=30)[
            0
        ]
        for r in pick
    ]
    for tool in TOOLS[:-1]:
        for c in CENTS:
            per = []
            for x in xs:
                y, _ = transforms.pitch(x, SR, c, tool)
                per.append(window_lags(x, y, SR))
            s = summary(np.concatenate(per))
            s["piece_medians"] = [float(np.median(p)) for p in per]
            res["piano"][f"{tool}_{c}"] = s
            print(
                f"{tool:11s} {c:+5d}  {s['median']:+7.1f} [{s['p5']:+7.1f},{s['p95']:+7.1f}]  {s['frac_over_50ms']:.2f}  pieces {[round(v) for v in s['piece_medians']]}",
                flush=True,
            )

    json.dump(res, open(f"{ROOT}/results/rb_offset.json", "w"), indent=1)
