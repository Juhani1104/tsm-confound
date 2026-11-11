"""Does the shifting tool, not the shift, decide a beat tracker's measured robustness?

usage: python src/beats.py N > results/beats.txt

Beat This! (final0) on N FMA-small tracks of beat-driven genres, 30 s each. Its beats on
the clean audio are the reference, so the score is how much a transform changes the
output. Each track gets one random pitch offset in [-10, 10] cents and one random tempo
factor in [0.95, 1.05], each made three ways:
vocoder      librosa phase vocoder (pitch_shift / time_stretch)
signalsmith  signalsmith-stretch, the current audiomentations default
resample     changes pitch and tempo together and adds no time-stretch artifact
Reference beats are rescaled by the tempo change. F-measure uses mir_eval's 70 ms
window, once as is and once at the best constant lag in +-60 ms, so a fixed tool latency
is separated from beats actually lost.
"""

import io
import json
import os
import sys
import zipfile

import librosa
import mir_eval
import numpy as np
import pandas as pd
import python_stretch
import soundfile as sf
from beat_this.inference import Audio2Beats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BEAT_THIS, FMA_META, FMA_ZIP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMA = FMA_ZIP
META = FMA_META
SR = 22050
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
GENRES = {"Rock", "Electronic", "Hip-Hop", "Pop"}
LAGS = np.arange(-0.06, 0.0601, 0.005)

tracker = Audio2Beats(checkpoint_path=str(BEAT_THIS), device="cuda")
meta = pd.read_csv(META, index_col=0)
z = zipfile.ZipFile(FMA)
names = sorted(
    m
    for m in z.namelist()
    if m.endswith(".mp3") and meta["genre"].get(int(os.path.basename(m)[:-4])) in GENRES
)
names = list(np.random.default_rng(0).permutation(names))
rng = np.random.default_rng(1)


def signalsmith(x, semitones=0.0, time=1.0):
    """Pitch shift by semitones or time factor (>1 is faster) with signalsmith-stretch."""
    s = python_stretch.Signalsmith.Stretch()
    s.preset(1, SR)
    s.setTransposeSemitones(semitones)
    s.setTimeFactor(time)
    return s.process(x[None])[0].astype("float32")


def resample(x, ratio):
    """Play x ratio times faster: pitch and tempo both scale by ratio."""
    return librosa.resample(x, orig_sr=SR * ratio, target_sr=SR, res_type="soxr_hq")


def fmeasure(ref, est):
    """F-measure at lag 0 and at the best constant lag."""
    ref = mir_eval.beat.trim_beats(ref)
    f0 = mir_eval.beat.f_measure(ref, mir_eval.beat.trim_beats(est))
    best = max(
        mir_eval.beat.f_measure(
            ref, mir_eval.beat.trim_beats(np.clip(est - g, 0, None))
        )
        for g in LAGS
    )
    return f0, best


rows, used = [], 0
for name in names:
    if used == N:
        break
    try:
        x, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=True)
    except Exception:  # a few FMA files do not decode
        continue
    x = librosa.resample(x.mean(1), orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    if len(x) < 30 * SR:
        continue
    x = x[: 30 * SR]
    ref, _ = tracker(x, SR)
    if len(ref) < 20:  # no clear pulse: skip
        continue
    used += 1
    c = float(rng.uniform(-10, 10))
    t = float(rng.uniform(0.95, 1.05))
    r = 2 ** (c / 1200)
    cases = {
        "pitch_vocoder": (
            librosa.effects.pitch_shift(x, sr=SR, n_steps=c / 100, res_type="soxr_hq"),
            1.0,
        ),
        "pitch_signalsmith": (signalsmith(x, semitones=c / 100), 1.0),
        "pitch_resample": (resample(x, r), 1 / r),
        "tempo_vocoder": (librosa.effects.time_stretch(x, rate=t), 1 / t),
        "tempo_signalsmith": (signalsmith(x, time=t), 1 / t),
        "tempo_resample": (resample(x, t), 1 / t),
    }
    row = {"track": os.path.basename(name), "cents": c, "tempo": t}
    for k, (y, scale) in cases.items():
        est, _ = tracker(y, SR)
        row[k] = fmeasure(ref * scale, est)
    rows.append(row)
    print(
        row["track"], {k: tuple(round(v, 3) for v in row[k]) for k in cases}, flush=True
    )
json.dump(rows, open(f"{ROOT}/results/beats.json", "w"), indent=0)
print("\ntracks", len(rows))
for k in cases:
    a = np.array([r[k] for r in rows])
    print(
        f"{k:18s} F lag0 {a[:, 0].mean():.3f}  F best-lag {a[:, 1].mean():.3f}"
        f"  tracks below 0.9 (best-lag) {(a[:, 1] < 0.9).mean():.2f}"
    )
