"""Does the shifting tool decide measured robustness on a timbre task? PANNs on FMA genres.

usage: python src/timbre.py > results/timbre.txt

PANNs Cnn14 (AudioSet) embeds 10 s from the middle of each FMA-small track. A logistic
probe for the 8 FMA genres is trained on clean embeddings of the official train split
(100 tracks per genre) and tested on the test split (50 per genre) under the transforms
of beats.py: one random pitch offset in [-10, 10] cents and one tempo factor in
[0.95, 1.05] per track, each made with the phase vocoder, signalsmith-stretch, or a plain
resample. Besides genre accuracy against FMA labels, it reports how far each transform
moves the embedding (cosine to clean) and the AudioSet top-5 tags (overlap with clean).
"""

import io
import json
import os
import sys
import zipfile

import librosa
import numpy as np
import pandas as pd
import python_stretch
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panns_cnn14 import Cnn14  # noqa: E402
from paths import FMA_TRACKS, FMA_ZIP, PANNS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMA = FMA_ZIP
TRACKS = FMA_TRACKS
SR = 32000
PER = {"training": 100, "test": 50}

net = Cnn14(
    sample_rate=SR,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=527,
)
ck = torch.load(PANNS, map_location="cpu", weights_only=True)
net.load_state_dict(ck["model"], strict=True)
net = net.cuda().eval()

t = pd.read_csv(TRACKS, index_col=0, header=[0, 1], low_memory=False)
small = t[t[("set", "subset")] == "small"]
z = zipfile.ZipFile(FMA)
have = {int(os.path.basename(m)[:-4]): m for m in z.namelist() if m.endswith(".mp3")}
rng = np.random.default_rng(0)


def pick(split):
    """Up to PER[split] track ids per genre from the official FMA-small split."""
    s = small[small[("set", "split")] == split]
    out = []
    for g, grp in s.groupby(("track", "genre_top")):
        ids = [i for i in rng.permutation(grp.index.values) if i in have]
        out += [(i, g) for i in ids[: PER[split]]]
    return out


def load(i):
    """Mono 32 kHz, the middle 10 s of track i, or None if it does not decode."""
    try:
        x, sr = sf.read(io.BytesIO(z.read(have[i])), dtype="float32", always_2d=True)
    except Exception:
        return None
    x = librosa.resample(x.mean(1), orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    if len(x) < 12 * SR:
        return None
    m = len(x) // 2
    return x[m - 5 * SR : m + 5 * SR]


def embed(x):
    """(2048-d embedding, 527 AudioSet probabilities) of mono 32 kHz x."""
    with torch.no_grad():
        o = net(torch.from_numpy(x[None].astype("float32")).cuda(), None)
    return o["embedding"][0].cpu().numpy(), o["clipwise_output"][0].cpu().numpy()


def signalsmith(x, semitones=0.0, time=1.0):
    """Pitch shift by semitones or time factor (>1 is faster) with signalsmith-stretch."""
    s = python_stretch.Signalsmith.Stretch()
    s.preset(1, SR)
    s.setTransposeSemitones(semitones)
    s.setTimeFactor(time)
    return s.process(x[None])[0].astype("float32")


def transforms(x):
    """Every transformed version of x, with one random offset and tempo shared by all tools."""
    c = float(rng.uniform(-10, 10))
    f = float(rng.uniform(0.95, 1.05))
    r = 2 ** (c / 1200)

    def rs(y, q):
        return librosa.resample(y, orig_sr=SR * q, target_sr=SR, res_type="soxr_hq")

    return {
        "clean": x,
        "pitch_vocoder": librosa.effects.pitch_shift(
            x, sr=SR, n_steps=c / 100, res_type="soxr_hq"
        ),
        "pitch_signalsmith": signalsmith(x, semitones=c / 100),
        "pitch_resample": rs(x, r),
        "tempo_vocoder": librosa.effects.time_stretch(x, rate=f),
        "tempo_signalsmith": signalsmith(x, time=f),
        "tempo_resample": rs(x, f),
    }


Xtr, ytr = [], []
for i, g in pick("training"):
    x = load(i)
    if x is not None:
        Xtr.append(embed(x)[0])
        ytr.append(g)
scaler = StandardScaler().fit(Xtr)
probe = LogisticRegression(max_iter=3000, C=0.5).fit(scaler.transform(Xtr), ytr)
print("train tracks", len(ytr), flush=True)

rows = []
for i, g in pick("test"):
    x = load(i)
    if x is None:
        continue
    out = {k: embed(y) for k, y in transforms(x).items()}
    e0, p0 = out["clean"]
    top0 = set(np.argsort(-p0)[:5])
    row = {"track": int(i), "genre": g}
    for k, (e, p) in out.items():
        row[k] = dict(
            correct=bool(probe.predict(scaler.transform([e]))[0] == g),
            cos=float(e @ e0 / (np.linalg.norm(e) * np.linalg.norm(e0) + 1e-9)),
            top5=len(top0 & set(np.argsort(-p)[:5])) / 5,
        )
    rows.append(row)
json.dump(rows, open(f"{ROOT}/results/timbre.json", "w"), indent=0)

print("test tracks", len(rows))
for k in rows[0]:
    if k in ("track", "genre"):
        continue
    a = [r[k] for r in rows]
    print(
        f"{k:18s} genre acc {np.mean([v['correct'] for v in a]):.3f}  cos to clean {np.mean([v['cos'] for v in a]):.4f}  top5 overlap {np.mean([v['top5'] for v in a]):.3f}"
    )
