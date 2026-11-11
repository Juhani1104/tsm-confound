"""Tool confound on timbre tasks, two models and three magnitudes: PANNs Cnn14 and AST.

usage: python src/timbre2.py > results/timbre2.txt

Same FMA-small genre probe as timbre.py, now for PANNs Cnn14 (CNN, 32 kHz) and the
Audio Spectrogram Transformer (AST, 16 kHz), both AudioSet models. Each test track is
shifted by a fixed magnitude with a random sign: pitch 10, 50 or 200 cents, tempo 5, 10
or 20 percent, made with the phase vocoder, signalsmith-stretch, or a plain resample.
Transforms are made once at 32 kHz and resampled to 16 kHz for AST, so both models see
the same audio. Rows go to results/timbre2.json.
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
from panns_inference.models import Cnn14
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import ASTFeatureExtractor, ASTForAudioClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import AST, FMA_TRACKS, FMA_ZIP, PANNS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMA = FMA_ZIP
TRACKS = FMA_TRACKS
SR = 32000
PER = {"training": 100, "test": 50}
PITCH = [10, 50, 200]  # cents
TEMPO = [0.05, 0.10, 0.20]  # fraction faster or slower

panns = Cnn14(
    sample_rate=SR,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=527,
)
panns.load_state_dict(
    torch.load(PANNS, map_location="cpu", weights_only=True)["model"], strict=True
)
panns = panns.cuda().eval()
ast = (
    ASTForAudioClassification.from_pretrained(str(AST), use_safetensors=True)
    .cuda()
    .eval()
)
ast_fe = ASTFeatureExtractor.from_pretrained(str(AST))

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


@torch.no_grad()
def embed(x):
    """{model: (embedding, AudioSet probabilities)} for mono 32 kHz x."""
    o = panns(torch.from_numpy(x[None].astype("float32")).cuda(), None)
    out = {
        "panns": (
            o["embedding"][0].cpu().numpy(),
            o["clipwise_output"][0].cpu().numpy(),
        )
    }
    y = librosa.resample(x, orig_sr=SR, target_sr=16000, res_type="soxr_hq")
    f = ast_fe(y, sampling_rate=16000, return_tensors="pt")["input_values"].cuda()
    h = ast.audio_spectrogram_transformer(f).pooler_output[0]
    out["ast"] = (
        h.cpu().numpy(),
        torch.sigmoid(ast.classifier(h[None]))[0].cpu().numpy(),
    )
    return out


def signalsmith(x, semitones=0.0, time=1.0):
    """Pitch shift by semitones or time factor (>1 is faster) with signalsmith-stretch."""
    s = python_stretch.Signalsmith.Stretch()
    s.preset(1, SR)
    s.setTransposeSemitones(semitones)
    s.setTimeFactor(time)
    return s.process(x[None])[0].astype("float32")


def rs(y, q):
    """Play y q times faster: pitch and tempo both scale by q."""
    return librosa.resample(y, orig_sr=SR * q, target_sr=SR, res_type="soxr_hq")


def transforms(x):
    """The clean audio and every (kind, magnitude, tool) version of it."""
    out = {"clean": x}
    for c in PITCH:
        c = c * rng.choice([-1, 1])
        out[f"pitch{abs(c)}_vocoder"] = librosa.effects.pitch_shift(
            x, sr=SR, n_steps=c / 100, res_type="soxr_hq"
        )
        out[f"pitch{abs(c)}_signalsmith"] = signalsmith(x, semitones=c / 100)
        out[f"pitch{abs(c)}_resample"] = rs(x, 2 ** (c / 1200))
    for d in TEMPO:
        f = 1 + d * rng.choice([-1, 1])
        k = int(round(d * 100))
        out[f"tempo{k}_vocoder"] = librosa.effects.time_stretch(x, rate=f)
        out[f"tempo{k}_signalsmith"] = signalsmith(x, time=f)
        out[f"tempo{k}_resample"] = rs(x, f)
    return out


train = {"panns": [], "ast": []}
ytr = []
for i, g in pick("training"):
    x = load(i)
    if x is None:
        continue
    e = embed(x)
    for m in train:
        train[m].append(e[m][0])
    ytr.append(g)
probes = {}
for m, X in train.items():
    sc = StandardScaler().fit(X)
    probes[m] = (sc, LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X), ytr))
print("train tracks", len(ytr), flush=True)

rows = []
for n, (i, g) in enumerate(pick("test")):
    x = load(i)
    if x is None:
        continue
    out = {k: embed(y) for k, y in transforms(x).items()}
    row = {"track": int(i), "genre": g}
    for m, (sc, pr) in probes.items():
        e0, p0 = out["clean"][m]
        top0 = set(np.argsort(-p0)[:5])
        for k, v in out.items():
            e, p = v[m]
            row[f"{m}|{k}"] = dict(
                correct=bool(pr.predict(sc.transform([e]))[0] == g),
                cos=float(e @ e0 / (np.linalg.norm(e) * np.linalg.norm(e0) + 1e-9)),
                top5=len(top0 & set(np.argsort(-p)[:5])) / 5,
            )
    rows.append(row)
    if n % 50 == 0:
        print("test", n, flush=True)
json.dump(rows, open(f"{ROOT}/results/timbre2.json", "w"), indent=0)

print("test tracks", len(rows))
for k in [c for c in rows[0] if "|" in c]:
    a = [r[k] for r in rows]
    print(
        f"{k:28s} acc {np.mean([v['correct'] for v in a]):.3f}  cos {np.mean([v['cos'] for v in a]):.4f}  top5 {np.mean([v['top5'] for v in a]):.3f}"
    )
