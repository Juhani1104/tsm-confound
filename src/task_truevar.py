"""True-variation control: does a tool's pitch shift look like a real pitch change?

usage: python src/task_truevar.py N_PAIRS

NSynth has the same instrument (same velocity) recorded at neighbouring MIDI pitches. For
a pair (note at p, note at p+2), the real note at p+2 is what a 200-cent shift *should*
produce, and every tool's +200-cent shift of the note at p can be compared with it (and
the reverse direction with -200 cents). A model's drop under a tool shift is a fair test
of pitch robustness only if the real note at p+2 is classified about as well.

Per item: the family predicted by the clean-trained probes (PANNs, AST; same recipe as
task_nsynth_probe.py), the cosine distance between each version's embedding and the real
target's, and a log-mel distance (dB, 50 Hz-8 kHz) to the real target.
Versions: real (target recording), orig (unshifted source), and each tool's shift.
Rows go to results/truevar.jsonl.
"""

import collections
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import ASTFeatureExtractor, ASTForAudioClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms  # noqa: E402
from panns_cnn14 import Cnn14  # noqa: E402
from paths import AST, NSYNTH, PANNS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N_TRAIN = 150
OUT = f"{ROOT}/results/truevar.jsonl"
NS = {s: f"{NSYNTH}/nsynth-{s}" for s in ("valid", "test")}
TOOLS = ("librosa", "torchaudio", "signalsmith", "rubberband", "rbsmall", "resample")

panns = Cnn14(
    sample_rate=32000,
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
rng = np.random.default_rng(0)
MEL = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=64, fmin=50, fmax=8000)


@torch.no_grad()
def embed(x):
    """{model: embedding} for mono 16 kHz x."""
    y = librosa.resample(x, orig_sr=SR, target_sr=32000, res_type="soxr_hq")
    p = (
        panns(torch.from_numpy(y[None].astype("float32")).cuda(), None)["embedding"][0]
        .cpu()
        .numpy()
    )
    f = ast_fe(x, sampling_rate=SR, return_tensors="pt")["input_values"].cuda()
    a = ast.audio_spectrogram_transformer(f).pooler_output[0].cpu().numpy()
    return {"panns": p, "ast": a}


def logmel(x):
    """64-band log-mel power (dB), 50 Hz to 8 kHz."""
    S = MEL @ np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) ** 2
    return librosa.power_to_db(S, ref=1.0, top_db=None)


def mel_dist(a, b):
    """Mean |dB| difference over frames where either is within 80 dB of the louder's peak."""
    A, B = logmel(a), logmel(b)
    n = min(A.shape[1], B.shape[1])
    A, B = A[:, :n], B[:, :n]
    floor = max(A.max(), B.max()) - 80
    m = (A > floor) | (B > floor)
    return float(np.abs(np.maximum(A, floor) - np.maximum(B, floor))[m].mean())


def cos(a, b):
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load(split, k):
    x, sr = sf.read(f"{NS[split]}/audio/{k}.wav", dtype="float32")
    assert sr == SR
    return x


# probes, as in task_nsynth_probe.py
meta_v = json.load(open(f"{NS['valid']}/examples.json"))
by = collections.defaultdict(list)
for k, v in meta_v.items():
    by[v["instrument_family_str"]].append(k)
train = [(k, f) for f in sorted(by) for k in rng.permutation(by[f])[:N_TRAIN]]
X = {"panns": [], "ast": []}
for k, _ in train:
    e = embed(load("valid", k))
    for m in X:
        X[m].append(e[m])
yl = [f for _, f in train]
probe = {}
for m in X:
    sc = StandardScaler().fit(X[m])
    probe[m] = (
        sc,
        LogisticRegression(max_iter=4000, C=0.5).fit(sc.transform(X[m]), yl),
    )
print("probes trained on", len(yl), flush=True)

meta = json.load(open(f"{NS['test']}/examples.json"))
inst = collections.defaultdict(dict)
for k, v in meta.items():
    inst[(v["instrument_str"], v["velocity"])][v["pitch"]] = k
pairs = sorted((d[p], d[p + 2]) for d in inst.values() for p in d if p + 2 in d)
pairs = [pairs[i] for i in rng.permutation(len(pairs))[:N_PAIRS]]

done = (
    {(r["src"], r["tgt"]) for r in map(json.loads, open(OUT))}
    if os.path.exists(OUT)
    else set()
)
with open(OUT, "a") as fout:
    for lo, hi in pairs:
        for src, tgt, cents in ((lo, hi, 200), (hi, lo, -200)):
            if (src, tgt) in done:
                continue
            x, t = load("test", src), load("test", tgt)
            versions = {"real": t, "orig": x}
            for tool in TOOLS:
                y, _ = transforms.pitch(x, SR, cents, tool)
                versions[tool] = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
            E = {v: embed(a) for v, a in versions.items()}
            row = {"src": src, "tgt": tgt, "cents": cents, "family": meta[src]["instrument_family_str"],
                   "source": meta[src]["instrument_source_str"], "pitch": meta[src]["pitch"]}  # fmt: skip
            for v, a in versions.items():
                row[v] = {
                    "pred": {
                        m: str(pr.predict(sc.transform([E[v][m]]))[0])
                        for m, (sc, pr) in probe.items()
                    },
                    "cos": {m: cos(E[v][m], E["real"][m]) for m in probe},
                    "mel": mel_dist(a, t),
                }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
print("done", flush=True)
