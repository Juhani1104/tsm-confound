"""Instrument family recognition on NSynth with a linear probe, under every condition.

usage: python src/task_nsynth_probe.py N_TRAIN N_TEST

The zero-shot version (task_nsynth.py) sat near 25% accuracy, too close to the floor to
show any change. Here a logistic probe per model is trained on clean embeddings of
N_TRAIN notes per family from the NSynth valid split, and tested on N_TEST notes per
family from the test split under all conditions. Embeddings: PANNs Cnn14 (2048-d) and
AST pooled output (768-d). Rows go to results/nsynth_probe.jsonl.
"""

import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from panns_inference.models import Cnn14
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import ASTFeatureExtractor, ASTForAudioClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from paths import AST, NSYNTH, PANNS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 150
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 40
OUT = f"{ROOT}/results/nsynth_probe{harness.TAG}.jsonl"
NS = {s: f"{NSYNTH}/nsynth-{s}" for s in ("valid", "test")}

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


def pick(split, per):
    """Up to `per` note ids per instrument family from an NSynth split."""
    meta = json.load(open(f"{NS[split]}/examples.json"))
    by = {}
    for k, v in meta.items():
        by.setdefault(v["instrument_family_str"], []).append(k)
    return [(k, f) for f in sorted(by) for k in rng.permutation(by[f])[:per]], meta


def load(split, k):
    x, sr = sf.read(f"{NS[split]}/audio/{k}.wav", dtype="float32")
    assert sr == SR
    return x


train, _ = pick("valid", N_TRAIN)
X = {"panns": [], "ast": []}
y = []
for k, f in train:
    e = embed(load("valid", k))
    for m in X:
        X[m].append(e[m])
    y.append(f)
probe = {}
for m in X:
    sc = StandardScaler().fit(X[m])
    probe[m] = (sc, LogisticRegression(max_iter=4000, C=0.5).fit(sc.transform(X[m]), y))
print("train notes", len(y), "families", sorted(set(y)), flush=True)

test, meta = pick("test", N_TEST)
COND = harness.conditions()
done = {json.loads(l)["note"] for l in open(OUT)} if os.path.exists(OUT) else set()
with open(OUT, "a") as fout:
    for k, fam in test:
        if k in done:
            continue
        x = load("test", k)
        sg = harness.signs(rng)
        row = {"note": k, "family": fam, "pitch": meta[k]["pitch"]}
        for cond in COND:
            z, _, _ = harness.apply(x, SR, cond, sg.get((cond[0], cond[1]), 1))
            e = embed(z)
            row[harness.key(cond)] = {
                m: str(pr.predict(sc.transform([e[m]]))[0])
                for m, (sc, pr) in probe.items()
            }
        fout.write(json.dumps(row) + "\n")
        fout.flush()
print("done", flush=True)
