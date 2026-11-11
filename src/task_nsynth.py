"""Instrument family recognition on NSynth test notes, zero-shot from AudioSet labels.

usage: python src/task_nsynth.py N

N notes from the NSynth test split (4 s, 16 kHz), balanced over the 11 instrument
families. PANNs Cnn14 and AST output AudioSet probabilities; each family is scored as
the highest probability among the AudioSet classes mapped to it, and the prediction is
the best-scoring family. No training, so nothing is fitted to clean audio. A pitch or
tempo change does not change the instrument, so every condition shares the clean label.
Rows go to results/nsynth.jsonl.
"""

import csv
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from panns_inference.models import Cnn14
from transformers import ASTFeatureExtractor, ASTForAudioClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from paths import AST, NSYNTH, PANNS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
N = int(sys.argv[1]) if len(sys.argv) > 1 else 440
OUT = f"{ROOT}/results/nsynth.jsonl"
FAMILY = {
    "bass": ["Bass guitar", "Double bass"],
    "brass": ["Brass instrument", "Trumpet", "Trombone", "French horn"],
    "flute": ["Flute"],
    "guitar": ["Guitar", "Acoustic guitar", "Electric guitar"],
    "keyboard": ["Piano", "Keyboard (musical)", "Electric piano"],
    "mallet": ["Marimba, xylophone", "Vibraphone", "Glockenspiel", "Mallet percussion"],
    "organ": ["Organ", "Electronic organ", "Hammond organ"],
    "reed": ["Saxophone", "Clarinet", "Wind instrument, woodwind instrument"],
    "string": ["Bowed string instrument", "Violin, fiddle", "Cello", "String section"],
    "synth_lead": ["Synthesizer"],
    "vocal": ["Singing", "Choir", "Vocal music"],
}

labels = [
    r[2]
    for r in csv.reader(
        open(os.path.expanduser("~/panns_data/class_labels_indices.csv"))
    )
][1:]
IDX = {f: [labels.index(c) for c in cs] for f, cs in FAMILY.items()}

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
assert [ast.config.id2label[i] for i in range(527)] == labels, (
    "AST and PANNs label orders differ"
)


@torch.no_grad()
def probs(x):
    """{model: 527 AudioSet probabilities} for mono 16 kHz x."""
    y = librosa.resample(x, orig_sr=SR, target_sr=32000, res_type="soxr_hq")
    p = (
        panns(torch.from_numpy(y[None].astype("float32")).cuda(), None)[
            "clipwise_output"
        ][0]
        .cpu()
        .numpy()
    )
    f = ast_fe(x, sampling_rate=SR, return_tensors="pt")["input_values"].cuda()
    q = torch.sigmoid(ast(f).logits)[0].cpu().numpy()
    return {"panns": p, "ast": q}


def family(p):
    """Predicted instrument family from AudioSet probabilities."""
    return max(IDX, key=lambda f: p[IDX[f]].max())


# extracted once with tarfile's "data" filter; a .tar.gz cannot be read at random
NS = f"{NSYNTH}/nsynth-test"
meta = json.load(open(f"{NS}/examples.json"))
rng = np.random.default_rng(0)
by = {}
for k, v in meta.items():
    by.setdefault(v["instrument_family_str"], []).append(k)
pick = [k for f in sorted(by) for k in rng.permutation(by[f])[: N // len(by)]]
COND = harness.conditions()

done = {json.loads(l)["note"] for l in open(OUT)} if os.path.exists(OUT) else set()
with open(OUT, "a") as fout:
    for k in pick:
        if k in done:
            continue
        x, sr = sf.read(f"{NS}/audio/{k}.wav", dtype="float32")
        assert sr == SR
        sg = harness.signs(rng)
        row = {
            "note": k,
            "family": meta[k]["instrument_family_str"],
            "pitch": meta[k]["pitch"],
        }
        for cond in COND:
            y, _, _ = harness.apply(x, SR, cond, sg.get((cond[0], cond[1]), 1))
            p = probs(y)
            row[harness.key(cond)] = {m: family(v) for m, v in p.items()}
        fout.write(json.dumps(row) + "\n")
        fout.flush()
print("done", flush=True)
