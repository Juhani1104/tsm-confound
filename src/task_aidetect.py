"""AI-music detection under every tool and condition: does the tool make real music look fake?

usage: CUDA_VISIBLE_DEVICES= python src/task_aidetect.py N

N real FMA-small tracks from the Deezer detector's held-out test split (12 s from the middle,
mono 44.1 kHz) and, as the AI class, the same clips through EnCodec 48 kHz at 6 kbps (an
autoencoder like those the detector was trained to catch). Each version gets every harness
condition (pitch 10/50/200 cents, tempo 5/10/20%, tools librosa, torchaudio, signalsmith,
rubberband, rbsmall, resample; one random sign per track shared by its real and fake
version and by all tools). Detectors (deezer.py, artifactnet.py): Deezer's released amplitude model
(P(real), mean over 64x64 patches) and ArtifactNet v9.4 (P(AI), median over 4 s chunks; a NaN
chunk is rescored at -6 dB). Runs on CPU so the GPU stays free. Rows to results/aidetect.jsonl.
"""

import json
import os
import sys

import numpy as np
import torch
from encodec import EncodecModel
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifactnet  # noqa: E402
import deezer  # noqa: E402
import fma_test  # noqa: E402
import harness  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 44100
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = f"{ROOT}/results/aidetect.jsonl"
TOOLS = ("librosa", "torchaudio", "signalsmith", "rubberband", "rbsmall", "resample")
COND = harness.conditions(tools=TOOLS)
torch.set_num_threads(8)
codec = EncodecModel.encodec_model_48khz().eval()
codec.set_target_bandwidth(6.0)
rng = np.random.default_rng(0)


def encodec48(x):
    """Mono x through EnCodec 48 kHz (stereo model, both channels the same) at 6 kbps."""
    y = resample_poly(np.c_[x, x], 320, 294, axis=0).astype("float32").T
    with torch.no_grad():
        z = codec(torch.from_numpy(y.copy())[None])[0, :, : y.shape[1]].numpy()
    return resample_poly(z.T, 294, 320, axis=0).astype("float32")[: len(x)].mean(1)


def anet(x):
    """(ArtifactNet P(AI) median over chunks, number of NaN chunks rescored at -6 dB)."""
    p = artifactnet.p_ai(x)
    bad = np.isnan(p)
    if bad.any():
        q = artifactnet.p_ai(0.5 * x)
        p[bad] = q[bad]
    return float(np.nanmedian(p)) if np.isfinite(p).any() else None, int(bad.sum())


def score(x):
    """Both detectors' scores for one clip."""
    a, nan = anet(x)
    return {
        "deezer": float(deezer.p_real(x, patch=(64, 64)).mean()),
        "anet": a,
        "anet_nan": nan,
    }


if __name__ == "__main__":
    done = {json.loads(l)["track"] for l in open(OUT)} if os.path.exists(OUT) else set()
    names = fma_test.test_tracks(N)
    with open(OUT, "a") as fout:
        for name in names:
            if name in done:
                continue
            x = fma_test.load(name, seconds=12.0).mean(1).astype(np.float32)
            x = x / max(1.0, np.abs(x).max() / 0.99)
            fake = encodec48(x)
            sg = harness.signs(rng)
            row = {"track": name}
            for cond in COND:
                s = sg.get((cond[0], cond[1]), 1)
                yr, _, _ = harness.apply(x, SR, cond, s)
                yf, _, _ = harness.apply(fake, SR, cond, s)
                row[harness.key(cond)] = {"real": score(yr), "fake": score(yf)}
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            c = row["clean"]
            print(
                name,
                "clean deezer real/fake",
                round(c["real"]["deezer"], 2),
                round(c["fake"]["deezer"], 2),
                "| pitch50 librosa/signalsmith real",
                round(row["pitch50_librosa"]["real"]["deezer"], 2),
                round(row["pitch50_signalsmith"]["real"]["deezer"], 2),
                flush=True,
            )
    print("done", flush=True)
