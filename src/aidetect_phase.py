"""Why does a phase vocoder hide EnCodec's traces from the Deezer detector?

usage: CUDA_VISIBLE_DEVICES= python src/aidetect_phase.py N

Resampling moves every frequency yet leaves detection intact (results/summary_aidetect.txt),
so the detector does not rely on fixed-frequency peaks. Hypothesis: it relies on fine
time-frequency texture that the vocoder's lost vertical phase coherence scrambles. Causal
test at +-10 cents on N FMA test tracks (real and EnCodec 6 kbps): pitch shift by the same
STFT phase vocoder (n_fft 2048, hop 512, then soxr resample) with librosa's per-bin phase
rule vs identity phase locking (src/vocoder_phase.py), plus librosa and signalsmith as
references. If locking restores detection, phase incoherence is the cause. The window
size is varied too (1024, 4096) to see how much of the rest is time resolution.
"""

import os
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deezer  # noqa: E402
import fma_test  # noqa: E402
import transforms  # noqa: E402
from task_aidetect import encodec48  # noqa: E402  (loads EnCodec on CPU)
from vocoder_phase import stretch  # noqa: E402

SR = 44100
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = np.random.default_rng(1)


def own_vocoder(x, cents, locked, n_fft=2048):
    """Pitch shift by our own phase vocoder: stretch by 1/r, then resample by r."""
    r = 2 ** (cents / 1200)
    y = stretch(x, 1 / r, n_fft, locked)
    y = librosa.resample(y, orig_sr=SR / r, target_sr=SR, res_type="soxr_hq")
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)].astype(np.float32)


versions = {
    "clean": lambda x, c: x,
    "librosa": lambda x, c: transforms.pitch(x, SR, c, "librosa")[0],
    "vocoder, per-bin phase": lambda x, c: own_vocoder(x, c, False),
    "vocoder, phase-locked": lambda x, c: own_vocoder(x, c, True),
    "per-bin, n_fft 1024": lambda x, c: own_vocoder(x, c, False, 1024),
    "phase-locked, n_fft 1024": lambda x, c: own_vocoder(x, c, True, 1024),
    "phase-locked, n_fft 4096": lambda x, c: own_vocoder(x, c, True, 4096),
    "signalsmith": lambda x, c: transforms.pitch(x, SR, c, "signalsmith")[0],
}
scores = {v: ([], []) for v in versions}
for name in fma_test.test_tracks(N, seed=0):
    x = fma_test.load(name, seconds=12.0).mean(1).astype(np.float32)
    x = x / max(1.0, np.abs(x).max() / 0.99)
    fake = encodec48(x)
    c = 10 * int(rng.choice([-1, 1]))
    for v, f in versions.items():
        scores[v][0].append(1 - float(deezer.p_real(f(x, c), patch=(64, 64)).mean()))
        scores[v][1].append(1 - float(deezer.p_real(f(fake, c), patch=(64, 64)).mean()))
    print(name, flush=True)


def auc(pos, neg):
    """P(a positive scores above a negative); ties count half."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


print(
    f"\n{'version (+-10 cents)':26s} {'real flagged':>13s} {'fake flagged':>13s} {'AUC':>6s}"
)
for v, (r, f) in scores.items():
    print(
        f"{v:26s} {np.mean(np.array(r) > 0.5):13.2f} {np.mean(np.array(f) > 0.5):13.2f} {auc(f, r):6.3f}"
    )
