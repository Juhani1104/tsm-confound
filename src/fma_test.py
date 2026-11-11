"""FMA-small tracks from the Deezer detector's held-out test split, and a loader for them."""

import io
import os
import sys
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_npy  # noqa: E402
from paths import DEEZER_SPLIT as SPLIT  # noqa: E402
from paths import FMA_ZIP as FMA  # noqa: E402

SR = 44100


def test_tracks(n, seed=0):
    """n zip member names of FMA-small tracks in the detector's held-out test split."""
    test = set(map(str, safe_npy.load(SPLIT).item()["test"]))
    names = sorted(
        m for m in zipfile.ZipFile(FMA).namelist() if os.path.basename(m) in test
    )
    return list(np.random.default_rng(seed).permutation(names)[:n])


def load(name, seconds=10.0):
    """Stereo float32 (samples, 2) at 44.1 kHz: the middle `seconds` of an FMA track."""
    x, sr = sf.read(
        io.BytesIO(zipfile.ZipFile(FMA).read(name)), dtype="float32", always_2d=True
    )
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    if sr != SR:
        x = resample_poly(x, SR, sr, axis=0).astype("float32")
    n = int(seconds * SR)
    s = max(0, len(x) // 2 - n // 2)
    return x[s : s + n]
