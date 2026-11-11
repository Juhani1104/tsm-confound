"""Model-free damage of each tool, measured on what models see, plus a window-length test.

usage: python src/artifacts2.py N > results/artifacts2.txt

Fixes artifacts.py, whose log-spectral distance was dominated by near-empty bins above
the resampling band edge. Here every comparison is on a 64-band log-mel spectrogram from
50 Hz to 8 kHz, floored at 80 dB below the clip's peak, the input range of most music
models. Measures on N FMA-small clips (10 s, 32 kHz):
round trip   change then inverse with the same tool; mean absolute log-mel difference
             (dB) to the input after the best alignment within +-20 ms
onset peak   mean height of the top onset-strength peaks after one change, relative to
             the clean clip
The window test runs a phase-vocoder pitch shift with n_fft 256 to 4096 (hop n_fft/4),
to check whether window length, not the method's name, sets the onset smearing.
"""

import io
import json
import os
import sys
import zipfile

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms  # noqa: E402
from paths import FMA_ZIP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMA = FMA_ZIP
SR = 32000
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
PITCH = [10, 50, 200]
TEMPO = [1.05, 1.10, 1.20]
WINDOWS = [256, 512, 1024, 2048, 4096]
MAXLAG = int(0.02 * SR)


def clips(n):
    """n mono 10 s clips from the middle of FMA-small tracks."""
    z = zipfile.ZipFile(FMA)
    names = [m for m in z.namelist() if m.endswith(".mp3")]
    out = []
    for m in np.random.default_rng(0).permutation(names):
        try:
            x, sr = sf.read(io.BytesIO(z.read(m)), dtype="float32", always_2d=True)
        except Exception:
            continue
        x = librosa.resample(x.mean(1), orig_sr=sr, target_sr=SR, res_type="soxr_hq")
        if len(x) < 12 * SR or np.abs(x).max() < 1e-3:
            continue
        mid = len(x) // 2
        out.append((os.path.basename(m), x[mid - 5 * SR : mid + 5 * SR]))
        if len(out) == n:
            break
    return out


def logmel(x):
    """64-band log-mel (dB) from 50 Hz to 8 kHz, floored 80 dB under the peak."""
    m = librosa.feature.melspectrogram(
        y=x, sr=SR, n_fft=1024, hop_length=320, n_mels=64, fmin=50, fmax=8000
    )
    return librosa.power_to_db(m, ref=np.max, top_db=80)


def mel_damage(ref, y):
    """Mean absolute log-mel difference (dB) of y against ref after the best +-20 ms lag."""
    n = min(len(ref), len(y))
    c = np.correlate(ref[MAXLAG : n - MAXLAG], y[:n], "valid")
    y = np.roll(y[:n], int(np.argmax(c)) - MAXLAG)
    a, b = logmel(ref[MAXLAG : n - MAXLAG]), logmel(y[MAXLAG : n - MAXLAG])
    k = min(a.shape[1], b.shape[1])
    return float(np.mean(np.abs(a[:, :k] - b[:, :k])))


def onset_peaks(x):
    """Mean height of the top 50 peaks of the onset-strength envelope."""
    env = librosa.onset.onset_strength(y=x, sr=SR)
    pk = librosa.util.peak_pick(
        env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=5
    )
    return float(np.sort(env[pk])[-50:].mean()) if len(pk) else 0.0


def pv_pitch(x, cents, n_fft):
    """torchaudio phase-vocoder pitch shift with a chosen window length."""
    return transforms._torchaudio_pitch(x, SR, cents, n_fft=n_fft)


rows, win = [], []
for name, x in clips(N):
    p0 = onset_peaks(x)
    for tool in transforms.TOOLS:
        for kind, amounts in (("pitch", PITCH), ("tempo", TEMPO)):
            f = transforms.pitch if kind == "pitch" else transforms.tempo
            for a in amounts:
                y, _ = f(x, SR, a, tool)
                back, _ = f(y, SR, -a if kind == "pitch" else 1 / a, tool)
                rows.append(
                    dict(
                        clip=name,
                        tool=tool,
                        kind=kind,
                        amount=a,
                        mel_db=mel_damage(x, back),
                        onset_ratio=onset_peaks(y) / (p0 + 1e-9),
                    )
                )
    for n_fft in WINDOWS:
        y = pv_pitch(x, 50, n_fft)
        win.append(
            dict(
                clip=name,
                n_fft=n_fft,
                onset_ratio=onset_peaks(y) / (p0 + 1e-9),
                mel_db=mel_damage(x, pv_pitch(y, -50, n_fft)),
            )
        )
    print(name, flush=True)
json.dump(
    {"tools": rows, "windows": win},
    open(f"{ROOT}/results/artifacts2.json", "w"),
    indent=0,
)

print("\ntool        kind   amount  roundtrip_mel_dB  onset_peak_ratio")
for tool in transforms.TOOLS:
    for kind, amounts in (("pitch", PITCH), ("tempo", TEMPO)):
        for a in amounts:
            r = [
                v
                for v in rows
                if v["tool"] == tool and v["kind"] == kind and v["amount"] == a
            ]
            print(
                f"{tool:11s} {kind:5s} {a:7.2f}  {np.mean([v['mel_db'] for v in r]):10.2f}  {np.mean([v['onset_ratio'] for v in r]):10.3f}"
            )
print("\nphase vocoder pitch shift 50 cents by window length")
for n_fft in WINDOWS:
    r = [v for v in win if v["n_fft"] == n_fft]
    print(
        f"n_fft {n_fft:5d} ({1000 * n_fft / SR:6.1f} ms)  onset ratio {np.mean([v['onset_ratio'] for v in r]):.3f}  roundtrip mel {np.mean([v['mel_db'] for v in r]):.2f} dB"
    )
