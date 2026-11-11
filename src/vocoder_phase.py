"""Is the phase vocoder's time displacement caused by lost vertical phase coherence?

usage: python src/vocoder_phase.py

Each bin of librosa's phase vocoder accumulates its own phase; across the bins of one
partial the relative phases (which place the energy inside the analysis window) are lost,
most of all where a sound starts after silence, so a whole note can be displaced by up to
half a window. Prediction: on smooth Gaussian tones (no transients) the displacement is
zero-mean with a spread that grows with n_fft, and identity phase locking (Laroche &
Dolson 1999: every bin takes its phase advance from the nearest spectral peak, keeping the
original phase offset to it) removes most of it. Same STFT, same magnitudes, only the
phase rule differs.
"""

import librosa
import numpy as np
import scipy.signal

SR = 16000
t0 = np.arange(0.5, 39.0, 0.6)
tt = np.arange(int(40 * SR)) / SR
rng = np.random.default_rng(0)
f0 = rng.uniform(200, 1000, len(t0))
g = sum(
    (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * 2 * f * tt))
    * np.exp(-0.5 * ((tt - t) / 0.05) ** 2)
    for t, f in zip(t0, f0)
).astype(np.float32)


def centroids(y, t0):
    """Energy centroid (s) of the tone near each time in t0, within +-250 ms."""
    out = []
    for t in t0:
        a, b = int((t - 0.25) * SR), int((t + 0.25) * SR)
        e = y[max(a, 0) : b].astype(np.float64) ** 2
        out.append((max(a, 0) + (np.arange(len(e)) * e).sum() / e.sum()) / SR)
    return np.array(out)


def stretch(x, rate, n_fft, locked):
    """librosa-style phase vocoder stretch; locked=True uses identity phase locking."""
    hop = n_fft // 4
    D = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    steps = np.arange(0, D.shape[1], rate)
    omega = 2 * np.pi * hop * np.arange(D.shape[0]) / n_fft
    Dp = np.pad(D, [(0, 0), (0, 2)])
    acc = np.angle(D[:, 0])
    out = np.zeros((D.shape[0], len(steps)), complex)
    for t, s in enumerate(steps):
        c0, c1 = Dp[:, int(s)], Dp[:, int(s) + 1]
        a = s % 1
        mag = (1 - a) * np.abs(c0) + a * np.abs(c1)
        if locked:
            # identity phase locking: peaks keep their own accumulated phase, other bins keep
            # their analysis phase offset to the peak whose region they belong to
            peaks, _ = scipy.signal.find_peaks(mag)
            if len(peaks):
                owner = peaks[
                    np.clip(
                        np.searchsorted(
                            (peaks[:-1] + peaks[1:]) / 2, np.arange(len(mag))
                        ),
                        0,
                        len(peaks) - 1,
                    )
                ]
                ph = acc[owner] + np.angle(c0) - np.angle(c0[owner])
            else:
                ph = acc
            out[:, t] = mag * np.exp(1j * ph)
        else:
            out[:, t] = mag * np.exp(1j * acc)
        dphi = np.angle(c1) - np.angle(c0) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        acc = acc + omega + dphi
    return librosa.istft(out, hop_length=hop, length=int(round(len(x) / rate)))


if __name__ == "__main__":
    ref = centroids(g, t0)
    print(
        f"{'rate':>6s} {'n_fft':>6s} {'rule':>10s} {'mean lag ms':>12s} {'sd ms':>7s} {'|lag|>20ms':>11s}"
    )
    for rate in (0.9942, 0.9715, 0.8909, 1.1):
        for n_fft in (1024, 2048, 4096):
            for locked in (False, True):
                y = stretch(g, rate, n_fft, locked)
                l = (centroids(y, t0 / rate) - ref / rate) * 1000
                print(
                    f"{rate:6.4f} {n_fft:6d} {'locked' if locked else 'librosa':>10s} {l.mean():+12.1f} {l.std():7.1f} {np.mean(np.abs(l) > 20):11.2f}",
                    flush=True,
                )

    # Transient test: the decaying bursts of rb_offset.py (sharp attack). A detector firing on
    # the rising edge shows the systematic early bias seen on piano; does phase locking remove it?
    import os  # noqa: E402
    import sys  # noqa: E402

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rb_offset import burst_lags, bursts  # noqa: E402

    xb, tb = bursts(SR, dur=40.0)
    print("\nbursts (sharp attack): half-peak onset lag")
    print(f"{'rate':>6s} {'n_fft':>6s} {'rule':>10s} {'median ms':>10s} {'sd ms':>7s}")
    for rate in (0.9715, 1.1):
        for n_fft in (1024, 2048):
            for locked in (False, True):
                y = stretch(xb, rate, n_fft, locked)
                l = burst_lags(y, SR, tb / rate)
                print(
                    f"{rate:6.4f} {n_fft:6d} {'locked' if locked else 'librosa':>10s} {np.median(l):+10.1f} {np.std(l):7.1f}",
                    flush=True,
                )
