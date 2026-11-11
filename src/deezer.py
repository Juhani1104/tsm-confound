"""Deezer's released amplitude detector (Afchar et al., ICASSP 2025), with its eval preprocessing.

Preprocessing follows the repository's EvalAugmenter for conf specnn_amplitude:
44.1 kHz, mean of the two channels, STFT window 2048 / hop 512 / n_fft 2048,
log10 power clipped to [1e-10, 1e6], (x + 4) / 3, bins below 16 kHz kept.
The output "deepfake" is the probability of REAL: the training code labels real
tracks 1 and autoencoded tracks 0.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tf_keras

from paths import DEEZER

SR = 44100
SLICE = int(0.78 * SR)
HF_BINS = int(16000 * 2 / SR * 1025)
_model = None


def model():
    """The Keras model, loaded once; fully convolutional, so any input size works."""
    global _model
    if _model is None:
        _model = tf_keras.models.load_model(str(DEEZER), compile=False)
    return _model


def features(x):
    """Normalised log-power spectrogram (frames, bins, 1) of mono 44.1 kHz audio x."""
    s = tf.signal.stft(tf.constant(x, tf.float32), 2048, 512, fft_length=2048)
    p = tf.clip_by_value(tf.square(tf.abs(s)), 1e-10, 1e6)
    d = tf.math.log(p) / tf.math.log(10.0)
    d = (d + 4.0) / 3.0
    return d[:, :HF_BINS, None].numpy()


def p_real(x, hop=SLICE, patch=None):
    """P(real) for every 0.78 s window of x; patch=(t, f) scores 64x64 crops instead."""
    starts = range(0, max(len(x) - SLICE, 0) + 1, hop)
    feats = [features(x[s : s + SLICE]) for s in starts]
    if patch:
        t, f = patch
        feats = [
            z[:t, k : k + f] for z in feats for k in range(0, z.shape[1] - f + 1, f)
        ]
    out = model()(np.stack(feats), training=False)
    return np.asarray(out[0] if isinstance(out, (list, tuple)) else out).ravel()
