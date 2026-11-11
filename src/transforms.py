"""One interface for pitch and tempo changes made by different tools.

Every function takes mono float32 audio x at rate sr and returns (y, time_scale): the
transformed audio and the factor that maps an event time in x to its time in y.
A tool either changes pitch at fixed duration, tempo at fixed pitch, or both at once.

tools
    librosa      librosa phase vocoder (effects.pitch_shift / time_stretch)
    torchaudio   torchaudio phase vocoder: pitch_shift's recipe (n_fft 512, resampled with soxr
                 instead of its own resampler) and functional.phase_vocoder for tempo
    signalsmith  signalsmith-stretch (python-stretch), audiomentations >= 0.39 default
    rubberband   Rubber Band library through Spotify's pedalboard (time_stretch defaults)
    rbsmall      the same Rubber Band engine and options through pylibrb, fed 4096-sample
                 blocks. pedalboard feeds the R3 engine its largest allowed block, and with
                 large blocks R3 output drifts in time by about 2 ms per second
                 (src/rb_offset.py); small blocks do not.
    resample     playback-rate change: pitch and tempo move together, no time-scale
                 modification at all, the reference for a "true" change
"""

import librosa
import numpy as np
import pedalboard
import pylibrb
import python_stretch
import torch
import torchaudio

TOOLS = ("librosa", "torchaudio", "signalsmith", "rubberband", "rbsmall", "resample")


def pitch(x, sr, cents, tool):
    """Shift x by `cents`; resample also speeds up by the same ratio."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    steps = cents / 100
    if tool == "librosa":
        return librosa.effects.pitch_shift(
            x, sr=sr, n_steps=steps, res_type="soxr_hq"
        ), 1.0
    if tool == "torchaudio":
        return _torchaudio_pitch(x, sr, cents), 1.0
    if tool == "signalsmith":
        s = python_stretch.Signalsmith.Stretch()
        s.preset(1, sr)
        s.setTransposeSemitones(steps)
        return s.process(x[None])[0].astype(np.float32), 1.0
    if tool == "rubberband":
        y = pedalboard.time_stretch(
            x[None], sr, stretch_factor=1.0, pitch_shift_in_semitones=steps
        )
        return y[0].astype(np.float32), 1.0
    if tool == "rbsmall":
        return _rubberband(x, sr, 1.0, 2 ** (cents / 1200)), 1.0
    if tool == "resample":
        r = 2 ** (cents / 1200)
        return _rate(x, sr, r), 1 / r
    raise ValueError(tool)


def tempo(x, sr, factor, tool):
    """Play x `factor` times faster (>1 is shorter); resample also raises the pitch."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if tool == "librosa":
        return librosa.effects.time_stretch(x, rate=factor), 1 / factor
    if tool == "torchaudio":
        n_fft, hop = 2048, 512
        win = torch.hann_window(n_fft)
        spec = torch.stft(
            torch.from_numpy(x), n_fft, hop, window=win, return_complex=True
        )
        phase = torch.linspace(0, np.pi * hop, spec.shape[-2])[..., None]
        st = torchaudio.functional.phase_vocoder(spec, factor, phase)
        y = torch.istft(st, n_fft, hop, window=win, length=int(round(len(x) / factor)))
        return y.numpy().astype(np.float32), 1 / factor
    if tool == "signalsmith":
        s = python_stretch.Signalsmith.Stretch()
        s.preset(1, sr)
        s.setTimeFactor(factor)
        return s.process(x[None])[0].astype(np.float32), 1 / factor
    if tool == "rubberband":
        y = pedalboard.time_stretch(x[None], sr, stretch_factor=factor)
        return y[0].astype(np.float32), 1 / factor
    if tool == "rbsmall":
        return _rubberband(x, sr, 1 / factor, 1.0), 1 / factor
    if tool == "resample":
        return _rate(x, sr, factor), 1 / factor
    raise ValueError(tool)


def _torchaudio_pitch(x, sr, cents, n_fft=512):
    """torchaudio's pitch_shift recipe (phase-vocoder stretch, then resample), n_fft 512 hop n_fft/4
    as its defaults. Its own resampler builds a kernel from the gcd of the two rates, which for
    fractional ratios grows past 12 GB, so the resample step uses soxr like the other tools."""
    hop = n_fft // 4
    rate = 2 ** (-cents / 1200)
    win = torch.hann_window(n_fft)
    spec = torch.stft(torch.from_numpy(x), n_fft, hop, window=win, return_complex=True)
    phase = torch.linspace(0, np.pi * hop, spec.shape[-2])[..., None]
    st = torchaudio.functional.phase_vocoder(spec, rate, phase)
    y = torch.istft(
        st, n_fft, hop, window=win, length=int(round(len(x) / rate))
    ).numpy()
    y = _rate(y, sr, 1 / rate)
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)].astype(np.float32)


def _rubberband(x, sr, time_ratio, pitch_scale, block=4096):
    """Rubber Band R3 offline with pedalboard's default options, studied and processed in
    `block`-sample chunks."""
    O = pylibrb.Option
    opts = (O.PROCESS_OFFLINE | O.THREADING_NEVER | O.CHANNELS_TOGETHER | O.PitchHighQuality | O.ENGINE_FINER
            | O.TRANSIENTS_CRISP | O.DETECTOR_COMPOUND | O.PHASE_LAMINAR | O.WINDOW_STANDARD
            | O.SMOOTHING_OFF | O.FORMANT_PRESERVED)  # fmt: skip
    st = pylibrb.RubberBandStretcher(int(sr), 1, opts, time_ratio, pitch_scale)
    st.set_logging_level(0)
    a = x[None]
    st.set_expected_input_duration(len(x))
    st.set_max_process_size(block)
    for i in range(0, len(x), block):
        st.study(np.ascontiguousarray(a[:, i : i + block]), final=i + block >= len(x))
    out = []
    for i in range(0, len(x), block):
        st.process(np.ascontiguousarray(a[:, i : i + block]), final=i + block >= len(x))
        out.append(st.retrieve_available()[0])
    while not st.is_done():
        out.append(st.retrieve_available()[0])
    return np.concatenate(out).astype(np.float32)


def _rate(x, sr, r):
    """Playback r times faster: read x as if recorded at sr * r."""
    return librosa.resample(x, orig_sr=sr * r, target_sr=sr, res_type="soxr_hq").astype(
        np.float32
    )
