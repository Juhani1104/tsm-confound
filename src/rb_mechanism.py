"""Predict the Rubber Band R3 drift from the engine's own hop decisions, no audio analysis.

usage: python src/rb_mechanism.py

R3Stretcher::consume() (src/finer/R3Stretcher.cpp, v3.3 and v4.0) asks StretchCalculator::
calculateSingle() for an integer output hop ONCE per call and then processes every frame
that fits in the input buffer with that hop. calculateSingle() normally dithers the hop
(e.g. 127/128) so the mean matches inhop * ratio exactly, but it advances its frame
counters once per call, not once per frame. With small process() blocks each call covers a
few frames and the dither works; with a block as large as getProcessSizeLimit() one call
covers thousands of frames at one rounded hop, so the output time base drifts by
(outhop - inhop * R) / (inhop * R) per frame.

This script runs pylibrb (Rubber Band 3.3.0) at log level 2 on white noise, reads every
consume() call's outhop and the number of frames it emitted, and integrates the timing
error frame by frame into a predicted lag curve. It compares that with the measured curve
(results/bt_lagcurves.npz, pedalboard on click trains) for Beat This!'s settings (44.1 kHz,
tempo -20..+20 %), and prints the 16 kHz pitch-shift slopes against rb_offset.py.
"""

import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = r"""
import sys, numpy as np, pylibrb
O = pylibrb.Option
sr, time_ratio, pitch, dur = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
x = np.random.default_rng(0).standard_normal(int(sr * dur)).astype(np.float32) * 0.1
opts = O.PROCESS_OFFLINE | O.THREADING_NEVER | O.CHANNELS_TOGETHER | O.PitchHighQuality | O.ENGINE_FINER | O.FORMANT_PRESERVED
st = pylibrb.RubberBandStretcher(sr, 1, opts, time_ratio, pitch)
st.set_logging_level(2)
a = x[None]
st.set_expected_input_duration(len(x))
B = st.get_process_size_limit()
st.set_max_process_size(B)
for i in range(0, len(x), 4096):
    st.study(np.ascontiguousarray(a[:, i:i + 4096]), final=i + 4096 >= len(x))
for i in range(0, len(x), B):
    st.process(np.ascontiguousarray(a[:, i:i + B]), final=i + B >= len(x)); st.retrieve_available()
"""


def hops(sr, time_ratio, pitch, dur):
    """[(outhop, frames)] per consume() call, the inhop, and samples emitted per frame scale."""
    log = subprocess.run([sys.executable, "-I", "-c", CHILD, str(sr), str(time_ratio), str(pitch), str(dur)],
                         capture_output=True, text=True).stderr  # fmt: skip
    inhop = int(
        re.search(
            r"PhaseAdvance: initial inhop and outhop: \((\d+), (\d+)\)", log
        ).group(1)
    )
    calls, cur = [], None
    for line in log.splitlines():
        m = re.search(r"consume: write space and outhop: \(\d+, (\d+)\)", line)
        if m:
            cur = [int(m.group(1)), 0]
            calls.append(cur)
        elif "consume: read space" in line and cur is not None:
            cur[1] += (
                1  # one per loop pass; the last pass of each call is the exit check
            )
    return [(o, max(n - 1, 0)) for o, n in calls], inhop


def predicted_curve(sr, time_ratio, pitch, dur):
    """(output time s, lag ms) after every frame, from the per-call outhops."""
    calls, inhop = hops(sr, time_ratio, pitch, dur)
    exact = (
        inhop * time_ratio
    )  # output samples per input hop, after resampling by 1/pitch
    t, lag, out = [], [], 0.0
    err = 0.0
    for o, n in calls:
        for _ in range(n):
            emitted = o / pitch  # synthesis hop, then resampled by the pitch scale
            err += emitted - exact
            out += emitted
            t.append(out / sr)
            lag.append(err / sr * 1000)
    return np.array(t), np.array(lag), inhop, sorted({o for o, _ in calls}), len(calls)


if __name__ == "__main__":
    curves = dict(np.load(f"{RESULTS}/bt_lagcurves.npz"))
    print("44.1 kHz tempo (Beat This! settings): predicted vs measured lag, ms")
    print(
        f"{'pct':>4s} {'inhop':>5s} {'outhops':>12s} {'calls':>5s}   at 30 s       60 s      120 s      200 s     rms diff"
    )
    for s in [p for p in range(-20, 21, 4) if p]:
        f = 1 + s / 100
        t, lag, inhop, outs, ncalls = predicted_curve(44100, 1 / f, 1.0, 240)
        mt, ml = curves[f"ts{s}_t"], curves[f"ts{s}_lag"]
        grid = np.arange(5, 200, 1.0)
        p_, m_ = np.interp(grid, t, lag), np.interp(grid, mt, ml)
        cells = " ".join(
            f"{np.interp(g, t, lag):+5.0f}/{np.interp(g, mt, ml):+5.0f}"
            for g in (30, 60, 120, 199)
        )
        print(
            f"{s:+4d} {inhop:5d} {str(outs):>12s} {ncalls:5d}   {cells}   {np.sqrt(np.mean((p_ - m_) ** 2)):6.1f}",
            flush=True,
        )
    print(
        "\n16 kHz pitch shift: predicted drift slope (ms/s) over the first 30 s; measured from rb_offset.py"
    )
    measured = {10: 2.08, -10: -2.05, 50: 2.86, -50: -2.14, 200: 0.30, -200: -3.14}
    for c, m in measured.items():
        t, lag, inhop, outs, ncalls = predicted_curve(16000, 1.0, 2 ** (c / 1200), 30)
        k = t < 30
        print(
            f"{c:+5d} cents inhop {inhop} outhops {outs}: predicted {np.polyfit(t[k], lag[k], 1)[0]:+.2f}  measured {m:+.2f}",
            flush=True,
        )
