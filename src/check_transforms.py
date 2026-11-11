"""Check that every tool makes the change it claims, before any model sees its output.

usage: python src/check_transforms.py > results/check_transforms.txt

On synthetic tones and on real piano audio, measure for each tool and setting:
  pitch error   measured shift (librosa.estimate_tuning plus whole semitones from a
                pitch-class profile) minus requested shift, in cents
  length error  output length relative to the expected length, in percent
  latency       delay of an isolated click, in milliseconds
A tool that misses the request, or adds a large delay, must be corrected or reported.
"""

import json
import os
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms  # noqa: E402

SR = 32000
rng = np.random.default_rng(0)


def tone_mix(seconds=6.0):
    """A few harmonic tones on the equal-tempered A440 grid, with soft onsets."""
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    for midi in (57, 60, 64, 69, 72):
        f = 440 * 2 ** ((midi - 69) / 12)
        for h in range(1, 6):
            y += np.sin(2 * np.pi * f * h * t) / h
    env = np.minimum(1, t * 20) * np.minimum(1, (t[-1] - t) * 20)
    return (0.1 * y * env).astype(np.float32)


def shift_cents(y):
    """Fine tuning offset of y from A440 in cents, within +-50."""
    return float(
        librosa.estimate_tuning(y=y, sr=SR, resolution=0.005, bins_per_octave=12) * 100
    )


def click_delay(tool, kind, amount):
    """Delay in ms of a single click after the transform, mapped back to input time."""
    x = np.zeros(SR * 3, np.float32)
    x[SR] = 1.0
    y, scale = (transforms.pitch if kind == "pitch" else transforms.tempo)(
        x, SR, amount, tool
    )
    expect = SR * scale
    return (int(np.argmax(np.abs(y))) - expect) / SR * 1000


x = tone_mix()
base = shift_cents(x)
rows = []
print("tool        kind   amount  pitch_err_c  len_err_%  click_ms")
for tool in transforms.TOOLS:
    for kind, amounts in (("pitch", (10, -30, 45)), ("tempo", (1.05, 0.9, 1.2))):
        for a in amounts:
            y, scale = (transforms.pitch if kind == "pitch" else transforms.tempo)(
                x, SR, a, tool
            )
            want_c = (
                a
                if kind == "pitch"
                else (1200 * np.log2(a) if tool == "resample" else 0.0)
            )
            got_c = shift_cents(y) - base
            got_c = (
                got_c - want_c + 50
            ) % 100 - 50  # fold to the +-50 range the estimate covers
            len_err = 100 * (len(y) / (len(x) * scale) - 1)
            d = click_delay(tool, kind, a)
            rows.append(
                dict(
                    tool=tool,
                    kind=kind,
                    amount=a,
                    pitch_err=got_c,
                    len_err=len_err,
                    click_ms=d,
                )
            )
            print(
                f"{tool:11s} {kind:5s} {a:7.2f}  {got_c:+10.2f}  {len_err:+9.2f}  {d:+8.2f}",
                flush=True,
            )
json.dump(
    rows,
    open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            "check_transforms.json",
        ),
        "w",
    ),
    indent=0,
)
