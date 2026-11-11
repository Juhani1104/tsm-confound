"""Shared conditions for every task: one clean version and every (kind, amount, tool) change.

A condition is (kind, magnitude, tool). Each item gets one random sign per kind and
magnitude, shared by all tools, so tools are compared on exactly the same change.

PITCH_INVARIANT   pitch magnitudes for tasks whose labels do not depend on pitch
PITCH_LABELLED    for note tasks: 10 cents stays inside the 50-cent scoring tolerance
                  of the original labels; 200 cents moves every label by 2 semitones.
                  50 cents is left out, it sits exactly between two notes.
TEMPO             tempo factors; the sign picks faster (f) or slower (1/f)
"""

import os

import numpy as np

import transforms

PITCH_INVARIANT = [10, 50, 200]
PITCH_LABELLED = [10, 200]
TEMPO = [1.05, 1.10, 1.20]
# TSM_TOOLS=a,b runs a task on other tools; TSM_TAG names its results file (task_TAG.jsonl)
TOOLS = (
    tuple(os.environ["TSM_TOOLS"].split(","))
    if os.environ.get("TSM_TOOLS")
    else transforms.TOOLS
)
TAG = "_" + os.environ["TSM_TAG"] if os.environ.get("TSM_TAG") else ""


def conditions(pitch_amounts=PITCH_INVARIANT, tempo_amounts=TEMPO, tools=None):
    """All condition keys, clean first."""
    tools = tools or TOOLS
    out = [("clean", 0, "none")]
    out += [("pitch", a, t) for a in pitch_amounts for t in tools]
    out += [("tempo", a, t) for a in tempo_amounts for t in tools]
    return out


def signs(rng, pitch_amounts=PITCH_INVARIANT, tempo_amounts=TEMPO):
    """One random sign per (kind, magnitude), drawn once per item."""
    return {
        (k, a): int(rng.choice([-1, 1]))
        for k, am in (("pitch", pitch_amounts), ("tempo", tempo_amounts))
        for a in am
    }


def apply(x, sr, cond, sign):
    """(audio, time_scale, cents) for one condition; cents is the pitch change made."""
    kind, a, tool = cond
    if kind == "clean":
        return x, 1.0, 0.0
    if kind == "pitch":
        c = sign * a
        y, scale = transforms.pitch(x, sr, c, tool)
        return y, scale, float(c)
    f = a if sign > 0 else 1 / a
    y, scale = transforms.tempo(x, sr, f, tool)
    cents = 1200 * np.log2(f) if tool == "resample" else 0.0
    return y, scale, float(cents)


def key(cond):
    """A short string name for a condition, e.g. 'pitch200_librosa'."""
    kind, a, tool = cond
    if kind == "clean":
        return "clean"
    return f"{kind}{int(round(a if kind == 'pitch' else (a - 1) * 100))}_{tool}"
