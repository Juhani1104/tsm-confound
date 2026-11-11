"""Guitar note transcription (Basic Pitch on GuitarSet) and beat tracking (Beat This!) under every condition.

usage: python src/task_guitar.py N

N GuitarSet excerpts (mono-mic audio, JAMS annotations). Notes are scored with
mir_eval onset-only F1 at 50 ms and 50 cents, labels moved by the pitch change rounded
to whole semitones (so 200 cents moves them by 2) and times by the tempo change.
Beats use mir_eval F-measure against the annotated beats, times rescaled the same way.
Beat This! was trained on GuitarSet, so its absolute beat scores here are optimistic;
the comparison between tools on the same audio is what matters.
Rows are appended to results/guitar.jsonl, one per excerpt.
"""

import io
import json
import os
import sys
import tempfile
import zipfile

import librosa
import mir_eval
import numpy as np
import soundfile as sf
import torch

# basic_pitch imports TensorFlow when it is installed, and TF's start-up hides the GPU from
# PyTorch unless CUDA is initialised first
torch.cuda.init()
_ = torch.zeros(1).cuda()
from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402
from basic_pitch.inference import Model, predict  # noqa: E402
from beat_this.inference import Audio2Beats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from paths import BEAT_THIS, GUITARSET  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = GUITARSET
SR = 22050
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = f"{ROOT}/results/guitar{harness.TAG}.jsonl"
TMP = tempfile.mkdtemp()

bp = Model(ICASSP_2022_MODEL_PATH)
beats = Audio2Beats(checkpoint_path=str(BEAT_THIS), device="cuda")
ann = zipfile.ZipFile(f"{D}/annotation.zip")
aud = zipfile.ZipFile(f"{D}/audio_mono-mic.zip")
jams = {os.path.basename(n)[:-5]: n for n in ann.namelist() if n.endswith(".jams")}
wavs = {
    os.path.basename(n)[: -len("_mic.wav")]: n
    for n in aud.namelist()
    if n.endswith("_mic.wav")
}
names = sorted(set(jams) & set(wavs))
rng = np.random.default_rng(0)
names = list(rng.permutation(names)[:N])
COND = harness.conditions(pitch_amounts=harness.PITCH_LABELLED)


def labels(name):
    """(note onsets, note MIDI pitches, beat times) from a GuitarSet JAMS file."""
    j = json.loads(ann.read(jams[name]))
    on, pi, bt = [], [], []
    for a in j["annotations"]:
        ns = a["namespace"]
        if ns == "note_midi":
            for d in a["data"]:
                on.append(d["time"])
                pi.append(d["value"])
        elif ns == "beat_position" or ns == "beat":
            bt += [d["time"] for d in a["data"]]
    o = np.argsort(on)
    return np.array(on)[o], np.array(pi)[o], np.unique(np.round(bt, 4))


def transcribe(y):
    """Onsets and MIDI pitches from Basic Pitch."""
    f = f"{TMP}/x.wav"
    sf.write(f, y, SR)
    _, _, ev = predict(f, bp)
    return np.array([e[0] for e in ev]), np.array([e[2] for e in ev])


def note_f1(est, ref_on, ref_pi):
    """Onset F1 (50 ms, 50 cents) of (onsets, pitches) against the labels."""
    on, pi = est
    if len(on) == 0:
        return 0.0
    return mir_eval.transcription.precision_recall_f1_overlap(
        np.c_[ref_on, ref_on + 0.1], librosa.midi_to_hz(ref_pi),
        np.c_[on, on + 0.1], librosa.midi_to_hz(pi), offset_ratio=None, onset_tolerance=0.05,
    )[2]  # fmt: skip


done = {json.loads(l)["name"] for l in open(OUT)} if os.path.exists(OUT) else set()
with open(OUT, "a") as fout:
    for name in names:
        if name in done:
            continue
        x, sr = sf.read(
            io.BytesIO(aud.read(wavs[name])), dtype="float32", always_2d=True
        )
        x = librosa.resample(x.mean(1), orig_sr=sr, target_sr=SR, res_type="soxr_hq")
        r_on, r_pi, r_bt = labels(name)
        sg = harness.signs(rng, pitch_amounts=harness.PITCH_LABELLED)
        row = {"name": name}
        for cond in COND:
            y, scale, cents = harness.apply(x, SR, cond, sg.get((cond[0], cond[1]), 1))
            semis = round(cents / 100)
            est_bt, _ = beats(y, SR)
            row[harness.key(cond)] = dict(
                note_f1=note_f1(transcribe(y), r_on * scale, r_pi + semis),
                beat_f=float(
                    mir_eval.beat.f_measure(
                        mir_eval.beat.trim_beats(r_bt * scale),
                        mir_eval.beat.trim_beats(est_bt),
                    )
                )
                if len(r_bt) > 2
                else None,
                cents=cents,
            )
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(
            name,
            {
                k: round(v["note_f1"], 3)
                for k, v in row.items()
                if k != "name" and k.endswith(("clean", "librosa", "signalsmith"))
            },
            flush=True,
        )
print("done", flush=True)
