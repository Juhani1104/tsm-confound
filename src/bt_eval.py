"""Re-evaluate the trained Beat This! checkpoints on GTZAN, separating bias from jitter.

usage: python src/bt_eval.py [SPLIT]

SPLIT is test (GTZAN, default) or val: the validation part of the single split of every
training dataset (full pieces, never augmented). Results for val go to bt_eval_val.jsonl and
results/bt_pred_val/.

F-measure at tight tolerances mixes two things: a constant offset of a model's beats (which
a fixed latency correction would remove) and the spread of its beats around the truth.
For every checkpoint in ckpt/bt (from bt_train.py), GTZAN beats are predicted once, then
scored at 70/50/30/20 ms both as is and after subtracting that model's median signed error
(estimated on the same matched beats, so it is an oracle constant shift). Predictions are
kept in results/bt_pred/NAME.json; the table goes to results/bt_eval.jsonl.
"""

import json
import os
import pathlib
import sys
from pathlib import Path

import mir_eval
import numpy as np
import torch

# our own checkpoints store the data module's PosixPath in their hyperparameters; allow just
# that class so they still load with weights_only=True
torch.serialization.add_safe_globals([pathlib.PosixPath])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_this.dataset.dataset import BeatDataModule  # noqa: E402
from beat_this.model.pl_module import PLBeatThis  # noqa: E402
from beat_this.model.postprocessor import Postprocessor  # noqa: E402

import bt_train_shim  # noqa: E402,F401  numpy shims for beat_this's npz reader
from paths import BT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = BT
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
assert SPLIT in ("test", "val")
TAG = (
    "_" + os.environ["BT_TAG"] if os.environ.get("BT_TAG") else ""
)  # same as bt_train.py
SUF = TAG + ("" if SPLIT == "test" else "_val")
PRED = ROOT / "results" / f"bt_pred{SUF}"
PRED.mkdir(exist_ok=True)
OUT = ROOT / "results" / f"bt_eval{SUF}.jsonl"
TOL = (0.07, 0.05, 0.03, 0.02)
# datasets present when each run set was trained (val scores only these)
TRAIN_SETS = {"beatles", "candombe", "guitarset", "hainsworth"}
if TAG == "_big":
    TRAIN_SETS |= {"jaah", "filosax", "rwc", "smc", "hjdb", "simac", "ballroom"}
pp = Postprocessor(type="minimal", fps=50)


def predict(ckpt):
    """{item: {beat, downbeat}} times for one checkpoint, cached in PRED."""
    p = PRED / (ckpt.stem + ".json")
    if p.exists():
        return json.load(open(p))
    model = PLBeatThis.load_from_checkpoint(ckpt, map_location="cuda").cuda().eval()
    dm = BeatDataModule(
        DATA,
        batch_size=1,
        num_workers=4,
        augmentations={},
        test_dataset="gtzan",
        predict_datasplit=SPLIT,
    )
    if SPLIT == "test":
        dm.setup(stage="test")
    else:
        dm.setup(stage="fit")
    dm.setup(stage="predict")
    if SPLIT == "val":
        # only the datasets the checkpoints were trained on; bundles downloaded later would
        # otherwise change the evaluation set between checkpoints
        dm.predict_dataset.items = [
            i
            for i in dm.predict_dataset.items
            if str(i["spect_path"]).split("/")[0] in TRAIN_SETS
        ]
    out = {}
    # spectrograms are float16; run as the trainer's 16-mixed precision does
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for batch in dm.predict_dataloader():
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            _, pred, _, spect_path = model.predict_step(batch, 0)
            beat, down = pp(pred["beat"].cpu(), pred["downbeat"].cpu(), None)
            out[str(Path(spect_path[0]).parent)] = {
                "beat": np.asarray(beat).ravel().tolist(),
                "downbeat": np.asarray(down).ravel().tolist(),
            }
    json.dump(out, open(p, "w"))
    return out


def errors(ref, est):
    """Signed error (s) of every estimated beat within 70 ms of a reference beat."""
    if not len(ref) or not len(est):
        return np.array([])
    d = est[:, None] - ref[None, :]
    j = np.abs(d).argmin(1)
    e = d[np.arange(len(est)), j]
    return e[np.abs(e) < 0.07]


done = {json.loads(l)["name"] for l in open(OUT)} if OUT.exists() else set()
for ckpt in sorted((ROOT / "ckpt" / f"bt{TAG}").glob("*_e*.ckpt")):
    if ckpt.stem in done:
        continue
    pred = predict(ckpt)
    # predictions cached before the val option were keyed by the GTZAN stem alone
    pred = {(k if "/" in k else "gtzan/" + k): v for k, v in pred.items()}
    refs = {}
    for stem in pred:
        ds, name = stem.split("/", 1)
        a = np.loadtxt(
            DATA / "annotations" / ds / "annotations" / "beats" / f"{name}.beats",
            ndmin=2,
        )
        if a.shape[1] < 2:  # beat-only datasets: no downbeat column
            a = np.c_[a[:, 0], np.zeros(len(a))]
        refs[stem] = {
            "beat": mir_eval.beat.trim_beats(a[:, 0]),
            "downbeat": mir_eval.beat.trim_beats(a[a[:, 1] == 1, 0]),
        }
    row = {"name": ckpt.stem}
    for k in ("beat", "downbeat"):
        # items without annotated downbeats are left out of the downbeat scores
        items = [s for s in sorted(pred) if len(refs[s][k])]
        est = {s: mir_eval.beat.trim_beats(np.array(pred[s][k])) for s in items}
        bias = float(
            np.median(np.concatenate([errors(refs[s][k], est[s]) for s in items]))
        )
        row[f"{k}_bias_ms"] = bias * 1000
        for t in TOL:
            row[f"{k}_f{int(t * 1000)}"] = float(
                np.mean([mir_eval.beat.f_measure(refs[s][k], est[s], t) for s in items])
            )
            row[f"{k}_f{int(t * 1000)}_cal"] = float(
                np.mean(
                    [
                        mir_eval.beat.f_measure(
                            refs[s][k], np.maximum(est[s] - bias, 0), t
                        )
                        for s in items
                    ]
                )
            )
        row[f"{k}_per_item_f20_cal"] = [
            mir_eval.beat.f_measure(refs[s][k], np.maximum(est[s] - bias, 0), 0.02)
            for s in items
        ]
        row[f"{k}_items"] = len(items)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        ckpt.stem,
        {k: round(v, 3) for k, v in row.items() if isinstance(v, float)},
        flush=True,
    )
