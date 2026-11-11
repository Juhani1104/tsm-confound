"""Does the pedalboard drift in Beat This!'s tempo augmentation hurt the trained model?

usage: python src/bt_train.py CONDITION SEED [EPOCHS]

Trains a small Beat This! (transformer_dim 128, 6 layers; the beat_this package's own model,
loss, data module and Lightning module) on the released spectrograms in data/bt (every
dataset bundle present there except gtzan, or only those in BT_SETS; single split; BT_TAG
names a separate run set), with Beat This!'s augmentations
(pitch -5..+6, tempo -20..+20% step 4, masking). Conditions differ only in the tempo
augmentation:

  asis     released tempo-augmented spectrograms, beat labels divided by the factor
           (what Beat This! does; the labels are off by the pedalboard drift)
  fixed    same spectrograms, labels moved by the simulated drift curve of that factor
           (results/bt_lagcurves.npz; residual against the released data is about +-3 ms)
  notempo  no tempo augmentation
  spectempo  the same tempo factors, made by linearly resampling the ORIGINAL spectrogram
           along time instead of loading the time-stretched audio's spectrogram; labels
           divided by the factor (exact). Same tempo distribution as asis/fixed, no audio
           time-stretching artifacts and no drift (like the hop-size method of Böck & Davies)

Then predicts the GTZAN test set (full pieces) and scores beats and downbeats with
mir_eval F-measure at 70 ms (the standard), and at 50, 30 and 20 ms to show timing precision,
plus the median signed error of matched beats. One row per run to results/bt_train.jsonl,
checkpoint to ckpt/bt/.
"""

import json
import os
import sys
from pathlib import Path

import mir_eval
import numpy as np
import pytorch_lightning as pl

# beat_this's memory-mapped npz reader calls private numpy helpers that numpy 2.5 dropped;
# these use the public header readers instead
if not hasattr(np.lib.format, "_check_version"):

    def _check_version(v):
        if v not in ((1, 0), (2, 0), (3, 0)):
            raise ValueError(f"unsupported npy version {v}")

    np.lib.format._check_version = _check_version
if not hasattr(np.lib.format, "_read_array_header"):
    np.lib.format._read_array_header = lambda fp, v: (
        np.lib.format.read_array_header_1_0
        if v == (1, 0)
        else np.lib.format.read_array_header_2_0
    )(fp)

import beat_this.dataset.augment as aug  # noqa: E402
from beat_this.dataset.dataset import BeatDataModule, BeatTrackingDataset  # noqa: E402
from beat_this.model.pl_module import PLBeatThis  # noqa: E402
from beat_this.model.postprocessor import Postprocessor  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BT, RESULTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = BT
COND, SEED = sys.argv[1], int(sys.argv[2])
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
assert COND in ("asis", "fixed", "notempo", "spectempo")
# BT_TAG=big keeps a second experiment (e.g. more datasets) apart: ckpt/bt_big, bt_train_big.jsonl
TAG = "_" + os.environ["BT_TAG"] if os.environ.get("BT_TAG") else ""
OUT = ROOT / "results" / f"bt_train{TAG}.jsonl"
CKPT = ROOT / "ckpt" / f"bt{TAG}"

curves = dict(np.load(RESULTS / "bt_lagcurves.npz"))


def stretch_fixed(item, percentage):
    """Beat This!'s stretch_annotations, plus the simulated drift of that percentage."""
    if not percentage:
        return item
    t = item["beat_time"] / (1.0 + percentage / 100)
    t = (
        t
        + np.interp(t, curves[f"ts{percentage}_t"], curves[f"ts{percentage}_lag"])
        / 1000
    )
    return {**item, "beat_time": t}


if COND == "fixed":
    aug.stretch_annotations = stretch_fixed  # augment_tempo looks it up in its module


def stretch_filename_spec(item, percentage):
    """Keep the original spectrogram; remember the factor for _get_spect below."""
    return {**item, "spec_factor": 1.0 + percentage / 100} if percentage else item


def get_spect_scaled(self, item, _orig=BeatTrackingDataset._get_spect):
    """The item's spectrogram, resampled in time when it carries a spec_factor."""
    s = _orig(self, item)
    f = item.get("spec_factor")
    if not f:
        return s
    # frame j of the faster version shows original time j * f; interpolate between frames
    pos = np.arange(int(len(s) / f)) * f
    i0 = np.minimum(pos.astype(int), len(s) - 1)
    i1 = np.minimum(i0 + 1, len(s) - 1)
    w = (pos - i0)[:, None].astype(np.float32)
    return ((1 - w) * s[i0].astype(np.float32) + w * s[i1].astype(np.float32)).astype(
        s.dtype
    )


if COND == "spectempo":
    aug.stretch_filename = stretch_filename_spec
    BeatTrackingDataset._get_spect = get_spect_scaled

augmentations = {
    "pitch": {"min": -5, "max": 6},
    "mask": {
        "kind": "permute",
        "min_count": 1,
        "max_count": 6,
        "min_len": 0.1,
        "max_len": 2,
        "min_parts": 5,
        "max_parts": 9,
    },
}
if COND != "notempo":  # spectempo keeps the tempo entry; its files are never opened
    augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}

pl.seed_everything(SEED, workers=True)
dm = BeatDataModule(DATA, batch_size=8, train_length=1500, spect_fps=50, num_workers=6,
                    test_dataset="gtzan", length_based_oversampling_factor=0.65,
                    augmentations=augmentations, no_val=False)  # fmt: skip
dm.setup(stage="fit")
# BT_SETS=beatles,candombe,... trains and validates on those bundles only (the first,
# 4-dataset run set was trained before the other bundles were downloaded)
if os.environ.get("BT_SETS"):
    keep = set(os.environ["BT_SETS"].split(","))
    for ds in (dm.train_dataset, dm.val_dataset):
        ds.items = [i for i in ds.items if str(i["spect_path"]).split("/")[0] in keep]
pos = dm.get_train_positive_weights(widen_target_mask=3)
print(
    "datasets",
    sorted({i["dataset"] for i in dm.train_dataset.items}),
    "train items",
    len(dm.train_dataset),
    "pos",
    pos,
    flush=True,
)
model = PLBeatThis(spect_dim=128, fps=50, transformer_dim=128, ff_mult=4, n_layers=6, stem_dim=32,
                   dropout={"frontend": 0.1, "transformer": 0.2}, lr=0.0008, weight_decay=0.01,
                   pos_weights=pos, head_dim=32, loss_type="shift_tolerant_weighted_bce",
                   warmup_steps=1000, max_epochs=EPOCHS, use_dbn=False, eval_trim_beats=5,
                   sum_head=True, partial_transformers=True)  # fmt: skip
name = f"{COND}_s{SEED}_e{EPOCHS}"
trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="gpu", devices=1, precision="16-mixed",
                     accumulate_grad_batches=8, check_val_every_n_epoch=5, logger=False,
                     enable_checkpointing=False, num_sanity_val_steps=1, enable_progress_bar=False)  # fmt: skip
trainer.fit(model, dm)
CKPT.mkdir(parents=True, exist_ok=True)
trainer.save_checkpoint(CKPT / f"{name}.ckpt")

# ---- GTZAN evaluation at several tolerances
dm.setup(stage="test")
dm.predict_datasplit = "test"
dm.setup(stage="predict")
pp = Postprocessor(type="minimal", fps=50)
ann = DATA / "annotations" / "gtzan" / "annotations" / "beats"
TOL = (0.07, 0.05, 0.03, 0.02)
scores = {f"{k}_f{int(t * 1000)}": [] for k in ("beat", "downbeat") for t in TOL}
err = []
model.eval()
for out in trainer.predict(model, dm.predict_dataloader()):
    _, pred, _, spect_path = out
    stem = Path(spect_path[0]).parent.name
    a = np.loadtxt(ann / f"{stem}.beats", ndmin=2)
    ref = {"beat": a[:, 0], "downbeat": a[a[:, 1] == 1, 0]}
    est = dict(zip(("beat", "downbeat"), pp(pred["beat"], pred["downbeat"], None)))
    for k in ("beat", "downbeat"):
        r, e = (
            mir_eval.beat.trim_beats(ref[k]),
            mir_eval.beat.trim_beats(np.asarray(est[k]).ravel()),
        )
        for t in TOL:
            scores[f"{k}_f{int(t * 1000)}"].append(
                mir_eval.beat.f_measure(r, e, f_measure_threshold=t)
            )
        if k == "beat" and len(e) and len(r):
            d = e[:, None] - r[None, :]
            j = np.abs(d).argmin(1)
            m = np.abs(d[np.arange(len(e)), j]) < 0.07
            err += list(d[np.arange(len(e)), j][m])
row = {"cond": COND, "seed": SEED, "epochs": EPOCHS, "n_gtzan": len(scores["beat_f70"]),
       **{k: float(np.mean(v)) for k, v in scores.items()}, "beat_err_median_ms": float(np.median(err) * 1000),
       "beat_err_mad_ms": float(np.median(np.abs(np.array(err) - np.median(err))) * 1000)}  # fmt: skip
with open(OUT, "a") as f:
    f.write(json.dumps(row) + "\n")
np.save(
    ROOT / "results" / f"bt_gtzan{TAG}_{name}.npy",
    np.array([scores[k] for k in sorted(scores)]),
)
print(row, flush=True)
