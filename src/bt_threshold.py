"""Is the downbeat gap between asis and fixed a calibration (threshold) effect?

usage: python src/bt_threshold.py [TAG]     (TAG: '' for the 4-dataset runs, 'big' for 11)

Every model over-predicts downbeats on GTZAN, and the fixed-label models more so (higher
recall, lower precision). Hypothesis: the misaligned labels make asis's downbeat head output
lower logits, which happens to offset the over-prediction; the representation is not better.
Test: predict GTZAN once per checkpoint keeping the frame logits, then run Beat This!'s
minimal postprocessing with the downbeat logits shifted by -tau for tau in -2..4, and score
downbeat F (70 ms). If fixed matches or beats asis at each model's best tau, the gap is
calibration. The beat head is left at its default threshold.
"""

import json
import os
import pathlib
import sys
from pathlib import Path

import mir_eval
import numpy as np
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_this.dataset.dataset import BeatDataModule  # noqa: E402
from beat_this.model.pl_module import PLBeatThis  # noqa: E402
from beat_this.model.postprocessor import Postprocessor  # noqa: E402

import bt_train_shim  # noqa: E402,F401
from paths import BT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = BT
TAG = "_" + sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else ""
LOG = ROOT / "results" / f"bt_logits{TAG}"
LOG.mkdir(exist_ok=True)
TAUS = np.arange(-2.0, 4.01, 0.5)
pp = Postprocessor(type="minimal", fps=50)
ann = DATA / "annotations" / "gtzan" / "annotations" / "beats"


def logits(ckpt):
    """{item/beat, item/downbeat: frame logits} on GTZAN for one checkpoint, cached."""
    p = LOG / (ckpt.stem + ".npz")
    if p.exists():
        return dict(np.load(p, allow_pickle=False))
    model = PLBeatThis.load_from_checkpoint(ckpt, map_location="cuda").cuda().eval()
    dm = BeatDataModule(
        DATA,
        batch_size=1,
        num_workers=4,
        augmentations={},
        test_dataset="gtzan",
        predict_datasplit="test",
    )
    dm.setup(stage="test")
    dm.setup(stage="predict")
    out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for batch in dm.predict_dataloader():
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            _, pred, _, sp = model.predict_step(batch, 0)
            stem = Path(sp[0]).parent.name
            out[f"{stem}/beat"] = pred["beat"][0].float().cpu().numpy()
            out[f"{stem}/downbeat"] = pred["downbeat"][0].float().cpu().numpy()
    np.savez(p, **out)
    return out


refs = {}
res = {}
for ckpt in sorted((ROOT / "ckpt" / f"bt{TAG}").glob("*_e*.ckpt")):
    cond = ckpt.stem.split("_")[0]
    if cond not in ("asis", "fixed", "notempo", "spectempo"):
        continue
    L = logits(ckpt)
    stems = sorted({k.rsplit("/", 1)[0] for k in L})
    f = np.zeros(len(TAUS))
    n = 0
    for stem in stems:
        if stem not in refs:
            a = np.loadtxt(ann / f"{stem}.beats", ndmin=2)
            refs[stem] = (
                mir_eval.beat.trim_beats(a[a[:, 1] == 1, 0])
                if a.shape[1] > 1
                else np.array([])
            )
        if not len(refs[stem]):
            continue
        b = torch.from_numpy(L[f"{stem}/beat"])
        d = torch.from_numpy(L[f"{stem}/downbeat"])
        for i, tau in enumerate(TAUS):
            _, down = pp(b, d - float(tau), None)
            f[i] += mir_eval.beat.f_measure(
                refs[stem], mir_eval.beat.trim_beats(np.asarray(down))
            )
        n += 1
    f /= n
    res.setdefault(cond, []).append(f)
    print(
        ckpt.stem,
        "F at tau=0",
        round(f[TAUS == 0][0], 4),
        "best",
        round(f.max(), 4),
        "at tau",
        TAUS[f.argmax()],
        flush=True,
    )

print(
    f"\ndownbeat F70 on GTZAN by threshold shift tau (mean over seeds){' [' + TAG[1:] + ']' if TAG else ''}"
)
print("tau       " + " ".join(f"{t:+6.1f}" for t in TAUS))
for c, v in res.items():
    print(f"{c:9s} " + " ".join(f"{x:6.3f}" for x in np.mean(v, 0)))
summary = {
    c: {
        "f_by_tau": np.mean(v, 0).tolist(),
        "best": float(np.mean([x.max() for x in v])),
        "at0": float(np.mean([x[TAUS == 0][0] for x in v])),
    }
    for c, v in res.items()
}
json.dump(
    {"taus": TAUS.tolist(), **summary},
    open(ROOT / "results" / f"bt_threshold{TAG}.json", "w"),
    indent=1,
)
for c, s in summary.items():
    print(
        f"{c:9s} at tau 0: {s['at0']:.4f}   each model at its own best tau: {s['best']:.4f}"
    )
