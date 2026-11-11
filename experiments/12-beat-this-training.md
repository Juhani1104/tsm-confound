# 12. Does the drift hurt a trained model?

**Question:** If a beat tracker is trained on tempo-augmented data whose labels drift by up to about ±100 ms, is the trained model worse? And if we fix the labels, does it get better?

## What it does

`src/bt_train.py` trains a small Beat This! model (2.1 million parameters) with the beat_this package's own model, loss and data code, on the released spectrograms. The settings follow the original training script. The runs differ only in how tempo augmentation is done:

| Condition | Tempo augmentation |
|---|---|
| `asis` | the released tempo-changed spectrograms, with labels divided by the tempo factor (what Beat This! did) |
| `fixed` | the same spectrograms, with labels moved by the predicted drift curve (`results/bt_lagcurves.npz`) |
| `notempo` | none |
| `spectempo` | the same tempo factors, made by stretching the original spectrogram in time instead of using the stretched audio, so there is no drift and no audio artifact at all |

There are two training sizes:

- **4 datasets** (Beatles, Candombe, GuitarSet, Hainsworth): 100 epochs, 5 seeds. `BT_SETS` limits training to these.
- **11 datasets** (adding JAAH, Filosax, RWC, SMC, HJDB, SIMAC and Ballroom): 25 epochs, which is about the same number of training steps, and 3 seeds. `BT_TAG=big` keeps these runs separate. `spectempo` was only run at this size.

Then:

1. `src/bt_eval.py` tests every checkpoint on GTZAN (993 songs, a common beat-tracking test set) and on the held-out part of the training datasets. It reports the beat and downbeat F-measure at 70, 50, 30 and 20 ms. It also reports the same scores after removing each model's own constant timing offset, because a fixed delay is easy to correct and should not count against a model.
2. `src/bt_compare.py` compares the conditions seed by seed and song by song.
3. `src/bt_threshold.py` checks whether a small downbeat difference is only a matter of where the decision threshold sits. It saves each model's raw downbeat scores, tries a range of thresholds, and compares every model at its own best one.

Each training run takes about an hour on an 8 GB GPU.

## Run it

```
run/12_beat_this_training.sh
```

## Output

- `results/bt_train.jsonl` and `results/bt_train_big.jsonl`: one line per training run
- `results/bt_eval*.jsonl`: evaluation of every checkpoint
- `results/summary_bt*.txt`: comparisons between conditions
- `results/bt_threshold.txt`: the threshold check
- `ckpt/`: the trained models (not in git)
