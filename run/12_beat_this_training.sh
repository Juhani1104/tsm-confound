#!/bin/sh
# Retrain a small Beat This! with drifted, corrected or no tempo augmentation (and, on the
# larger set, spectrogram-domain tempo), then evaluate. Each run is about an hour on one
# 8 GB GPU. See experiments/12-beat-this-training.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
# 4 training sets, 100 epochs, 5 seeds
for s in 0 1 2 3 4; do for c in asis fixed notempo; do
  BT_SETS=beatles,candombe,guitarset,hainsworth $PY src/bt_train.py $c $s 100
done; done
for split in test val; do $PY src/bt_eval.py $split; done
$PY src/bt_compare.py results/bt_eval.jsonl > results/summary_bt.txt
$PY src/bt_compare.py results/bt_eval_val.jsonl > results/summary_bt_val.txt
$PY src/bt_threshold.py > results/bt_threshold.txt
# 11 training sets, 25 epochs (about the same number of steps), 3 seeds
export BT_TAG=big
for s in 0 1 2; do for c in asis fixed notempo spectempo; do
  $PY src/bt_train.py $c $s 25
done; done
for split in test val; do $PY src/bt_eval.py $split; done
$PY src/bt_compare.py results/bt_eval_big.jsonl > results/summary_bt_big.txt
$PY src/bt_compare.py results/bt_eval_big_val.jsonl > results/summary_bt_big_val.txt
