#!/bin/sh
# Check Beat This!'s released training spectrograms for the drift, and test the simulated
# correction curves against them. See experiments/11-beat-this-data.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
for d in guitarset candombe beatles; do
  $PY src/bt_spect_check.py data/bt/audio/spectrograms/$d.npz > results/bt_spect_check_$d.txt
done
$PY src/bt_lagcurves.py data/bt/audio/spectrograms/*.npz > results/bt_lagcurves.txt
