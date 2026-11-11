#!/bin/sh
# Why phase-vocoder output looks early, and whether phase locking fixes it.
# See experiments/10-vocoder-timing.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
$PY src/vocoder_offset.py > results/vocoder_offset.txt
$PY src/vocoder_phase.py > results/vocoder_phase.txt
