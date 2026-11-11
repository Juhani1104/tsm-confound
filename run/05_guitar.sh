#!/bin/sh
# Guitar notes (Basic Pitch) and beats (Beat This!) on GuitarSet under every tool.
# See experiments/05-guitar.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
$PY src/task_guitar.py 60
for m in note_f1 beat_f; do $PY src/analyze.py results/guitar.jsonl $m; done > results/summary_guitar.txt
