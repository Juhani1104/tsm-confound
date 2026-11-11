#!/bin/sh
# Real pitch change vs each tool's pitch shift, on NSynth note pairs two semitones apart.
# See experiments/07-real-pitch.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
$PY src/task_truevar.py 400
$PY src/analyze_truevar.py > results/summary_truevar.txt
