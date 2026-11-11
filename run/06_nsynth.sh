#!/bin/sh
# Instrument family on NSynth: the zero-shot attempt (too close to chance to use) and the
# linear probe that replaced it. See experiments/06-nsynth.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
$PY src/task_nsynth.py 40
$PY src/analyze_nsynth.py > results/summary_nsynth.txt
$PY src/task_nsynth_probe.py 150 40
for m in panns ast; do $PY src/analyze.py results/nsynth_probe.jsonl model:$m family; done > results/summary_nsynth_probe.txt
