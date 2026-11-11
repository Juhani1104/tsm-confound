#!/bin/sh
# Piano transcription under every tool: 30 MAESTRO pieces, Kong and Edwards models, then
# the timing/content split and the two cause tests. See experiments/04-piano.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
export TSM_TAG=all
PIANO_MODEL=kong $PY src/task_piano.py 30 60
PIANO_MODEL=edwards $PY src/task_piano.py 30 60
for m in "" _edwards; do
  $PY src/analyze.py results/piano_all$m.jsonl value > results/summary_piano_all$m.txt
  $PY src/piano_rescore.py results/piano_all${m}_notes.jsonl > results/summary_piano_all${m}_rescore.txt
done
$PY src/piano_nfft.py > results/piano_nfft.txt
$PY src/piano_formant.py > results/piano_formant.txt
