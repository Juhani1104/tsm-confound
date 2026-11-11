#!/bin/sh
# AI-music detection under every tool, then the phase-locking test. CPU only.
# See experiments/08-ai-detection.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
export CUDA_VISIBLE_DEVICES=
$PY src/task_aidetect.py 80
$PY src/analyze_aidetect.py > results/summary_aidetect.txt
$PY src/aidetect_phase.py 40 > results/aidetect_phase.txt
