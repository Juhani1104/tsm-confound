#!/bin/sh
# Re-run the published piano pitch-shift test (Edwards et al. 2024) with three tools.
# See experiments/03-replicate.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
$PY src/replicate.py > results/replicate.txt
