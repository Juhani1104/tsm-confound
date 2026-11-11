#!/bin/sh
# Genre and beat tracking on FMA small, three tools, three shift sizes.
# See experiments/02-fma.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
$PY src/beats.py 100 > results/beats.txt
$PY src/timbre.py > results/timbre.txt
$PY src/timbre2.py > results/timbre2.txt
