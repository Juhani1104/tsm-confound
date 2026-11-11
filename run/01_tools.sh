#!/bin/sh
# Check each tool, without any model: does it make the change it claims, and how much
# does it damage the sound? See experiments/01-tools.md.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
$PY src/check_transforms.py > results/check_transforms.txt
$PY src/artifacts2.py 30 > results/artifacts2.txt
