#!/bin/sh
# Draw docs/fig1-5 from the result files.
set -e
cd "$(dirname "$0")/.."
${PY:-python} -I src/make_figures.py
