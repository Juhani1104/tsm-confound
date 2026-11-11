#!/bin/sh
# The Rubber Band R3 time drift: measure it, check library 4.0 directly, predict it from the
# engine's own log, and measure it at Beat This!'s settings. See experiments/09-rubberband-drift.md.
# RUBBERBAND_LIB: path to librubberband.so.3 version 4.0 (for example from Ubuntu's
# librubberband3 package); its folder must also hold libfftw3 and libsamplerate.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python} -I"
$PY src/rb_offset.py > results/rb_offset.txt
LD_LIBRARY_PATH="$(dirname "$RUBBERBAND_LIB")" $PY src/rb4_check.py "$RUBBERBAND_LIB" > results/rb4_check.txt
$PY src/bt_lagcurves.py
$PY src/rb_mechanism.py > results/rb_mechanism.txt
$PY src/rb_beatthis.py > results/rb_beatthis.txt
