# 09. The Rubber Band time drift

**Question:** With Rubber Band (through Spotify's pedalboard), piano transcription collapsed at a ±10 cent shift but held up at ±200 cents. Why?

## What we found to check

Rubber Band's high-quality engine (R3) drifts in time when it is given a large block of audio at once. The output has the right length, but its content slowly moves away from where it should be, by about 2 ms every second. pedalboard always gives it the largest block allowed. These scripts measure the drift, check where it comes from, and predict it exactly.

## What it does

1. `src/rb_offset.py` measures where events end up after each tool:
   - on a click train (short tone bursts at known times), the delay of every burst;
   - on 6 real piano pieces, the delay in 3-second windows.

   The small-block Rubber Band (`rbsmall`, 4096 samples at a time) is included as the fix.
2. `src/rb4_check.py` calls Rubber Band 4.0, the newest release, directly through its C interface, with several block sizes. This shows the drift is in the library itself, not in pedalboard.
3. `src/rb_mechanism.py` predicts the drift without listening to any audio. The engine picks a whole-number output step once per call and then uses it for every frame in the block. The script reads those choices from the engine's debug log, adds up the rounding error frame by frame, and compares the predicted curve with the measured one.
4. `src/rb_beatthis.py` measures the drift at the exact settings Beat This! used for its tempo augmentation: 44.1 kHz, tempo changes from −20% to +20% in 4% steps, on 240-second click trains and on real piano.
5. `src/bt_lagcurves.py` saves the drift curve for each of those tempo changes in `results/bt_lagcurves.npz`. Experiments 11 and 12 use these curves to correct the labels.

Step 2 needs Rubber Band 4.0 as a shared library. On Ubuntu, `apt-get download librubberband3 libfftw3-double3 libsamplerate0` and `dpkg-deb -x` give you one without installing anything. Set `RUBBERBAND_LIB` to the `librubberband.so.3` file.

## Run it

```
RUBBERBAND_LIB=/path/to/librubberband.so.3 run/09_rubberband.sh
```

## Output

- `results/rb_offset.txt` and `results/rb_offset.json`
- `results/rb4_check.txt`
- `results/rb_mechanism.txt`
- `results/rb_beatthis.txt`
- `results/bt_lagcurves.npz`
