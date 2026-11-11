# 11. Checking Beat This!'s released training data

**Question:** Beat This! made its tempo augmentation with pedalboard's default time stretch. Are the beats in its released training data out of line with their labels?

## What it does

Beat This! released its training spectrograms. For every song there is the original and 21 changed versions: 10 tempo changes and 11 pitch changes. During training, the beat times of a tempo-changed version are simply divided by the tempo factor. If the audio drifted, those labels are wrong.

1. `src/bt_spect_check.py` compares each tempo-changed spectrogram with the original. It turns both into an onset envelope (how much new sound starts in each frame), stretches the original's envelope to the new tempo, and finds the delay between the two in 3-second windows. Pitch-changed versions serve as a control: they were made with pedalboard's PitchShift plugin, which does not drift.
2. `src/bt_lagcurves.py`, given the spectrogram files, subtracts the drift curve predicted in experiment 09 from these measured delays. If the leftover is close to zero, the curves describe the released data well enough to correct the labels.

You can fetch one dataset's spectrogram file with `tools/zget.sh NAME` (for example, `tools/zget.sh candombe`).

## Run it

```
run/11_beat_this_data.sh
```

## Output

- `results/bt_spect_check_*.txt` and `.json`: delay over time for each tempo and pitch version
- `results/bt_lagcurves.txt`: what is left after subtracting the predicted curves, for each dataset
