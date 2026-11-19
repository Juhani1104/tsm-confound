# 03. Re-running a published robustness test

**Question:** Edwards et al. (2024) reported that a well-known piano transcription model (Kong et al.) loses a lot of accuracy when the audio is shifted by only up to ±10 cents. Can we reproduce their numbers, and does the result depend on the tool?

## What it does

`src/replicate.py` follows their test as closely as we can:

1. Take 30 pieces from the MAESTRO test set and the same 30 pieces from Studio MAESTRO (a re-recording of the same performances in a studio).
2. Give each piece one random pitch shift between −10 and +10 cents.
3. Make that shift with three tools: the librosa phase vocoder (what their library used at the time), signalsmith-stretch, and plain resampling.
4. Transcribe the first 90 seconds with Kong's model and compute the note-onset F1 score against the MIDI file.

Studio MAESTRO recordings do not start exactly where their MIDI files start. Before scoring, each recording is shifted by its own small offset, listed in `index/studio_lags.json`.

If only the phase vocoder reproduces the published drops, the published drop is mostly the vocoder's doing.

## Run it

```
run/03_replicate.sh
```

## Output

- `results/replicate.txt`

The copy in `results/` was made by this script's first version in the
[piano-tuning](https://github.com/Juhani1104/piano-tuning) repository. That version read
the same piece lists and Studio MAESTRO offsets as the files in `index/`, and differs
only in file paths and formatting.
