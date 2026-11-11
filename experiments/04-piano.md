# 04. Piano transcription under every tool

**Question:** How much does each tool hurt piano transcription? Is the damage about timing (notes found at the wrong time) or about content (wrong or missing notes)? And does a model trained with one tool become robust to the others?

## What it does

1. `src/task_piano.py` takes the first 60 seconds of 30 MAESTRO test pieces and applies every condition: pitch ±10 and ±200 cents, tempo ±5, 10 and 20%, each with all six tools. It transcribes every version with one of two models:
   - **Kong et al.**, trained on clean audio;
   - **Edwards et al.**, the same kind of model retrained with phase-vocoder pitch shifts as data augmentation.

   It saves the onset F1 for each version, and also every note the model found, so the notes can be scored again later.
2. `src/piano_rescore.py` separates timing from content. For each version, it fits a straight line that best maps the found note times onto the true note times. It then moves the notes along that line and scores them again. Whatever this recovers was a timing problem; whatever is still lost is a content problem. Clean audio, signalsmith and small-block Rubber Band serve as controls: alignment should not change their scores.
3. Two cause tests:
   - `src/piano_nfft.py` changes only the window size of the torchaudio vocoder (256 to 4096 samples), to see whether short windows create false notes and long windows smear timing.
   - `src/piano_formant.py` turns Rubber Band's formant preservation on and off, to see whether that setting explains Rubber Band's loss at ±200 cents.

Some tempo changes made by resampling move the pitch by a fraction of a semitone, but transcription models only output whole semitones. Those conditions cannot be scored fairly and are left out.

## Run it

```
run/04_piano.sh
```

## Output

- `results/piano_all.jsonl` (Kong), `results/piano_all_edwards.jsonl`, and the matching `*_notes.jsonl`
- `results/summary_piano_all*.txt`: scores and paired differences
- `results/summary_piano_all*_rescore.txt`: the timing/content split
- `results/piano_nfft.txt` and `results/piano_formant.txt`
