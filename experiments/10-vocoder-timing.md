# 10. Why phase-vocoder output sounds early

**Question:** After a librosa phase vocoder, note starts are detected about 20 ms too early. Is the sound really moved, or is it only smeared?

## What it does

1. `src/vocoder_offset.py` runs librosa's pitch shift and time stretch at window sizes from 512 to 4096 samples. It uses two kinds of test sound:
   - short tone bursts with a sharp start, where we measure when the sound reaches half its peak;
   - smooth, symmetric tones, where we measure the centre of their energy. Symmetric smearing would not move this centre, so if it moves, the sound has really been shifted in time.
2. `src/vocoder_phase.py` tests a cause. A phase vocoder keeps a separate running phase for every frequency bin, so the bins of one note can slowly fall out of step with each other. When that happens, the note's energy moves inside the analysis window. We build our own vocoder with the same spectrogram and magnitudes, in two versions:
   - one where every bin keeps its own phase, as in librosa;
   - one with phase locking (Laroche & Dolson, 1999), where each bin follows the strongest nearby peak.

   If phase locking removes the random shifts, lost phase alignment between bins is the cause.

## Run it

```
run/10_vocoder.sh
```

## Output

- `results/vocoder_offset.txt`
- `results/vocoder_phase.txt`
