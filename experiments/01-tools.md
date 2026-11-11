# 01. Checking the tools themselves

**Question:** Before any model is involved, does each tool make the change it says it makes? And how much does it damage the sound along the way?

## What it does

1. `src/check_transforms.py` sends simple test sounds through every tool:
   - a steady tone, to check that the pitch really moves by the right amount;
   - a click train, to check that a tempo change really changes the timing by the right amount;
   - a single click, to see whether the tool delays or smears it.
2. `src/artifacts2.py` measures damage on 30 real FMA songs. Each song is shifted up and then back down by the same amount. In a perfect tool, this round trip would give back the original, so whatever differs is damage the tool added. We measure two kinds:
   - **spectral damage:** the average difference between the original and the round-trip version, in decibels, on a mel spectrogram (the kind of input most music models see);
   - **onset smearing:** how much sharper or softer the note starts are afterwards.

   It also repeats the vocoder test with different window lengths. This shows that onset smearing depends on how long the analysis window is.

## Run it

```
run/01_tools.sh
```

## Output

- `results/check_transforms.txt`
- `results/artifacts2.txt` and `results/artifacts2.json`
