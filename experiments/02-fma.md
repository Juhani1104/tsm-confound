# 02. Genre and beats on real songs (FMA)

**Question:** If we shift a song's pitch or tempo, does a genre classifier or a beat tracker change its answer because of the shift, or because of the tool that made it?

## What it does

We use songs from FMA small and change each one with three tools:

- a phase vocoder (librosa), the old default in the audiomentations library;
- signalsmith-stretch, the new default in audiomentations;
- plain resampling, which changes pitch and tempo together the way a faster record player would. It adds no time-stretching artifacts.

Then we compare each model's output before and after.

1. `src/beats.py`: Beat This! finds the beats in 100 songs (rock, electronic, hip-hop, pop). We score the beats it finds in the shifted song against the beats it found in the original.
2. `src/timbre.py`: we train a small linear classifier (a "probe") on PANNs embeddings to predict the genre. It is trained on 800 songs and tested on 400 shifted songs.
3. `src/timbre2.py`: the same test with a second model (AST, a transformer) and three shift sizes (10, 50 and 200 cents of pitch; 5, 10 and 20% of tempo).

The size test is the key one. If the score drops by the same amount whether the shift is tiny or large, the drop is caused by the tool, not by the change.

## Run it

```
run/02_fma.sh
```

## Output

- `results/beats.txt` and `results/beats.json`
- `results/timbre.txt` and `results/timbre.json`
- `results/timbre2.txt` and `results/timbre2.json`
