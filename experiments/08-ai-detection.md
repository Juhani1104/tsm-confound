# 08. AI-music detection

**Question:** Detectors of AI-generated music are often tested by shifting pitch or tempo, to see whether a small edit can hide AI music. Does the answer depend on the tool used for the edit?

## What it does

1. `src/task_aidetect.py` takes 12 seconds from each of 80 real songs in the detector's own held-out test split. It makes an "AI" copy of each by passing it through the EnCodec neural audio codec, the kind of artifact these detectors are trained to catch. Both copies get every condition, and two detectors score them:
   - Deezer's released detector, which outputs the chance that a clip is real;
   - ArtifactNet. On EnCodec copies it is barely better than a coin flip even before any edit, so we leave it out of the conclusions.

   For each condition we report how often real songs are wrongly flagged, how often AI copies are caught, and the AUC, which measures how well the detector separates the two groups.
2. `src/aidetect_phase.py` tests one possible cause. It makes a ±10 cent shift with our own phase vocoder in two versions: one where every frequency bin keeps its own phase (like librosa), and one with phase locking. If phase locking brings detection back, the vocoder's phase errors are what hide the codec's traces.

This experiment runs on the CPU only.

## Run it

```
run/08_ai_detection.sh
```

## Output

- `results/aidetect.jsonl` and `results/summary_aidetect.txt`
- `results/aidetect_phase.txt`
