# 07. A real pitch change versus a simulated one

**Question:** When a tool raises a note by two semitones, does the result look like the same instrument really playing two semitones higher?

## What it does

NSynth often has the same instrument, at the same loudness, recorded at neighbouring pitches. That gives us a real answer to compare against: the recording two semitones up.

`src/task_truevar.py` takes 400 such pairs and uses each pair in both directions (+200 and −200 cents). For each one it compares:

- the **real** note at the target pitch;
- the **original** note, not shifted at all;
- the original note shifted by each of the six tools.

For every version it records three things:

1. which family a clean-trained probe predicts (PANNs and AST, trained as in experiment 06);
2. how far its embedding is from the real note's embedding (cosine distance);
3. how far its mel spectrogram is from the real note's (in decibels).

If the model is just as accurate on the real higher note, then a real pitch change does not bother it. Any drop under a tool then comes from the tool.

## Run it

```
run/07_real_pitch.sh
```

## Output

- `results/truevar.jsonl`: one line per note and direction
- `results/summary_truevar.txt`: accuracy and distances for each version, overall and by instrument source (acoustic, electronic, synthetic)
