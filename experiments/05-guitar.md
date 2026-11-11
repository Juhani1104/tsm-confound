# 05. Guitar notes and beats

**Question:** Do the tool effects seen on piano also appear on another instrument and another kind of task?

## What it does

`src/task_guitar.py` uses 60 excerpts from GuitarSet, which has hand-checked note and beat labels. For every condition (pitch ±10 and ±200 cents, tempo ±5, 10 and 20%, all six tools), it runs two models:

- **Basic Pitch** (Spotify) transcribes the notes. We score them with onset F1: the onset must be within 50 ms and the pitch within 50 cents.
- **Beat This!** finds the beats. We score them with the standard beat F-measure, which allows 70 ms of error.

The labels move with the change: pitch labels move by the shift rounded to whole semitones, and times are scaled by the tempo change.

Beat This! was trained on GuitarSet, so its beat scores here are higher than they would be on new music. That does not matter for this test, because we only compare tools on the same audio.

## Run it

```
run/05_guitar.sh
```

## Output

- `results/guitar.jsonl`: one line per excerpt
- `results/summary_guitar.txt`: scores and paired differences for notes and beats
