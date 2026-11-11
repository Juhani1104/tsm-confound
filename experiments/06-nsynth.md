# 06. Instrument family on single notes (NSynth)

**Question:** On short, single notes from one instrument, do the tools still change what a model hears?

## What it does

NSynth has four-second notes from many instruments, grouped into families such as guitar, brass and keyboard.

1. **First attempt, kept as a lesson:** `src/task_nsynth.py` asked PANNs and AST to name the family directly, by mapping their AudioSet labels onto NSynth families. They were right only about 25% of the time, even on clean notes. When the starting score is that close to chance, it has no room to drop, so this test cannot show anything. `src/analyze_nsynth.py` summarises it.
2. **What we used instead:** `src/task_nsynth_probe.py` trains a simple linear classifier (a "probe") on the models' embeddings of clean notes from the NSynth validation set: 150 notes per family. It then tests it on 40 notes per family from the test set, under every condition. Clean accuracy is high, so a drop can show.

## Run it

```
run/06_nsynth.sh
```

## Output

- `results/nsynth.jsonl` and `results/summary_nsynth.txt`: the zero-shot attempt
- `results/nsynth_probe.jsonl` and `results/summary_nsynth_probe.txt`: the probe
