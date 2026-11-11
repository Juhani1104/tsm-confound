# Setup

This page lists what you need before running any experiment.

## The idea in one paragraph

To test whether a music model can handle a change in pitch or tempo, people usually take a recording, change it with an audio tool, and check whether the model's answer changes. But every tool also adds its own side effects. These experiments make the same change with several tools, so we can tell how much of the effect comes from the change itself and how much comes from the tool.

## Python packages

We used Python 3.12 with a CUDA GPU (8 GB is enough). The main packages are:

- audio tools: `librosa`, `torchaudio`, `python-stretch` (signalsmith), `pedalboard` (Rubber Band), `pylibrb` (Rubber Band with control over block size)
- models: `piano_transcription_inference`, `basic-pitch`, `beat-this`, `panns_inference`, `transformers`, `tensorflow` with `tf_keras`, `onnxruntime`, `encodec`
- everything else: `mir_eval`, `pretty_midi`, `soundfile`, `scikit-learn`, `pandas`, `pytorch_lightning`, `matplotlib`

Run every script from the repository root as `python -I src/<script>.py`. The `-I` flag stops Python from loading files from the current folder by accident.

## Data and model weights

`src/paths.py` defines every path. By default, data goes in `data/` and model weights go in `ext/`. Neither folder is in git. To keep them on another disk, set `TSM_DATA` and `TSM_EXT`.

| What | Where it goes | Where to get it |
|---|---|---|
| MAESTRO v3 | `data/maestro/` | magenta.tensorflow.org/datasets/maestro |
| Studio MAESTRO | `data/studio_maestro/` | Zenodo record 10082144 |
| GuitarSet | `data/guitarset/annotation.zip`, `audio_mono-mic.zip` | Zenodo record 3371780 |
| NSynth valid and test | `data/nsynth/x/nsynth-valid/`, `nsynth-test/` | magenta.tensorflow.org/datasets/nsynth |
| FMA small and metadata | `data/fma_small.zip`, `data/fma_metadata/tracks.csv` | github.com/mdeff/fma |
| Deezer detector split | `data/deezer_split.npy` | github.com/deezer/deepfake-detector |
| Beat This! spectrograms | `data/bt/audio/spectrograms/*.npz` | Zenodo record 13922116 (`tools/zget.sh NAME` fetches one) |
| Beat This! annotations | `data/bt/annotations/` | the same Zenodo record |
| Kong piano model | `ext/piano/note_F1.pth` | Zenodo record 4034264 (`CRNN_note_F1=0.9677_pedal_F1=0.9186.pth`) |
| Edwards piano model | `ext/piano/edwards_aug.pth` | Zenodo record 10610212 |
| PANNs Cnn14 | `ext/panns/Cnn14_mAP=0.431.pth` | Zenodo record 3987831 |
| AST (AudioSet) | `ext/ast/` | Hugging Face `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Beat This! final0 | `ext/beat_this-final0.ckpt` | the beat_this repository |
| Deezer detector | `ext/deezer_specnn_amplitude/` | github.com/deezer/deepfake-detector |
| ArtifactNet | `ext/artifactnet/artifactnet_v94_full.onnx` | its model card on Hugging Face |

The small lists in `index/` say which MAESTRO and Studio MAESTRO pieces we used, and how far each Studio recording is offset from its MIDI file.

## Shared code

- `src/transforms.py` makes one pitch or tempo change with any of the six tools.
- `src/harness.py` lists the test conditions: three pitch sizes, three tempo sizes, and every tool. Each piece gets one random direction (up or down) for each size, and every tool uses the same direction, so the tools can be compared piece by piece.
- `src/analyze.py` summarises any task's results: the average score for each condition, and the difference from signalsmith with a 95% confidence interval.

## Running

Each experiment has a script in `run/` and a page in this folder. The scripts write into `results/`. Most experiments take between a few minutes and a few hours. Retraining Beat This! (experiment 12) takes about a day.
