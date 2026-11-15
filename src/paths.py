"""Where every script finds its data, model weights and results.

Everything lives under the repository: data/ and ext/ hold downloaded datasets and
weights (neither is in git), index/ holds the small piece lists the experiments use,
results/ holds what the scripts write. TSM_DATA and TSM_EXT point data/ and ext/
somewhere else, for example a larger disk.

Expected layout (download sources in experiments/00-setup.md):
  data/maestro/            MAESTRO v3 audio and MIDI, as in the official zip
  data/studio_maestro/     Studio MAESTRO audio (Zenodo 10082144)
  data/guitarset/          GuitarSet annotation.zip and audio_mono-mic.zip
  data/nsynth/x/           nsynth-valid/ and nsynth-test/, unpacked
  data/fma_small.zip       FMA small, as downloaded
  data/fma_metadata/       tracks.csv from fma_metadata.zip, and track_meta.csv
  data/deezer_split.npy    the Deezer detector's train/test split of FMA
  data/bt/                 Beat This! annotations and spectrogram bundles
  ext/panns/, ext/ast/, ext/beat_this-final0.ckpt, ext/piano/, ext/deezer_specnn_amplitude/,
  ext/artifactnet/         model weights
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("TSM_DATA", ROOT / "data"))
EXT = Path(os.environ.get("TSM_EXT", ROOT / "ext"))
INDEX = ROOT / "index"
RESULTS = ROOT / "results"

# datasets
MAESTRO = DATA / "maestro"
STUDIO_MAESTRO = DATA / "studio_maestro"
GUITARSET = DATA / "guitarset"
NSYNTH = DATA / "nsynth" / "x"
FMA_ZIP = DATA / "fma_small.zip"
FMA_TRACKS = DATA / "fma_metadata" / "tracks.csv"
FMA_META = DATA / "fma_metadata" / "track_meta.csv"
DEEZER_SPLIT = DATA / "deezer_split.npy"
BT = DATA / "bt"

# piece lists: 30 MAESTRO v3 test pieces (every third is the original 10-piece set), the
# matching Studio MAESTRO recordings, and the Studio recordings' alignment lags in seconds
MAESTRO_PICK = INDEX / "maestro_pick.json"
STUDIO_PICK = INDEX / "studio_pick.json"
STUDIO_LAGS = INDEX / "studio_lags.json"

# model weights
PANNS = EXT / "panns" / "Cnn14_mAP=0.431.pth"
AST = EXT / "ast"
BEAT_THIS = EXT / "beat_this-final0.ckpt"
KONG = EXT / "piano" / "note_F1.pth"
EDWARDS = EXT / "piano" / "edwards_aug.pth"
DEEZER = EXT / "deezer_specnn_amplitude"
ARTIFACTNET = EXT / "artifactnet" / "artifactnet_v94_full.onnx"
# panns_inference keeps AudioSet's label list in <this folder>/panns_data (see panns_cnn14.py)
PANNS_HOME = Path(os.environ.get("TSM_PANNS_HOME", EXT))
