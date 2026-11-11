"""Load Kong et al.'s released model and Edwards et al.'s augmented retraining strictly.

PianoTranscription's own constructor is bypassed on purpose: it re-downloads Kong's
weights over any checkpoint smaller than 160 MB (Edwards's is 104 MB), and it loads
with strict=False, so mismatched keys would leave random weights without an error.
Here every key must match, and checkpoints are read with weights_only=True, so a
downloaded file can only hold tensors, never code.
"""

import numpy as np
import torch
from piano_transcription_inference import PianoTranscription, config
from piano_transcription_inference.models import (
    Note_pedal,
    Regress_onset_offset_frame_velocity_CRNN,
)

from paths import EDWARDS, KONG

CKPT = {
    "kong": (Note_pedal, KONG),
    "edwards": (Regress_onset_offset_frame_velocity_CRNN, EDWARDS),
}
NUMPY_SAFE = [
    (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
    np.ndarray,
    np.dtype,
    np.dtypes.Int64DType,
    np.dtypes.Int32DType,
    np.dtypes.Float64DType,
    np.dtypes.Float32DType,
    np.dtypes.BoolDType,
]


def load(name):
    """A PianoTranscription that runs the named model, with the package's thresholds."""
    cls, path = CKPT[name]
    pt = PianoTranscription.__new__(PianoTranscription)
    pt.segment_samples = 16000 * 10
    pt.frames_per_second = config.frames_per_second
    pt.classes_num = config.classes_num
    pt.onset_threshold = 0.3
    pt.offset_threshod = 0.3  # sic, the attribute name the package reads
    pt.frame_threshold = 0.1
    pt.pedal_offset_threshold = 0.2
    m = cls(frames_per_second=pt.frames_per_second, classes_num=pt.classes_num)
    # Edwards's file also stores the training sampler's numpy state; allow only the
    # numpy array and dtype types needed to read it, nothing that can run code
    with torch.serialization.safe_globals(NUMPY_SAFE):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
    pt.model = m.cuda().eval()
    return pt
