"""ArtifactNet v9.4 ONNX inference build (Oh, arXiv 2604.16254), as its model card runs it.

44.1 kHz mono, 4 s chunks, P(AI) per chunk, the song verdict is the median.
"""

import numpy as np
import onnxruntime as ort

from paths import ARTIFACTNET

SR = 44100
CHUNK = 4 * SR
_sess = None


def p_ai(x):
    """P(AI) of each whole 4 s chunk of mono 44.1 kHz x."""
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(
            str(ARTIFACTNET),
            providers=["CPUExecutionProvider"],
        )
    out = []
    for s in range(0, len(x) - CHUNK + 1, CHUNK):
        c = x[s : s + CHUNK].reshape(1, -1).astype(np.float32)
        out.append(float(np.ravel(_sess.run(None, {"audio": c})[0])[0]))
    return np.array(out)
