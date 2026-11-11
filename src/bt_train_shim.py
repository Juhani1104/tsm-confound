"""numpy 2.5 dropped two private helpers that beat_this's memory-mapped npz reader calls;
importing this module puts back equivalents built on the public header readers."""

import numpy as np

if not hasattr(np.lib.format, "_check_version"):

    def _check_version(v):
        if v not in ((1, 0), (2, 0), (3, 0)):
            raise ValueError(f"unsupported npy version {v}")

    np.lib.format._check_version = _check_version
if not hasattr(np.lib.format, "_read_array_header"):
    np.lib.format._read_array_header = lambda fp, v: (
        np.lib.format.read_array_header_1_0
        if v == (1, 0)
        else np.lib.format.read_array_header_2_0
    )(fp)
