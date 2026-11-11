"""Read an object-dtype .npy (a pickled dict of arrays) without running arbitrary code.

np.load(allow_pickle=True) unpickles whatever the file says. This unpickler only
builds numpy arrays, dtypes, strings, numbers, lists and dicts; any other global
in the stream raises.
"""

import io
import pickle

import numpy as np

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return ALLOWED[(module, name)]
        if module.startswith("numpy") and name.endswith("DType"):
            import numpy.dtypes as d

            return getattr(d, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load(path):
    """The object stored in a single-object .npy file."""
    with open(path, "rb") as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)
        return _Restricted(io.BytesIO(f.read())).load()
