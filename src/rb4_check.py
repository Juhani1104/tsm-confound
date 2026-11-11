"""Is the large-block drift still in Rubber Band 4.0? Calls librubberband's C API through ctypes.

usage: LD_LIBRARY_PATH=DIR python src/rb4_check.py DIR/librubberband.so.3

The library is Ubuntu's librubberband3 4.0.0+dfsg-2ubuntu1 (unpacked from the .deb, with its
libfftw3 and libsamplerate dependencies). Same options as pedalboard's time_stretch default
(offline, R3 Finer engine, threading never, channels together, pitch high quality, formants
preserved), studied in 4096-sample chunks, then processed in blocks of a given size, as
pedalboard does with the process size limit. Click train as in rb_offset.py, 60 s, 16 kHz.
"""

import ctypes as C
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rb_offset import burst_lags, bursts  # noqa: E402

rb = C.CDLL(sys.argv[1])
P = C.c_void_p
FP = C.POINTER(C.POINTER(C.c_float))
rb.rubberband_new.restype = P
rb.rubberband_new.argtypes = [C.c_uint, C.c_uint, C.c_int, C.c_double, C.c_double]
for f, a in (("rubberband_set_expected_input_duration", [P, C.c_uint]), ("rubberband_set_max_process_size", [P, C.c_uint]),
             ("rubberband_study", [P, FP, C.c_uint, C.c_int]), ("rubberband_process", [P, FP, C.c_uint, C.c_int]),
             ("rubberband_retrieve", [P, FP, C.c_uint]), ("rubberband_available", [P]), ("rubberband_delete", [P]),
             ("rubberband_get_process_size_limit", [P]), ("rubberband_get_engine_version", [P])):  # fmt: skip
    getattr(rb, f).argtypes = a
rb.rubberband_available.restype = C.c_int
rb.rubberband_retrieve.restype = C.c_uint
rb.rubberband_get_process_size_limit.restype = C.c_uint
OPTS = 0x0 | 0x10000 | 0x10000000 | 0x2000000 | 0x20000000 | 0x1000000
SR = 16000


def ptrs(a):
    """A float** for one channel, as the C API takes it."""
    p = (C.POINTER(C.c_float) * 1)()
    p[0] = a.ctypes.data_as(C.POINTER(C.c_float))
    return p


def run(x, time_ratio, pitch_scale, block):
    """(output, size limit, engine version, block) for x processed in `block` chunks."""
    st = rb.rubberband_new(SR, 1, OPTS, time_ratio, pitch_scale)
    limit = rb.rubberband_get_process_size_limit(st)
    block = min(block, limit)
    rb.rubberband_set_expected_input_duration(st, len(x))
    rb.rubberband_set_max_process_size(st, block)
    for i in range(0, len(x), 4096):
        c = np.ascontiguousarray(x[i : i + 4096])
        rb.rubberband_study(st, ptrs(c), len(c), int(i + 4096 >= len(x)))
    out = []

    def drain():
        """Move everything the stretcher has ready into out."""
        while (n := rb.rubberband_available(st)) > 0:
            buf = np.zeros(n, np.float32)
            got = rb.rubberband_retrieve(st, ptrs(buf), n)
            out.append(buf[:got])

    for i in range(0, len(x), block):
        c = np.ascontiguousarray(x[i : i + block])
        rb.rubberband_process(st, ptrs(c), len(c), int(i + block >= len(x)))
        drain()
    drain()
    eng = rb.rubberband_get_engine_version(st)
    rb.rubberband_delete(st)
    return np.concatenate(out), limit, eng, block


x, t0 = bursts(SR, dur=60.0)
x = x.astype(np.float32)
for block in (4096, 16384, 65536, 1 << 30):
    for cents, tr in ((10, 1.0), (50, 1.0), (200, 1.0), (0, 1 / 1.05)):
        y, limit, eng, b = run(x, tr, 2 ** (cents / 1200), block)
        l = burst_lags(y, SR, t0 * tr)
        k = np.searchsorted(t0, [5, 15, 30, 45, 55])
        print(
            f"R{eng} block {b:6d} (limit {limit}) cents {cents:+4d} ratio {tr:.3f} len {len(y) / len(x):.4f}  "
            + " ".join(f"{t0[j]:.0f}s:{l[j]:+.0f}" for j in k),
            flush=True,
        )
