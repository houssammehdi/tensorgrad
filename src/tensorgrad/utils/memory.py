"""Process-wide allocator tuning for NumPy-heavy training loops."""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

__all__ = ["retain_freed_memory"]

# mallopt(3) parameters from glibc's <malloc.h>.
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3


def retain_freed_memory(mmap_threshold: int = 64 << 20, trim_threshold: int = 1 << 30) -> bool:
    """Keep memory freed by NumPy inside the process instead of returning it to the OS.

    NumPy allocates array buffers with ``malloc``. glibc serves large requests with
    ``mmap`` and gives freed memory at the top of the heap back to the kernel, so a training
    loop that frees its activations after every step page-faults all of them back in on the
    next one. On the GPT benchmark in ``docs/performance.md`` that costs about 11,000 page
    faults and a fifth of the step time. This call raises glibc's thresholds so large
    arrays come from the heap and freed memory stays mapped for reuse.

    Both thresholds must change together: raising only the trim threshold also freezes
    glibc's adaptive mmap threshold at 128 KB, which makes things worse.

    Args:
        mmap_threshold: Requests up to this many bytes are served from the heap.
        trim_threshold: Free memory at the top of the heap is only returned to the OS once
            it exceeds this many bytes.

    Returns:
        ``True`` if glibc accepted both settings; ``False`` on other platforms or C
        libraries (musl, macOS, Windows), where nothing changes.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    except OSError:
        return False
    mallopt = getattr(libc, "mallopt", None)
    if mallopt is None:
        return False
    mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
    mallopt.restype = ctypes.c_int
    return bool(mallopt(_M_MMAP_THRESHOLD, mmap_threshold)) and bool(
        mallopt(_M_TRIM_THRESHOLD, trim_threshold)
    )
