"""Which kernel launches belong to a measurement, shared by two plugins.

_cudagraph_plugin wraps the function it times so that `depth` is above zero while that
function runs (warmup, capture and fallback timing alike); _ir_dump_plugin attributes a
launch to the current result row only then. Launches the timing harness makes on its
own, such as the fills torch issues when a CUDA graph capture begins, stay out of the
row. Without _cudagraph_plugin nothing is marked as measured, and _ir_dump_plugin gives a
row every launch made while it was current (scope "row" in rows.tsv).
"""

from __future__ import annotations

import functools

depth = 0


def timed(fn):
    """fn, marked as the measured function while it runs."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global depth
        depth += 1
        try:
            return fn(*args, **kwargs)
        finally:
            depth -= 1
    return wrapper


def in_measurement() -> bool:
    return depth > 0
