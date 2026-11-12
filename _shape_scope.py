"""Which shape a benchmark's input iterator is at, shared by two plugins.

An input iterator walks its shapes and yields one input for each, or several: an
input_fn may yield two variants per shape. _ir_dump_plugin names kernels and result rows
after the shape they belong to, and the position of an input in the iteration does not
tell it. `walk(shapes)` iterates the shapes and reports each to the listener that plugin
registers. _ir_dump_plugin walks Benchmark.shapes this way while get_input_iter runs;
_shape_override_plugin walks the yaml shapes it feeds to iterators that read none.
Without a listener walk() is a plain iteration.
"""

from __future__ import annotations

listener = None  # called with each shape as a walk reaches it, and with None after the last


def walk(shapes):
    """Iterate shapes, telling the listener where the iteration is."""
    for shape in shapes:
        if listener is not None:
            listener(shape)
        yield shape
    # Only when the walk ran to its end: an iterator that is dropped halfway (a failed
    # measurement) is finalized whenever the collector gets to it, possibly while
    # another benchmark is being fed.
    if listener is not None:
        listener(None)
