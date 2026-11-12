"""pytest plugin: swap `triton.testing.do_bench` for CUDA-graph timing.

Loaded via `-p _cudagraph_plugin` in run_pytest.sh's PLUGINS list (comment that
line out to time with plain do_bench). Replaces do_bench at session configure,
after Triton is loaded but before any benchmark calls it. Per measurement:

1. synchronize, so inputs still in flight on the default stream are complete;
2. warm fn up for the caller's warmup budget (_warmup_before_capture);
3. run fn once with host syncs turned into errors: a call that synchronizes
   with the host cannot be captured, so it goes straight to the fallback
   without an aborted capture (_find_host_sync);
4. time it with _graph_bench: triton's do_bench_cudagraph scheme (a graph of
   n back-to-back calls, replayed 10 times, median), with at most
   GRAPH_MAX_CALLS calls per graph; larger counts are reached by replaying
   the graph several times per sample, so a sample still spans the rep
   budget. An out-of-memory capture is retried with one call per graph.

A call that cannot be captured falls back to plain do_bench and prints a
BENCHMARK_DIRECT_NO_CUDAGRAPH marker into the log; a call that fails by itself,
during warmup and before any capture, raises as it would under do_bench.
Captures run under a scratch CUDA RNG state (_capture_with_scratch_rng); an
aborted capture is released (_release_memory) and followed by one untimed call
(_absorb_aborted_capture) before the fallback.
"""

from __future__ import annotations

import gc
import math
import os
import statistics
import sys

import _timing_scope

_ORIGINAL_DO_BENCH = None  # triton.testing.do_bench, saved in pytest_configure
_MAX_CALLS_ENV = "GRAPH_MAX_CALLS"
DEFAULT_MAX_CALLS = 1000
MIN_SAFE_CALLS = 250  # smaller graphs read 1 us kernels too fast (graph_plan)
N_SAMPLES = 10  # graph replays per measurement, as in do_bench_cudagraph


def max_calls_per_graph() -> int:
    """Cap on the calls unrolled into one graph; 0 = no cap."""
    value = os.environ.get(_MAX_CALLS_ENV, "").strip()
    if not value:
        return DEFAULT_MAX_CALLS
    try:
        cap = int(value)
    except ValueError:
        cap = -1
    if cap < 0:
        raise ValueError(f"{_MAX_CALLS_ENV}={value!r}: expected a call count, or 0 for no cap")
    return cap


def graph_plan(estimate_ms: float, rep_ms: float, cap: int):
    """(calls per graph, replays per sample) for a call taking estimate_ms.

    The uncapped plan is do_bench_cudagraph's: rep_ms worth of calls in one
    graph, one replay per sample. Capture time grows linearly with the calls
    in the graph (about 12 ms per call for conv1d with padding), so with a cap
    the graph holds fewer calls and each sample replays it back to back until
    it covers the same number of calls. From 250 calls per graph up the
    replays add nothing measurable, even for 1 us kernels; smaller graphs read
    such kernels too fast. The plan is the same on both sides of an A/B."""
    n_total = 1000 if estimate_ms <= 0 else max(1, int(rep_ms / estimate_ms))
    n_calls = min(n_total, cap) if cap > 0 else n_total
    return n_calls, math.ceil(n_total / n_calls)


def summarize(times, quantiles, return_mode):
    """The same summaries as triton.testing.do_bench."""
    if quantiles is not None:
        ordered = sorted(times)
        out = []
        for q in quantiles:
            pos = q * (len(ordered) - 1)
            lo, hi = math.floor(pos), math.ceil(pos)
            out.append(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))
        return out[0] if len(out) == 1 else out
    if return_mode == "all":
        return list(times)
    return {"min": min, "max": max, "mean": statistics.fmean,
            "median": statistics.median}[return_mode](times)


def _run_once(fn, grad_to_none):
    """One call of fn, with the gradients the caller wants reset cleared first."""
    if grad_to_none is not None:
        for x in grad_to_none:
            x.grad = None
    fn()


def _is_fatal_cuda_error(exc: Exception) -> bool:
    """True for CUDA context errors (illegal address, OOM, device-side assert):
    the context is poisoned, so re-raise rather than fall back and record a bogus
    latency as success.

    Markers must match the error text itself, not torch's generic footer: torch
    appends "Compile with `TORCH_USE_CUDA_DSA` to enable device-side
    assertions." to EVERY AcceleratorError, so a loose "device-side assert"
    marker would make every capture failure fatal and disable the do_bench
    fallback."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "illegal memory access",
        "cudaerrorillegaladdress",
        "out of memory",
        "device-side assert triggered",
        "an illegal instruction",
        "misaligned address",
        "uncorrectable ecc",
    )
    return any(m in msg for m in markers)


def _warmup_before_capture(fn, warmup_ms, grad_to_none):
    """Warm fn up before cudagraph capture.

    do_bench_cudagraph only warms up 5 iterations internally — too few for the
    MoE kernels, where the first calls trigger Triton autotune config
    compilation, libtuner selection and lazy JIT/init. If that work leaks into
    the captured graph or the first timed iteration the latency is unstable
    run-to-run (most visibly on the small-M multi-kernel path). Absorb the
    one-off compile with a throwaway run, then loop for the caller's warmup time
    budget so the graph is captured at steady state.

    The loop is clamped to 5..200 iterations: the cap keeps sub-ms kernels from
    spinning through a large budget (FlagGems' Config.warm_up defaults to
    1000 ms) — 200 iterations settles them fine in practice — and the floor
    covers kernels whose single run exceeds the budget. If first-run jitter
    reappears on a new op, raising the cap is the first knob to try.
    """
    import time

    import torch

    # First call absorbs autotune/JIT compilation (can take seconds); discard it.
    _run_once(fn, grad_to_none)
    torch.cuda.synchronize()
    # Size the warmup loop from one timed steady-state run. dt_ms includes
    # launch overhead, so n errs low for tiny kernels; warmup needn't be exact.
    t0 = time.perf_counter()
    _run_once(fn, grad_to_none)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    budget = warmup_ms if (warmup_ms and warmup_ms > 0) else 25.0
    n = int(budget / dt_ms) if dt_ms > 0 else 25
    n = max(5, min(n, 200))
    for _ in range(n):
        _run_once(fn, grad_to_none)
    torch.cuda.synchronize()


def _capture_with_scratch_rng(call):
    """Run a graph capture under a throwaway CUDA RNG state.

    torch.cuda.graph's capture_begin unconditionally registers the default CUDA
    RNG generator and sets its capturing flag; when a capture aborts midway the
    epilogue may never run, the flag sticks, and every later CUDA RNG call in
    the process (the next shape's torch.randn included) raises "Offset
    increment outside graph capture". Repairing after the fact is unreliable —
    when the poisoning surfaces depends on which stage the capture died in —
    so prevent it instead: swap in a scratch state via graphsafe_set_state
    before capturing and restore the original state object afterwards, success
    or failure. The real generator never takes part in a capture, so poisoning
    can only land on the discarded scratch state, and _seed_plugin determinism
    is left untouched."""
    import torch
    dev = torch.cuda.current_device()
    gen = torch.cuda.default_generators[dev]
    saved = gen.graphsafe_get_state()
    scratch = torch.Generator(device="cuda")
    scratch.manual_seed(gen.initial_seed())
    gen.graphsafe_set_state(scratch)
    try:
        return call()
    finally:
        gen.graphsafe_set_state(saved)


def _release_memory():
    """Free what an aborted or failed capture left behind. The graph and its
    private memory pool stay alive in reference cycles until the collector
    runs; for inputs of tens of GiB a few of them leave too little memory for
    the next measurement. Call outside any except block: a live exception's
    traceback still references the graph."""
    import torch
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _find_host_sync(run):
    """Error text if one call of run synchronizes with the host, else None.

    A host sync (tensor.item(), nonzero, a device-to-host copy) is illegal
    during capture, so such a call can only be timed with plain do_bench.
    Detecting it up front with torch's sync debug mode avoids an aborted
    capture and the memory it would strand."""
    import torch
    mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    except RuntimeError as exc:
        if "synchroniz" not in str(exc).lower():
            raise
        return " ".join(str(exc).split())
    finally:
        torch.cuda.set_sync_debug_mode(mode)
    torch.cuda.synchronize()
    return None


class _HostSync(RuntimeError):
    pass


def _graph_bench(fn, rep, grad_to_none, quantiles, return_mode):
    """CUDA-graph timing, following triton.testing.do_bench_cudagraph, with at
    most max_calls_per_graph() calls per graph (graph_plan)."""
    import torch

    def capture(n_calls):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_calls):
                _run_once(fn, grad_to_none)
        torch.cuda.synchronize()
        return g

    caller = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(caller)
    with torch.cuda.stream(stream):
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            fn()
        end.record()
        torch.cuda.synchronize()
        n_calls, replays = graph_plan(start.elapsed_time(end) / 5, rep, max_calls_per_graph())
        graph = None
        try:
            graph = capture(n_calls)
        except torch.OutOfMemoryError:
            pass  # retried below, once the failed graph is no longer referenced
        if graph is None:
            _release_memory()
            op = os.environ.get("FLAGGEMS_PERF_CURRENT_OP", "<unknown>")
            print(f"CUDAGRAPH_OOM_RETRY op={op} calls_per_graph=1", flush=True)
            n_total = n_calls * replays
            n_calls, replays = 1, n_total
            graph = capture(1)  # a second OOM is real: the caller treats it as fatal
        times = []
        for _ in range(N_SAMPLES):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / (n_calls * replays))
    return summarize(times, quantiles, return_mode)


def _absorb_aborted_capture(fn):
    """Run fn once, discarding a non-fatal error, after a capture aborted.

    Some libraries keep per-handle state that the aborted capture leaves
    broken, and report it on their next call only: after a failed capture of
    linalg_matrix_rank(hermitian=True), the next cusolverDnXsyevd returns
    CUSOLVER_STATUS_EXECUTION_FAILED once and then works again, on any stream.
    Clearing the CUDA runtime's last error does not help, it is cusolver's own
    state. Without this throwaway call the plain do_bench fallback would fail
    the op although nothing is wrong with it.
    """
    import torch
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_fatal_cuda_error(exc):
            raise


def _patched_do_bench(fn, warmup=25, rep=100, grad_to_none=None,
                      quantiles=None, return_mode="mean"):
    """Drop-in replacement for triton.testing.do_bench using CUDA-graph timing."""
    op = os.environ.get("FLAGGEMS_PERF_CURRENT_OP", "<unknown>")
    fn = _timing_scope.timed(fn)  # its launches are the measurement's (_ir_dump_plugin)
    # The graph runs on a fresh side stream; input tensors produced just
    # before (quantize/topk in the input iter) may still be in flight, and
    # racing on them intermittently kills the context (illegal instruction
    # from a device trap on garbage indices). Sync before switching streams.
    import torch
    torch.cuda.synchronize()
    # Settle autotune/JIT before capture (rationale: _warmup_before_capture). Outside
    # the fallback: fn runs here as it would under plain do_bench, so an error is fn's
    # own and plain do_bench would only raise it again. The autotuner times every
    # config of a sweep through do_bench and drops the ones that raise (too much
    # shared memory for the GPU, a failed compile); falling back would run each of
    # them once more for the same error, and mark it as a measurement that could not
    # be captured.
    _warmup_before_capture(fn, warmup, grad_to_none)
    phase = "host_sync"  # which step rejected the graph, for the fallback marker
    failure = None
    try:
        sync = _find_host_sync(lambda: _run_once(fn, grad_to_none))
        if sync:
            raise _HostSync(sync)
        phase = "capture"
        return _capture_with_scratch_rng(
            lambda: _graph_bench(fn, rep, grad_to_none, quantiles, return_mode))
    except Exception as exc:
        # Re-raise genuine device failures; anything else falls back to plain
        # do_bench, with the failing phase in the marker.
        if _is_fatal_cuda_error(exc):
            raise
        failure = (type(exc).__name__, " ".join(str(exc).split())[:160])
    # Out of the except block, so the failed graph is unreferenced and can be freed.
    from _term_style import YELLOW, paint
    print(
        paint(f"BENCHMARK_DIRECT_NO_CUDAGRAPH op={op} phase={phase} "
              f"reason={failure[0]} detail={failure[1]!r}", YELLOW),
        flush=True,
    )
    if phase == "capture":
        _release_memory()
        _absorb_aborted_capture(fn)
    return _ORIGINAL_DO_BENCH(fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none,
                              quantiles=quantiles, return_mode=return_mode)


def pytest_configure(config):
    global _ORIGINAL_DO_BENCH
    import pytest
    import triton.testing

    from _term_style import tag
    try:
        cap = max_calls_per_graph()
    except ValueError as exc:
        # a usage error: pytest prints the one line, not an internal-error traceback
        raise pytest.UsageError(str(exc)) from None
    _ORIGINAL_DO_BENCH = triton.testing.do_bench
    triton.testing.do_bench = _patched_do_bench
    print(f"{tag('[cudagraph-plugin]')} triton.testing.do_bench → CUDA-graph timing "
          f"({f'at most {cap} calls per graph' if cap > 0 else 'no cap on calls per graph'})",
          file=sys.stderr, flush=True)
    if 0 < cap < MIN_SAFE_CALLS:
        print(f"{tag('[cudagraph-plugin]')} warning: {_MAX_CALLS_ENV}={cap} is below "
              f"{MIN_SAFE_CALLS}; graphs that small read sub-microsecond kernels too fast",
              file=sys.stderr, flush=True)
