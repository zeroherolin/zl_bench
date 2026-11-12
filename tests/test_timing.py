import pytest

import _cudagraph_plugin as cg


def test_graph_plan_uncapped_is_do_bench_cudagraph():
    assert cg.graph_plan(0.01, 100, 0) == (10000, 1)
    assert cg.graph_plan(0.0, 100, 0) == (1000, 1)
    assert cg.graph_plan(500.0, 100, 0) == (1, 1)


def test_graph_plan_cap_keeps_the_total_call_count():
    calls, replays = cg.graph_plan(0.01, 100, 1000)
    assert calls == 1000 and replays == 10
    calls, replays = cg.graph_plan(0.03, 100, 1000)  # 3333 calls -> 1000 x 4
    assert calls == 1000 and replays == 4
    assert cg.graph_plan(1.0, 100, 1000) == (100, 1)


def test_max_calls_per_graph(monkeypatch):
    monkeypatch.delenv("GRAPH_MAX_CALLS", raising=False)
    assert cg.max_calls_per_graph() == cg.DEFAULT_MAX_CALLS
    monkeypatch.setenv("GRAPH_MAX_CALLS", "0")
    assert cg.max_calls_per_graph() == 0
    for bad in ("many", "-5", "1e3"):
        monkeypatch.setenv("GRAPH_MAX_CALLS", bad)
        with pytest.raises(ValueError, match="GRAPH_MAX_CALLS"):
            cg.max_calls_per_graph()


def test_summarize_matches_do_bench_modes():
    times = [3.0, 1.0, 2.0, 4.0]
    assert cg.summarize(times, None, "median") == 2.5
    assert cg.summarize(times, None, "mean") == 2.5
    assert cg.summarize(times, None, "min") == 1.0
    assert cg.summarize(times, None, "all") == times
    assert cg.summarize(times, [0.5], "mean") == 2.5
    assert cg.summarize(times, [0.0, 1.0], "mean") == [1.0, 4.0]


@pytest.mark.parametrize("text, fatal", [
    ("CUDA error: an illegal memory access was encountered", True),
    ("operation not permitted when stream is capturing. Compile with `TORCH_USE_CUDA_DSA` to "
     "enable device-side assertions.", False),
])
def test_fatal_errors(text, fatal):
    assert cg._is_fatal_cuda_error(RuntimeError(text)) is fatal


def test_timing_scope():
    import _timing_scope as ts
    seen = []
    wrapped = ts.timed(lambda: seen.append(ts.in_measurement()))
    assert not ts.in_measurement()
    wrapped()
    assert seen == [True] and not ts.in_measurement()


def test_parallel_warmup_gpu_count(monkeypatch, capsys):
    import _parallel_warmup_plugin as warmup

    monkeypatch.delenv("FLAGGEMS_PERF_WARMUP_WORKER", raising=False)
    for off in ("", "0", "1"):  # unset, or the documented ways to turn it off: silent
        monkeypatch.setenv("PARALLEL_WARMUP_GPUS", off)
        assert warmup._requested_gpus() == 0 and capsys.readouterr().out == ""
    for bad in ("all", "-2", "2.5"):  # asked for and not given: said so
        monkeypatch.setenv("PARALLEL_WARMUP_GPUS", bad)
        assert warmup._requested_gpus() == 0
        assert "is not a GPU count" in capsys.readouterr().out


def test_parallel_warmup_splits_every_entry_of_the_shape_set():
    import _parallel_warmup_plugin as warmup

    # The test is fused_marlin_moe_w4a16_mxfp4; the benchmark reads the entry named
    # after its op_name. Every entry is split, so the shards hold the one it reads.
    entries = {
        "fused_marlin_moe_w4a16": {"shape_desc": "M, E, K", "shapes": [[1], [2], [3], [4], [5]]},
        "Benchmark": {"shapes": [[9, 9]]}}
    shards = warmup._shard_entries(entries, 3)
    assert [s["fused_marlin_moe_w4a16"]["shapes"] for s in shards] == [[[1], [4]], [[2], [5]], [[3]]]
    assert all(s["fused_marlin_moe_w4a16"]["shape_desc"] == "M, E, K" for s in shards)
    # no shard is left without an entry: upstream would fall back to other shapes
    assert [s["Benchmark"]["shapes"] for s in shards] == [[[9, 9]]] * 3
    together = [shape for s in shards for shape in s["fused_marlin_moe_w4a16"]["shapes"]]
    assert sorted(together) == [[1], [2], [3], [4], [5]]


def test_parallel_warmup_reads_only_entries_with_shapes(tmp_path):
    import _parallel_warmup_plugin as warmup

    path = tmp_path / "set.yaml"
    path.write_text("mm:\n  shapes:\n  - [1, 2]\n  - [3, 4]\nempty:\n  shape_desc: M\nodd: 3\n")
    assert warmup._read_shape_yaml(str(path)) == {"mm": {"shapes": [[1, 2], [3, 4]]}}


def test_a_call_that_fails_by_itself_is_not_a_capture_fallback(monkeypatch, capsys):
    # The autotuner times every config of a sweep through do_bench and drops those
    # that raise (too much shared memory, a failed compile). Such a call is no
    # measurement that could not be captured: no marker, and no second attempt
    # through plain do_bench, which would only raise the same error again.
    import sys
    import types

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake)
    second_attempts = []
    monkeypatch.setattr(cg, "_ORIGINAL_DO_BENCH", lambda fn, **kw: second_attempts.append(fn))
    calls = []

    def rejected_config():
        calls.append(1)
        raise RuntimeError("out of resource: shared memory, Required: 262200, Hardware limit: 232448")

    with pytest.raises(RuntimeError, match="out of resource"):
        cg._patched_do_bench(rejected_config, warmup=25, rep=100)
    assert len(calls) == 1 and second_attempts == []
    assert "BENCHMARK_DIRECT_NO_CUDAGRAPH" not in capsys.readouterr().out


def test_a_call_that_cannot_be_captured_falls_back_with_a_marker(monkeypatch, capsys):
    import sys
    import types

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv("FLAGGEMS_PERF_CURRENT_OP", "nonzero")
    monkeypatch.setattr(cg, "_ORIGINAL_DO_BENCH", lambda fn, **kw: 0.25)
    monkeypatch.setattr(cg, "_find_host_sync", lambda run: "called a synchronizing CUDA operation")
    assert cg._patched_do_bench(lambda: None, warmup=0, rep=100) == 0.25
    out = capsys.readouterr().out
    assert "BENCHMARK_DIRECT_NO_CUDAGRAPH op=nonzero phase=host_sync reason=_HostSync" in out
