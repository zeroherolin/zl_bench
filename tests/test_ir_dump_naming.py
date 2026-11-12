import _binary_id
import _ir_dump_plugin as dump


def test_abbrev_and_fmt_value():
    assert dump._abbrev("BLOCK_SIZE_M") == "BSM"
    assert dump._abbrev("BLOCK_M2") == "BM2"
    assert dump._abbrev("SPLIT") == "SPLIT"
    assert dump._fmt_value(True) == "T"
    assert dump._fmt_value(1) == "1"  # not "T": 1 == True in Python
    assert dump._fmt_value(None) == "-"
    assert dump._fmt_value("triton.language.bfloat16") == "bfloat16"


def _variant(h, warps=4, stages=3, signature=None, **constexprs):
    return {"hash": h, "warps": warps, "stages": stages, "constexprs": constexprs,
            "signature": signature or {}, "attrs": {}}


def test_name_group_uses_only_varying_constexprs():
    items = [_variant("AAAA1111", BLOCK_SIZE_M=64, GROUP=8),
             _variant("BBBB2222", BLOCK_SIZE_M=128, GROUP=8)]
    named, legend = dump._name_group(items)
    assert sorted(name for _, name in named) == ["BSM128_w4s3.ttgir", "BSM64_w4s3.ttgir"]
    assert legend == {"BLOCK_SIZE_M": "BSM"}


def test_name_group_disambiguates_by_dtype_then_hash():
    by_dtype = [_variant("AAAA1111", signature={"x": "*fp16"}, B=1),
                _variant("BBBB2222", signature={"x": "*bf16"}, B=1)]
    named, _ = dump._name_group(by_dtype)
    assert sorted(name for _, name in named) == ["w4s3__bf16.ttgir", "w4s3__fp16.ttgir"]

    by_hash = [_variant("AAAA1111", B=1), _variant("BBBB2222", B=1)]
    named, _ = dump._name_group(by_hash)
    assert sorted(name for _, name in named) == ["w4s3__AAAA1111.ttgir", "w4s3__BBBB2222.ttgir"]


def test_rows_tsv_prefers_the_timed_kernels(tmp_path, monkeypatch):
    monkeypatch.setattr(dump, "_tables",
                        [("softmax", "torch.float16", [(0, "4x4"), (1, "8x8"), (None, "x")])])
    monkeypatch.setattr(dump, "_row_timed", {0: {"H2", "H1"}})
    monkeypatch.setattr(dump, "_row_any", {0: {"H1", "H2", "H3"}, 1: {"H4"}})
    dump._write_rows(tmp_path / "rows.tsv")
    lines = (tmp_path / "rows.tsv").read_text().splitlines()
    assert lines[0] == "op_name\tdtype\trow\tshape\tkernels\tscope\tbinaries"
    assert lines[1:] == ["softmax\ttorch.float16\t0\t4x4\tH1,H2\ttimed\t",
                         "softmax\ttorch.float16\t1\t8x8\tH4\trow\t",
                         "softmax\ttorch.float16\t2\tx\t?\t-\t"]


def _cached_variant(cache, name, kernel="softmax_kernel", cubin=b"\x7fELF-binary", shared=1024):
    folder = cache / name
    folder.mkdir(parents=True)
    (folder / f"{kernel}.cubin").write_bytes(cubin)
    (folder / f"{kernel}.json").write_text(
        '{"shared": %d, "num_warps": 4, "num_stages": 3, "hash": "%s"}' % (shared, name))


def test_rows_tsv_carries_the_binaries_the_row_ran(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _cached_variant(cache, "H1")
    _cached_variant(cache, "H2", cubin=b"\x7fELF-other")
    monkeypatch.setattr(dump, "_tables", [("softmax", "torch.float16", [(0, "4x4"), (1, "8x8")])])
    monkeypatch.setattr(dump, "_row_timed", {0: {"H2", "H1"}, 1: {"H3"}})
    monkeypatch.setattr(dump, "_row_any", {})
    dump._write_rows(tmp_path / "rows.tsv", cache)
    rows = [line.split("\t") for line in (tmp_path / "rows.tsv").read_text().splitlines()[1:]]
    launch = {"shared": 1024, "num_warps": 4}
    assert rows[0][4] == "H1,H2"
    assert rows[0][6] == ",".join(_binary_id.fingerprint("softmax_kernel", cubin, launch)
                                  for cubin in (b"\x7fELF-binary", b"\x7fELF-other"))
    assert rows[1][6] == "?"  # H3 is not in the cache: unknown, not a failed dump


def test_fingerprint_follows_the_binary_not_the_cache_hash(tmp_path):
    cache = tmp_path / "cache"
    _cached_variant(cache, "BUILD_A_HASH")
    _cached_variant(cache, "BUILD_B_HASH")  # another build, same cubin and launch metadata
    _cached_variant(cache, "OTHER_SHARED", shared=2048)
    _cached_variant(cache, "OTHER_KERNEL", kernel="log_softmax_kernel")
    a = dump._fingerprint(cache, "BUILD_A_HASH")
    assert a == dump._fingerprint(cache, "BUILD_B_HASH")
    others = [dump._fingerprint(cache, name) for name in ("OTHER_SHARED", "OTHER_KERNEL")]
    assert len(set([a] + others)) == 3
    assert dump._fingerprint(cache, "MISSING") == "?"


def test_short_label_is_unchanged():
    assert dump._label_of([64, 256, 1, 1, 8]) == "64x256x1x1x8"
    assert dump._label_of("mm") == "mm"


def test_nested_shape_keeps_its_dims_apart():
    # adaptive_max_pool3d: [input_shape, output_size]. Joined without a separator the
    # inner dims would read 225621414x21414, which is the label of another shape too.
    pool = [[2, 256, 2, 14, 14], [2, 14, 14]]
    assert dump._label_of(pool) == "2-256-2-14-14x2-14-14"
    assert dump._label_of([[22, 56, 2, 14, 14], [21, 4, 14]]) != dump._label_of(pool)
    assert dump._label_of([(1, 2), True, "x y"]) == "1-2xTruexxy"


def test_dump_replaces_the_ir_of_an_earlier_attempt(tmp_path, monkeypatch):
    # run_batch.py retries into the same run dir; the first attempt's files must not
    # pass for variants of the attempt that produced the result
    stale = tmp_path / "ttgir" / "4x4" / "old_kernel" / "w4s3.ttgir"
    stale.parent.mkdir(parents=True)
    stale.write_text("module")
    (tmp_path / "ttir" / "OLDHASH").mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(dump, "_tables", [])
    dump._dump_ir_guarded(str(cache), str(tmp_path))
    assert not stale.exists() and not (tmp_path / "ttir" / "OLDHASH").exists()
    assert (tmp_path / "ttgir" / "index.tsv").is_file() and (tmp_path / "rows.tsv").is_file()


def test_long_label_fits_a_directory_name_and_stays_distinct():
    # a varlen attention case carries its cu_seqlens: the plain join is far over the
    # 255-byte limit, and a name the filesystem rejects fails the whole dump
    a = dump._label_of(list(range(300)))
    b = dump._label_of(list(range(1, 301)))
    assert len(a) <= dump._MAX_LABEL and len(b) <= dump._MAX_LABEL
    assert a != b
    assert a.startswith("0x1x2x3x")   # still readable at the front


def test_dump_failure_is_reported_not_swallowed(tmp_path, capsys, monkeypatch):
    # the dump runs from an atexit hook, after pytest called the run a pass; an error
    # there must be visible, or the missing IR only surfaces in a later gate
    def boom(cache_dir, run_dir):
        raise OSError(36, "File name too long")

    monkeypatch.setattr(dump, "_dump_ir", boom)
    dump._dump_ir_guarded(str(tmp_path), str(tmp_path))
    err = capsys.readouterr().err
    assert "IR_DUMP_FAILED" in err
    assert "File name too long" in err


def test_rows_tsv_survives_a_failing_ir_dump(tmp_path, capsys, monkeypatch):
    # rows.tsv is what attributes result rows to kernels; the much larger IR copy
    # failing (a rejected path, a cache file that went away) must not take it along
    written = []

    def boom(cache_dir, run_dir):
        raise OSError(36, "File name too long")

    monkeypatch.setattr(dump, "_write_rows", lambda path, cache_dir=None: written.append(path))
    monkeypatch.setattr(dump, "_dump_ir", boom)
    dump._dump_ir_guarded(str(tmp_path), str(tmp_path))
    assert written, "rows.tsv was not written when the IR dump failed"
    err = capsys.readouterr().err
    assert "IR_DUMP_FAILED" in err and "ROWS_TSV_FAILED" not in err


class _Bench:
    """Stands for an upstream Benchmark: shapes, and a get_input_iter wrapped the way
    pytest_collection_finish wraps it."""

    def __init__(self, shapes, feed):
        self.shapes, self._feed = shapes, feed

    def inputs(self):
        return dump._wrap_input_iter(lambda bench, dtype: bench._feed(bench))(self, None)


def _labels(monkeypatch, bench):
    """The label in force while each input of the benchmark is measured."""
    monkeypatch.setattr(dump._shape_scope, "listener", dump._on_shape)
    return [(item, dump._current_shape) for item in bench.inputs()]


def test_inputs_are_labelled_by_the_shape_the_iterator_is_at(monkeypatch):
    def one_per_shape(bench):
        for shape in bench.shapes:
            yield shape

    def two_per_shape(bench):  # adaptive_max_pool3d: with and without indices
        for shape in bench.shapes:
            yield (shape, "plain")
            yield (shape, "indices")

    shapes = [(2, 3), (4, 5), (6, 7)]
    assert [lab for _, lab in _labels(monkeypatch, _Bench(shapes, one_per_shape))] == ["2x3", "4x5", "6x7"]
    # counting the inputs would label the second input of the first shape 4x5
    assert [lab for _, lab in _labels(monkeypatch, _Bench(tuple(shapes), two_per_shape))] == [
        "2x3", "2x3", "4x5", "4x5", "6x7", "6x7"]
    assert dump._current_shape is None  # nothing is being fed any more


def test_inputs_of_an_iterator_that_walks_no_shapes_are_numbered(monkeypatch):
    def hardcoded(bench):  # conv padding, pool: a list of cases of its own
        yield from ("a", "b", "c")

    def filtered_first(bench):  # igammac under float64: walks a filtered copy
        for shape in [s for s in bench.shapes if s[0] > 2]:
            yield shape

    shapes = [(2, 3), (4, 5), (6, 7)]
    # naming a shape here would name the wrong one: the inputs are not built from them
    assert _labels(monkeypatch, _Bench(shapes, hardcoded)) == [
        ("a", "input0"), ("b", "input1"), ("c", "input2")]
    assert _labels(monkeypatch, _Bench(shapes, filtered_first)) == [
        ((4, 5), "input0"), ((6, 7), "input1")]
    assert _labels(monkeypatch, _Bench(None, hardcoded))[0] == ("a", "input0")


def test_the_yaml_shapes_of_a_redirected_iterator_label_its_inputs(monkeypatch):
    # _shape_override_plugin feeds the yaml shapes to an iterator that reads none; the
    # benchmark's own self.shapes are then not what is measured
    def redirected(bench):
        for shape in dump._shape_scope.walk([(8, 8), (9, 9)]):
            yield shape

    assert [lab for _, lab in _labels(monkeypatch, _Bench([(1, 1)], redirected))] == ["8x8", "9x9"]


def test_the_benchmark_gets_its_own_shapes_back(monkeypatch):
    def looking(bench):
        for shape in bench.shapes:
            # what upstream does with them while inputs are fed still works
            assert bench.shapes[0] == (2, 3) and len(bench.shapes) == 2
            assert bench.shapes == [(2, 3), (4, 5)] and bench.shapes + [1] == [(2, 3), (4, 5), 1]
            yield shape

    monkeypatch.setattr(dump._shape_scope, "listener", dump._on_shape)
    shapes = [(2, 3), (4, 5)]
    bench = _Bench(shapes, looking)
    assert list(bench.inputs()) == shapes and bench.shapes is shapes
    it = bench.inputs()  # a measurement that fails drops the iterator halfway
    next(it)
    assert bench.shapes is not shapes
    it.close()
    assert bench.shapes is shapes


def test_without_the_ir_dump_a_walk_is_a_plain_iteration(monkeypatch):
    monkeypatch.setattr(dump._shape_scope, "listener", None)
    assert list(dump._shape_scope.walk([(1,), (2,)])) == [(1,), (2,)]


def _launch(monkeypatch, row):
    """One kernel launch while `row` is the current result row."""
    monkeypatch.setattr(dump, "_fn_to_cachedir", {"fn": "HASH"})
    monkeypatch.setattr(dump, "_current_row", row)

    class Launch:
        data = {"function": "fn"}

    dump._launch_hook(Launch())


def test_rows_say_whether_their_kernels_were_seen_in_the_timed_function(tmp_path, monkeypatch):
    for name in ("_row_timed", "_row_any"):
        monkeypatch.setattr(dump, name, dump.defaultdict(set))
    monkeypatch.setattr(dump, "_launches", {})
    monkeypatch.setattr(dump, "_tables", [("softmax", "torch.float16", [(0, "4x4"), (1, "8x8")])])
    dump._timing_scope.timed(lambda: _launch(monkeypatch, 0))()  # inside do_bench's fn
    _launch(monkeypatch, 1)  # no timed function: a run without _cudagraph_plugin
    dump._write_rows(tmp_path / "rows.tsv")
    lines = (tmp_path / "rows.tsv").read_text().splitlines()[1:]
    assert [ln.split("\t")[3:6] for ln in lines] == [["4x4", "HASH", "timed"], ["8x8", "HASH", "row"]]


class _Metric:
    """Stands for upstream's BenchmarkMetrics: the plugin wraps its __init__."""

    def __init__(self):
        pass


def _tracking(monkeypatch):
    """Fresh row tracking; returns (new_row, launch): new_row() makes a result row the
    way upstream does, launch(name, timed=False) launches the kernel in cache dir name."""
    for name in ("_row_timed", "_row_any"):
        monkeypatch.setattr(dump, name, dump.defaultdict(set))
    monkeypatch.setattr(dump, "_row_before", {})
    monkeypatch.setattr(dump, "_unclaimed", {True: set(), False: set()})
    monkeypatch.setattr(dump, "_launches", {})
    monkeypatch.setattr(dump, "_row_count", 0)
    monkeypatch.setattr(dump, "_current_row", None)
    monkeypatch.setattr(dump, "_fn_to_cachedir", {"A": "A", "B": "B", "C": "C"})
    monkeypatch.setattr(_Metric, "__init__", dump._wrap_metrics_init(_Metric.__init__))

    def launch(name, timed=False):
        class Launch:
            data = {"function": name}

        if timed:
            dump._timing_scope.timed(lambda: dump._launch_hook(Launch()))()
        else:
            dump._launch_hook(Launch())

    return _Metric, launch


def _rows_of(tmp_path, monkeypatch, metrics):
    monkeypatch.setattr(dump, "_tables", [("op", "torch.float16", [(m._zl_row, "s") for m in metrics])])
    dump._write_rows(tmp_path / "rows.tsv")
    return [ln.split("\t")[4:6] for ln in (tmp_path / "rows.tsv").read_text().splitlines()[1:]]


def test_a_row_made_after_its_measurement_still_gets_its_kernels(tmp_path, monkeypatch):
    # Most benchmarks make the row and then measure. One that measures first (a
    # multi-rank benchmark, mm_w8a8_int8) gets its kernels all the same: the gate reads
    # a row without kernels as one no compiler change can touch.
    new_row, launch = _tracking(monkeypatch)
    dump._new_input()
    launch("A")  # building the input
    launch("B", timed=True)  # the measurement, before any row exists
    first = new_row()
    launch("C", timed=True)  # no input boundary: a test that feeds itself measures the next row
    second = new_row()
    dump._new_input()  # the next input
    launch("A")
    third = new_row()  # made first, measured afterwards: the usual order
    launch("C", timed=True)
    assert _rows_of(tmp_path, monkeypatch, [first, second, third]) == [
        ["B", "timed"], ["C", "timed"], ["C", "timed"]]


def test_a_row_timed_without_do_bench_gets_every_launch_of_its_input(tmp_path, monkeypatch):
    new_row, launch = _tracking(monkeypatch)
    dump._new_input()
    launch("A")  # before the row: building its input
    row = new_row()
    launch("B")  # while it is current: the measurement
    dump._between_rows()  # the table is printed
    launch("C")  # what comes next is not this row's
    dump._new_input()
    launch("A")  # a timer of its own, and the measurement before the row
    early = new_row()
    assert _rows_of(tmp_path, monkeypatch, [row, early]) == [["B", "row"], ["A", "row"]]


def test_a_row_that_launched_nothing_says_so(tmp_path, monkeypatch):
    # small inputs may take a path without a Triton kernel; "-" tells that from a run
    # that recorded no binaries at all
    new_row, _ = _tracking(monkeypatch)
    dump._new_input()
    row = new_row()
    monkeypatch.setattr(dump, "_tables", [("op", "torch.float16", [(row._zl_row, "s")])])
    dump._write_rows(tmp_path / "rows.tsv", str(tmp_path))
    assert (tmp_path / "rows.tsv").read_text().splitlines()[1].split("\t")[4:] == ["", "row", "-"]
