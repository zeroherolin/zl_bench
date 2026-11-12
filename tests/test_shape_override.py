import _shape_override_plugin as plugin


class FakeBenchmark:
    def __init__(self, op_name, shapes):
        self.op_name, self._shapes = op_name, shapes

    def init_user_config(self):
        self.shapes = list(self._shapes)  # what a hardcoded set_shapes() leaves behind


def _patched(monkeypatch, entries):
    monkeypatch.setattr(plugin, "_entries", entries)

    class Bench(FakeBenchmark):
        pass

    plugin._patch_init_user_config(Bench)
    return Bench


def test_yaml_shapes_override_hardcoded(monkeypatch):
    Bench = _patched(monkeypatch, {"moe": {"shapes": [[1, 2, 3]], "shape_desc": "M, N, K"}})
    b = Bench("moe", [(9, 9, 9), (8, 8, 8)])
    b.init_user_config()
    assert b.shapes == [(1, 2, 3)]
    assert b.shape_desc == "M, N, K"


def test_ops_without_entry_keep_their_shapes(monkeypatch):
    Bench = _patched(monkeypatch, {"other": {"shapes": [[1]]}})
    b = Bench("moe", [(9, 9, 9)])
    b.init_user_config()
    assert b.shapes == [(9, 9, 9)]


def test_arity_normalization_is_preserved(monkeypatch):
    Bench = _patched(monkeypatch, {"mm": {"shapes": [[2, 64, 64, 64]]}})
    b = Bench("mm", [(64, 64, 64)])  # subclass normalized (B, M, N, K) -> (M, N, K)
    b.init_user_config()
    assert b.shapes == [(64, 64, 64)]


def test_standard_input_fn_signature():
    assert plugin._takes_shape_dtype_device(lambda shape, dtype, device: None)
    assert not plugin._takes_shape_dtype_device(lambda b, m, n, k, dtype, device, t: None)
    assert not plugin._takes_shape_dtype_device(None)


def _report(monkeypatch, capsys, entries, run, test_name=""):
    """The end-of-session report for a shape file. run(Base) configures the benchmarks
    of the session; Base stands for upstream's Benchmark, patched like it."""
    monkeypatch.setattr(plugin, "_entries", entries)
    monkeypatch.setattr(plugin, "_benches", [])
    monkeypatch.setattr(plugin, "_own_inputs", set())
    monkeypatch.setenv("FLAGGEMS_PERF_CURRENT_OP", test_name)

    class Benchmark(FakeBenchmark):
        pass

    plugin._patch_init_user_config(Benchmark)
    run(Benchmark)
    plugin.pytest_sessionfinish(None, 0)
    return capsys.readouterr().err


def _hardcoded(op_name, shapes):
    """Configure one benchmark whose set_shapes() hardcodes `shapes`."""
    return lambda Base: Base(op_name, shapes).init_user_config()


ONE = {"shapes": [[1, 2]]}


def test_renamed_key_is_reported(monkeypatch, capsys):
    # a curated set still carries the op's old name: no entry applies, own shapes ran
    err = _report(monkeypatch, capsys, {"grouped_topk_old": ONE}, _hardcoded("grouped_topk", [(9, 9)]))
    assert "SHAPE_KEYS_UNUSED grouped_topk has no entry" in err


def test_matching_key_is_silent_and_named(monkeypatch, capsys):
    err = _report(monkeypatch, capsys, {"grouped_topk": ONE}, _hardcoded("grouped_topk", [(9, 9)]))
    assert "SHAPE_KEYS_UNUSED" not in err
    assert "shapes of grouped_topk: 1 of the 1 of key 'grouped_topk'" in err


def test_class_key_fallback_is_not_a_problem(monkeypatch, capsys):
    # upstream's fallback chain: most ops of a whole-suite set have no entry of their own
    entries = {"Benchmark": {"shapes": [[4096, 4096]]}, "softmax": ONE}
    err = _report(monkeypatch, capsys, entries, _hardcoded("concat", [(4096, 4096)]))
    assert "SHAPE_KEYS_UNUSED" not in err
    assert "shapes of concat: 1 of the 1 of class key 'Benchmark'" in err
    # a benchmark that hardcodes its shapes does not read the class key either; the log
    # says so, without flagging every such op
    err = _report(monkeypatch, capsys, entries, _hardcoded("pool", [(2, 3, 8), (4, 16, 16)]))
    assert "SHAPE_KEYS_UNUSED" not in err
    assert "shapes of pool: its own 2 (class key 'Benchmark' not used)" in err


def test_key_on_the_test_name_instead_of_the_op_name_is_reported(monkeypatch, capsys):
    # FlagGems-vllm's test_fused_marlin_moe_w4a16_mxfp4 uses the op_name
    # fused_marlin_moe_w4a16: the entry under the test's name drives nothing
    entries = {"Benchmark": ONE, "fused_marlin_moe_w4a16_mxfp4": ONE}
    run = _hardcoded("fused_marlin_moe_w4a16", [(1, 8, 4096), (16, 8, 4096)])
    err = _report(monkeypatch, capsys, entries, run, test_name="fused_marlin_moe_w4a16_mxfp4")
    assert "SHAPE_KEYS_UNUSED fused_marlin_moe_w4a16 has no entry" in err
    assert "'fused_marlin_moe_w4a16_mxfp4' is the name of the test" in err
    # with its own entry the same benchmark is fine
    entries["fused_marlin_moe_w4a16"] = ONE
    assert "SHAPE_KEYS_UNUSED" not in _report(monkeypatch, capsys, entries, run,
                                              test_name="fused_marlin_moe_w4a16_mxfp4")


def _own_init(cls):
    """Wrap the class's own init_user_config the way pytest_collection_finish does."""
    own = cls.__dict__["init_user_config"]
    cls.init_user_config = plugin._configured(own, plugin._only_assigns_shapes(own))
    return cls


def test_an_init_user_config_that_hardcodes_shapes_gets_the_entry(monkeypatch, capsys):
    seen = []

    def run(Base):
        class Replacing(Base):  # adaptive_max_pool3d: reads the file, then sets its own list
            def init_user_config(self):
                super().init_user_config()
                self.shapes = [(7, 7), (8, 8)]

        class Alone(Base):  # beam_search_score: never calls Benchmark.init_user_config
            def init_user_config(self):
                self.shapes = [(3, 3)]

        _own_init(Replacing)
        _own_init(Alone)
        for cls, name in ((Replacing, "pool3d"), (Alone, "beam"), (Alone, "other")):
            bench = cls(name, [(9, 9)])
            bench.init_user_config()
            seen.append(bench.shapes)

    err = _report(monkeypatch, capsys, {"pool3d": ONE, "beam": ONE}, run)
    assert seen == [[(1, 2)], [(1, 2)], [(3, 3)]]  # an op without an entry keeps its own
    assert "SHAPE_KEYS_UNUSED other has no entry" in err and err.count("has no entry") == 1
    assert "shapes of pool3d: 1 of the 1 of key 'pool3d'" in err


def test_an_entry_the_benchmark_rejects_is_reported(monkeypatch, capsys):
    def run(Base):
        class Validating(Base):  # the chunk_* family: falls back to its defaults
            def init_user_config(self):
                super().init_user_config()
                if any(s[0] > 4 for s in self.shapes):
                    self.shapes = [(1, 1), (2, 2)]

        _own_init(Validating)("gdn", [(9, 9)]).init_user_config()

    err = _report(monkeypatch, capsys, {"gdn": {"shapes": [[8, 8]]}}, run)
    assert "SHAPE_KEYS_UNUSED gdn has an entry, but the benchmark replaced it" in err
    assert "shapes of gdn: its own 2 (key 'gdn' not used)" in err


def test_an_entry_the_benchmark_filters_is_fine(monkeypatch, capsys):
    def run(Base):
        class Filtering(Base):  # cat, isin: drop the shapes that would not fit in memory
            def init_user_config(self):
                super().init_user_config()
                self.shapes = [s for s in self.shapes if s[0] < 100]

        _own_init(Filtering)("cat", [(9, 9)]).init_user_config()

    err = _report(monkeypatch, capsys, {"cat": {"shapes": [[1, 2], [512, 2]]}}, run)
    assert "SHAPE_KEYS_UNUSED" not in err and "shapes of cat: 1 of the 2 of key 'cat'" in err


def test_only_assigns_shapes():
    class B:
        def hardcodes(self):
            self.mode = 1
            self.shapes = [(1,)]

        def filters(self):
            self.shapes = [s for s in self.shapes if s]

        def validates(self):
            if self.shapes == []:
                self.shapes = [(1,)]

        def iterates(self):
            for s in self.shapes:
                print(s)

        def elsewhere(self):
            self.mode = 1

    assert plugin._only_assigns_shapes(B.hardcodes)
    for name in ("filters", "validates", "iterates", "elsewhere"):
        assert not plugin._only_assigns_shapes(getattr(B, name)), name
    assert not plugin._only_assigns_shapes(len)  # no source: left alone


def test_inputs_built_without_reading_shapes_are_reported(monkeypatch, capsys):
    def run(Base):
        class Own(Base):
            device, input_fn = "cpu", None  # no standard input_fn to redirect through

            def get_input_iter(self, dtype):
                yield "an input from a hardcoded list"

        Own.get_input_iter = plugin._yaml_input_iter(Own.__dict__["get_input_iter"])
        bench = Own("conv_pad", [(9, 9)])
        bench.init_user_config()
        assert list(bench.get_input_iter(None)) == ["an input from a hardcoded list"]

    err = _report(monkeypatch, capsys, {"conv_pad": ONE}, run)
    assert "SHAPE_KEYS_UNUSED conv_pad has an entry, but its get_input_iter builds inputs" in err


def test_arity_normalization_is_not_reported(monkeypatch, capsys):
    entries = {"mm": {"shapes": [[2, 64, 64, 64]]}}
    err = _report(monkeypatch, capsys, entries, _hardcoded("mm", [(64, 64, 64)]))
    assert "SHAPE_KEYS_UNUSED" not in err and "derived from key 'mm' (other arity)" in err


def test_shapes_that_are_not_plain_tuples_do_not_break_the_report(monkeypatch, capsys):
    class Param:  # an unhashable stand-in for the TestParam dataclasses some benchmarks keep
        __hash__ = None

        def __repr__(self):
            return "Param(1)"

    err = _report(monkeypatch, capsys, {"Benchmark": ONE}, _hardcoded("sparse", [Param(), [[1, 2], [3]]]))
    assert "shapes of sparse: its own 2 (class key 'Benchmark' not used)" in err


def test_scalar_shapes_are_compared_the_way_upstream_reads_them(monkeypatch, capsys):
    # `shapes: [1024, 4096]` reaches the benchmark as (1024,), (4096,)
    err = _report(monkeypatch, capsys, {"vec": {"shapes": [1024, 4096]}}, _hardcoded("vec", [(9,)]))
    assert "SHAPE_KEYS_UNUSED" not in err and "shapes of vec: 2 of the 2 of key 'vec'" in err


def test_no_shape_file_is_silent(monkeypatch, capsys):
    assert _report(monkeypatch, capsys, {}, _hardcoded("mm", [(9, 9)])) == ""
