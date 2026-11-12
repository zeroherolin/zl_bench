import ast
import textwrap

import pytest
import yaml

import gen_inventory as gen
import shape_set

BENCH_SRC = textwrap.dedent('''
    import pytest

    class FooBenchmark(Benchmark):
        def __init__(self):
            super().__init__("foo_op", torch_op=None)

    @pytest.mark.skip(reason="broken")
    def test_skipped():
        GenericBenchmark(op_name="skipped_op").run()

    @pytest.mark.skipif(True, reason="conditional")
    def test_scaled():
        ScaledMMBenchmark("scaled_mm").run()

    def helper():
        pass
''')


def test_op_names_and_skip_marks():
    tree = ast.parse(BENCH_SRC)
    assert gen._op_names_in(tree) == {"foo_op", "skipped_op", "scaled_mm"}
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert any(gen._is_unconditional_skip(d) for d in funcs["test_skipped"].decorator_list)
    assert not any(gen._is_unconditional_skip(d) for d in funcs["test_scaled"].decorator_list)


def test_scan_repo(tmp_path):
    bench = tmp_path / "benchmark"
    (bench / "test_FLA").mkdir(parents=True)
    (bench / "test_foo.py").write_text(BENCH_SRC)
    (bench / "test_FLA" / "test_bar.py").write_text("def test_bar_perf():\n    pass\n")
    (bench / "test_benchmark_case_api.py").write_text("def test_harness():\n    pass\n")
    (bench / "models_benchmark").mkdir()
    (bench / "models_benchmark" / "test_model.py").write_text("def test_model():\n    pass\n")
    rows, op_names = gen.scan_repo("main", tmp_path)
    assert [(r["op"], r["op_file"], r["marked_skip"]) for r in rows] == [
        ("bar_perf", "FLA/test_bar", ""), ("skipped", "foo", "yes"), ("scaled", "foo", "")]
    assert op_names == {"foo_op", "skipped_op", "scaled_mm"}


def test_refresh_shapes(tmp_path):
    repo = tmp_path / "FlagGems"
    (repo / "benchmark").mkdir(parents=True)
    (repo / "benchmark" / "core_shapes.yaml").write_text(yaml.safe_dump({
        "BlasBenchmark": {"shapes": [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]},
        "curated": {"shapes": [[1], [2], [3]]},
    }))
    shapes = tmp_path / "set.yaml"
    shapes.write_text(yaml.safe_dump({
        "curated": {"shapes": [[7, 7]]},
        "renamed_op": {"shapes": [[1]]},
    }))
    gen.refresh_shapes(shapes, {"curated"}, [("main", repo)], one_shape=True)
    out = yaml.safe_load(shapes.read_text())
    assert out["curated"] == {"shapes": [[7, 7]]}            # curated entry wins
    assert out["BlasBenchmark"]["shapes"] == [[3, 4, 5, 6]]  # base key, one shape kept
    assert "renamed_op" not in out
    unmatched = yaml.safe_load((tmp_path / "set.unmatched.yaml").read_text())
    assert unmatched == {"renamed_op": {"shapes": [[1]]}}


def _repo(tmp_path, op_names=(), class_names=(), core_keys=()):
    """A minimal repo whose benchmark/ declares the given op_names, classes and core shapes."""
    bench = tmp_path / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    body = ["import x"]
    body += [f'B(op_name="{n}")' for n in op_names]
    body += [f"class {n}(Base):\n    pass" for n in class_names]
    (bench / "test_sample.py").write_text("\n".join(body) + "\n")
    if core_keys:
        (bench / "core_shapes.yaml").write_text(
            "\n".join(f"{k}:\n  shapes:\n  - [1]" for k in core_keys) + "\n")
    return tmp_path


def test_check_shapes_flags_only_keys_upstream_lost(tmp_path, capsys):
    repo = _repo(tmp_path / "r", op_names=["mm"], class_names=["BlasBenchmark"], core_keys=["from_core"])
    y = tmp_path / "s.yaml"
    y.write_text("mm:\n  shapes:\n  - [1]\nBlasBenchmark:\n  shapes:\n  - [2]\n"
              "from_core:\n  shapes:\n  - [3]\nrenamed_away:\n  shapes:\n  - [4]\n")
    assert gen.check_shapes([y], {"mm"}, [("main", repo)]) == 1
    assert "renamed_away" in capsys.readouterr().out


def test_check_shapes_clean_set(tmp_path, capsys):
    repo = _repo(tmp_path / "r", op_names=["mm"])
    y = tmp_path / "s.yaml"
    y.write_text("mm:\n  shapes:\n  - [1]\n")
    assert gen.check_shapes([y], {"mm"}, [("main", repo)]) == 0
    assert "0 stale" in capsys.readouterr().out


def test_check_shapes_counts_a_file_it_cannot_read(tmp_path, capsys):
    # These sets are written by hand, so a wrong path or a syntax slip is ordinary: no
    # traceback, but a set that cannot be read is not a set that passed the check.
    repo = _repo(tmp_path / "r", op_names=["mm"])
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [a, b\n")
    lst = tmp_path / "list.yaml"
    lst.write_text("- just\n- a list\n")
    good = tmp_path / "good.yaml"
    good.write_text("mm:\n  shapes:\n  - [1]\n")
    paths = [tmp_path / "nope.yaml", bad, lst, good]
    assert gen.check_shapes(paths, {"mm"}, [("main", repo)]) == 3
    out = capsys.readouterr().out
    assert "no " in out and "not valid YAML" in out and "not a mapping" in out
    assert "good.yaml 1 keys, 0 stale" in out


HARDCODED = textwrap.dedent('''
    class MoeBenchmark(Benchmark):
        def set_shapes(self, shape_file_path=None):
            self.shapes = [(1, 8), (16, 8)]

    class PlainBenchmark(Benchmark):
        def set_shapes(self, shape_file_path=None):
            super().set_shapes(shape_file_path)

    def test_moe_fp8():
        MoeBenchmark(op_name="{op_name}", torch_op=None).run()

    def test_plain():
        PlainBenchmark(op_name="plain", torch_op=None).run()
''')


def _two_repos(tmp_path):
    """main names the benchmark moe_fp8, vllm names the same class's benchmark moe."""
    repos = []
    for label, op_name in (("main", "moe_fp8"), ("vllm", "moe")):
        bench = tmp_path / label / "benchmark"
        bench.mkdir(parents=True)
        (bench / "test_moe_fp8.py").write_text(HARDCODED.replace("{op_name}", op_name))
        repos.append((label, tmp_path / label))
    return repos


def test_hardcoded_shape_users(tmp_path):
    users = gen.hardcoded_shape_users(_two_repos(tmp_path))
    assert users == {"MoeBenchmark": {"moe_fp8", "moe"}}  # PlainBenchmark reads the file


def test_check_shapes_flags_a_key_on_the_test_name(tmp_path, capsys):
    # The set has moe_fp8, which drives main. vllm's test of that name reads op_name moe,
    # for which there is nothing: it would run its hardcoded shapes and look fine.
    repos = _two_repos(tmp_path)
    tests = [r for label, root in repos for r in gen.scan_repo(label, root)[0]]
    y = tmp_path / "s.yaml"
    y.write_text("moe_fp8:\n  shapes:\n  - [1, 8]\n")
    names = {"moe_fp8", "moe", "plain"}
    assert gen.check_shapes([y], names, repos, tests) == 1
    out = capsys.readouterr().out
    assert "key 'moe_fp8' is the name of a test of vllm" in out and "op_name moe" in out
    assert "note: MoeBenchmark hardcodes its shapes and is also created as moe" in out
    # with an entry for each op_name the set is clean, and the note is gone
    y.write_text("moe_fp8:\n  shapes:\n  - [1, 8]\nmoe:\n  shapes:\n  - [1, 8]\n")
    assert gen.check_shapes([y], names, repos, tests) == 0
    assert "note:" not in capsys.readouterr().out


def test_missing_checkout_is_an_error_not_an_empty_repo(tmp_path):
    # scanning a path without benchmark/ would write an empty inventory over the real one
    with pytest.raises(SystemExit, match="no benchmark/"):
        gen._resolve_repos([f"main={tmp_path}"])
    (tmp_path / "benchmark").mkdir()
    assert gen._resolve_repos([f"main={tmp_path}"]) == [("main", tmp_path)]


def test_dump_shapes_writes_no_anchors():
    # two keys sharing one list: PyYAML would write &id001 / *id001, numbered in dump
    # order, so adding a key renumbers every alias after it
    shared = [[1, 2], [3, 4]]
    text = gen.dump_shapes({"b": {"shapes": shared}, "a": {"shapes": shared}})
    assert "&" not in text and "*" not in text
    assert text == "a:\n  shapes:\n  - [1, 2]\n  - [3, 4]\nb:\n  shapes:\n  - [1, 2]\n  - [3, 4]\n"


def _core_repo(tmp_path, label, core, op_names=(), class_names=()):
    """A repo with the given core_shapes.yaml, op_names and Benchmark classes."""
    repo = _repo(tmp_path / label, op_names=op_names, class_names=class_names)
    (repo / "benchmark" / "core_shapes.yaml").write_text(yaml.safe_dump(core))
    return label, repo


def test_refresh_scopes_a_key_the_other_repo_means_differently(tmp_path):
    # vllm's core_shapes has `recurrent` as a case name; main has a benchmark of that
    # op_name that takes a sequence length and no core_shapes key. Handing main the
    # entry breaks it, withholding it breaks vllm: the key is scoped to vllm.
    main = _core_repo(tmp_path, "main", {"shared": {"shapes": [[1]]}}, op_names=["recurrent", "shared"])
    vllm = _core_repo(tmp_path, "vllm", {"recurrent": {"shapes": [["decode1"], ["decode16"]]},
                                             "shared": {"shapes": [[1]]}, "only_vllm": {"shapes": [[9]]}},
                      op_names=["recurrent", "shared", "only_vllm"])
    names = {"main": {"recurrent", "shared"}, "vllm": {"recurrent", "shared", "only_vllm"}}
    path = tmp_path / "set.yaml"
    path.write_text("shared:\n  shapes:\n  - [7]\n")
    gen.refresh_shapes(path, {"recurrent", "shared", "only_vllm"}, [main, vllm], names_by_repo=names)
    out = yaml.safe_load(path.read_text())
    assert out["vllm:recurrent"] == {"shapes": [["decode1"], ["decode16"]]}
    assert "recurrent" not in out
    assert out["only_vllm"] == {"shapes": [[9]]}  # nobody else uses the name: a plain key
    assert out["shared"] == {"shapes": [[7]]}  # the set's own entry wins
    assert shape_set.resolve(out, "main").get("recurrent") is None
    assert shape_set.resolve(out, "vllm")["recurrent"]["shapes"][0] == ["decode1"]
    # a second refresh changes nothing
    before = path.read_text()
    gen.refresh_shapes(path, {"recurrent", "shared", "only_vllm"}, [main, vllm], names_by_repo=names)
    assert path.read_text() == before


def test_refresh_scopes_a_key_both_repos_define_differently(tmp_path):
    main = _core_repo(tmp_path, "main", {"bmm": {"shapes": [[1, 1024]]}}, op_names=["bmm"])
    vllm = _core_repo(tmp_path, "vllm", {"bmm": {"shapes": [[2, 64], [4, 129], [8, 8]]}}, op_names=["bmm"])
    names = {"main": {"bmm"}, "vllm": {"bmm"}}
    path = tmp_path / "set.yaml"
    path.write_text("")
    gen.refresh_shapes(path, {"bmm"}, [main, vllm], one_shape=True, names_by_repo=names)
    out = yaml.safe_load(path.read_text())
    assert out["bmm"] == {"shapes": [[1, 1024]]}  # the first repo keeps the plain key
    assert out["vllm:bmm"] == {"shapes": [[8, 8]]}  # the other its own, one shape kept


def test_refresh_keeps_the_form_of_a_key_the_set_already_has(tmp_path):
    # `Chunk` is a class in both repos and a core_shapes key of main only. The set
    # carries it as a plain key, which vllm reads too: a refresh must not scope it.
    main = _core_repo(tmp_path, "main", {"Chunk": {"shapes": [[1, 64], [2, 64]]}}, class_names=["Chunk"])
    vllm = _core_repo(tmp_path, "vllm", {}, class_names=["Chunk"])
    names = {"main": {"Chunk"}, "vllm": {"Chunk"}}
    path = tmp_path / "set.yaml"
    path.write_text("Chunk:\n  shapes:\n  - [1, 64]\n  - [2, 64]\n")
    gen.refresh_shapes(path, set(), [main, vllm], names_by_repo=names)
    out = yaml.safe_load(path.read_text())
    assert list(out) == ["Chunk"] and len(out["Chunk"]["shapes"]) == 2


def test_check_shapes_reads_scoped_keys(tmp_path, capsys):
    repo = _repo(tmp_path / "r", op_names=["mm"])
    y = tmp_path / "s.yaml"
    one = ":\n  shapes:\n  - [1]\n"
    y.write_text(f"vllm:mm{one}vllm:gone{one}elsewhere:mm{one}")
    assert gen.check_shapes([y], {"mm"}, [("main", repo)]) == 2
    out = capsys.readouterr().out
    assert "2 stale: elsewhere:mm, vllm:gone" in out
