import ast
import textwrap

import yaml

import gen_inventory as gen

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
    assert gen.check_shapes([y], set(["mm"]), [("main", repo)]) == 1
    assert "renamed_away" in capsys.readouterr().out


def test_check_shapes_clean_set(tmp_path, capsys):
    repo = _repo(tmp_path / "r", op_names=["mm"])
    y = tmp_path / "s.yaml"
    y.write_text("mm:\n  shapes:\n  - [1]\n")
    assert gen.check_shapes([y], set(["mm"]), [("main", repo)]) == 0
    assert "0 stale" in capsys.readouterr().out


def test_check_shapes_missing_file_does_not_crash(tmp_path, capsys):
    repo = _repo(tmp_path / "r", op_names=["mm"])
    assert gen.check_shapes([tmp_path / "nope.yaml"], set(["mm"]), [("main", repo)]) == 0
    assert "no " in capsys.readouterr().out


def test_check_shapes_survives_a_malformed_file(tmp_path, capsys):
    # these sets are hand-written, so a syntax slip is ordinary; it must not raise
    repo = _repo(tmp_path / "r", op_names=["mm"])
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [a, b\n")
    assert gen.check_shapes([bad], set(["mm"]), [("main", repo)]) == 0
    assert "not valid YAML" in capsys.readouterr().out


def test_check_shapes_survives_a_file_that_is_not_a_mapping(tmp_path, capsys):
    repo = _repo(tmp_path / "r", op_names=["mm"])
    lst = tmp_path / "list.yaml"
    lst.write_text("- just\n- a list\n")
    assert gen.check_shapes([lst], set(["mm"]), [("main", repo)]) == 0
    assert "not a mapping" in capsys.readouterr().out
