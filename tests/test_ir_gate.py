import ir_gate

TTGIR = """\
#loc = loc("k.py":1:0)
module {
  %0 = arith.constant 1 : i32 loc(#loc1)
  %1 = arith.addi %0, %0 : i32 loc(#loc2)
}
"""


def test_normalize_ignores_loc_ssa_names_and_order():
    renamed = TTGIR.replace("%0", "%a").replace("%1", "%b").replace("loc(#loc2)", "loc(#loc9)")
    lines = TTGIR.splitlines()
    reordered = "\n".join(lines[:2] + [lines[3], lines[2]] + lines[4:])
    assert ir_gate._normalize(TTGIR, exact=False) == ir_gate._normalize(renamed, exact=False)
    assert ir_gate._normalize(TTGIR, exact=False) == ir_gate._normalize(reordered, exact=False)
    assert ir_gate._normalize(TTGIR, exact=True) != ir_gate._normalize(renamed, exact=True)
    changed = TTGIR.replace("arith.addi", "arith.muli")
    assert ir_gate._normalize(TTGIR, exact=False) != ir_gate._normalize(changed, exact=False)


def _run_dir(root, kernels):
    """A run dir with ttir/<hash>/<kernel>.{ttir,json} and a matching index.tsv."""
    rows = ["shape\tkernel\tfile\tlaunches\tsweep_launches\tnum_warps\tnum_stages\t"
            "constexprs\tcache_hash"]
    for h, kernel, launches in kernels:
        d = root / "ttir" / h
        d.mkdir(parents=True)
        (d / f"{kernel}.ttir").write_text("module {}")
        (d / f"{kernel}.json").write_text("{}")
        rows.append(f"4x4\t{kernel}\tw4s3.ttgir\t{launches}\t0\t4\t3\t{{}}\t{h}")
    (root / "ttgir").mkdir()
    (root / "ttgir" / "index.tsv").write_text("\n".join(rows) + "\n")
    return root


def test_collect_dedups_by_cache_hash_and_labels_batch_ops(tmp_path):
    batch = tmp_path / "batch"
    (batch / "main").mkdir(parents=True)
    _run_dir(batch / "main" / "abs", [("H1", "abs_kernel", 2), ("H2", "fill_kernel", 1)])
    _run_dir(batch / "main" / "neg", [("H2", "fill_kernel", 1)])
    (batch / "ops_status.csv").write_text(
        "repo,op,op_file,run_name\nmain,abs,abs,abs\nmain,neg,neg,neg\n")
    items = ir_gate.collect([batch])
    assert sorted(v.ttir.parent.name for v in items) == ["H1", "H2"]
    assert {v.label for v in items} == {"main:abs@abs"}
    assert all(v.recorded and v.recorded.name == "w4s3.ttgir" for v in items)


def test_a_single_op_record_is_labelled_by_its_spec(tmp_path):
    # changed_ops.txt and all_ops.txt are --ops-file input: the op, not the dir name
    run = _run_dir(tmp_path / "softmax_20261110-120000", [("H1", "softmax_kernel", 1)])
    assert ir_gate.run_dirs([run])[0].label == "softmax_20261110-120000"  # a record without the line
    log = ">>> op=softmax  mode=record  USE_FLAGTUNE=0\n>>> spec=vllm:softmax@FLA/softmax\n>>> out=x\n"
    (run / "run.log").write_text(log)
    (rd,) = ir_gate.run_dirs([run])
    assert (rd.label, rd.run) == ("vllm:softmax@FLA/softmax", "")


def test_collect_single_run_dir(tmp_path):
    run = _run_dir(tmp_path / "softmax_run", [("H1", "softmax_kernel", 1)])
    (item,) = ir_gate.collect([run])
    assert item.label == "softmax_run"
    assert str(item.key) == "H1/softmax_kernel.ttgir"


def _digest(ttgir="t1", norm="n1", cubin="c1", shared=0):
    return {"ttgir": ttgir, "ttgir_norm": norm, "ops": {}, "llir": "l1", "ptx": "p1",
            "cubin": cubin, "meta": {"shared": shared, "num_warps": 4}}


def test_compare_finds_the_first_differing_stage():
    base = _digest()
    assert ir_gate.compare(base, _digest(), "cubin") == ("0", "-", "same")
    # same TTGIR, different binary: only the cubin gate sees it
    assert ir_gate.compare(base, _digest(cubin="c2"), "cubin") == ("1", "cubin", "same")
    assert ir_gate.compare(base, _digest(cubin="c2"), "ttgir") == ("0", "-", "same")
    # a reordered TTGIR that ends in the same binary measures the same: a control
    assert ir_gate.compare(base, _digest(ttgir="t2"), "cubin") == ("0", "(ttgir)", "order")
    assert ir_gate.compare(base, _digest(ttgir="t2"), "ttgir") == ("1", "ttgir", "order")
    other = _digest(ttgir="t2", norm="n2", cubin="c2")
    assert ir_gate.compare(base, other, "cubin") == ("1", "ttgir", "changed")
    assert ir_gate.compare(base, _digest(shared=1024), "cubin") == ("1", "meta", "same")
    assert ir_gate.compare(base, _digest(ttgir="t2", shared=1024), "cubin") == ("1", "ttgir", "order")
    assert ir_gate.compare(base, None, "cubin") == ("?", "error", "?")


def _row(kernels, row="0"):
    return {"op_name": "mm", "dtype": "torch.float16", "row": row, "shape": "4x4", "kernels": kernels}


def test_judge_rows():
    status = {"H0": ("0", "-", "same"), "H1": ("1", "cubin", "same"), "HX": ("?", "error", "?")}
    rows = [_row("H0"), _row("H0,H1", "1"), _row("H0,HX", "2"), _row("?", "3"), _row("H0,NEW", "4")]
    judged = ir_gate.judge_rows(rows, status, {"H0", "H1", "HX"})
    assert [(flag, which) for _, flag, which in judged] == [
        ("0", ""), ("1", "H1"), ("?", "HX"), ("?", ""), ("?", "NEW")]


def test_a_row_without_kernels_is_cleared_only_if_no_variant_is_unaccounted_for():
    status = {"H0": ("0", "-", "same"), "H1": ("1", "cubin", "same")}
    rows = [_row("H0"), _row("", "1")]
    # every variant the run used is in some row: the empty row launches no kernel
    assert [f for _, f, _ in ir_gate.judge_rows(rows, status, {"H0"})] == ["0", "0"]
    # the run used a variant no row claims (H1, changed): the empty row may have run it
    assert [f for _, f, _ in ir_gate.judge_rows(rows, status, {"H0", "H1"})] == ["0", "?"]


def test_a_cut_off_line_of_rows_tsv_does_not_break_the_gate(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "rows.tsv").write_text(
        "op_name\tdtype\trow\tshape\tkernels\nmm\ttorch.float16\t0\t4x4\tH0\nmm\ttorch.fl")
    rows = ir_gate.read_rows(run)
    assert len(rows) == 2
    judged = ir_gate.judge_rows(rows, {"H0": ("0", "-", "same")}, {"H0"})
    assert [(r["row"], flag) for r, flag, _ in judged] == [("0", "0")]


def test_run_dirs_and_rows(tmp_path):
    batch = tmp_path / "batch"
    run = _run_dir(batch / "main" / "abs", [("H1", "abs_kernel", 2)])
    (batch / "ops_status.csv").write_text("repo,op,op_file,run_name\nmain,abs,abs,abs\n")
    (run / "rows.tsv").write_text("op_name\tdtype\trow\tshape\tkernels\nabs\ttorch.float16\t0\t4x4\tH1\n")
    (rd,) = ir_gate.run_dirs([batch])
    assert (rd.label, rd.run) == ("main:abs@abs", "main/abs")
    assert ir_gate.run_hashes(rd.path) == {"H1"}
    assert ir_gate.read_rows(rd.path)[0]["kernels"] == "H1"
    assert ir_gate.read_rows(tmp_path) is None


def test_option_drift():
    # The record was made by a build without the pipeline_* options; the new build has
    # them. An operator library may set them where they exist, so the new side may
    # launch variants the record does not hold.
    old = {"options": ["num_stages", "num_warps"], "unrecorded_options": []}
    new = {"options": ["num_stages", "num_warps", "pipeline_merge_loads"],
               "unrecorded_options": ["pipeline_merge_loads"]}
    assert ir_gate.option_drift([old, old], [new, None]) == ["pipeline_merge_loads"]
    assert ir_gate.option_drift([old], [old]) == []
    # both builds have the option but the record was made without it
    assert ir_gate.option_drift([new], [new]) == ["pipeline_merge_loads"]
    # recorded with the newer build: nothing unrecorded, but the older side lacks it
    recorded_new = dict(new, unrecorded_options=[])
    assert ir_gate.option_drift([old], [recorded_new]) == ["pipeline_merge_loads"]
    # no --base (the recorded ttgir is the baseline) and failed compiles are no evidence
    assert ir_gate.option_drift([], [recorded_new]) == []
    assert ir_gate.option_drift([None], [None]) == []


def test_op_counts_and_diff():
    base = ir_gate.op_counts(TTGIR)
    assert base == {"arith.constant": 1, "arith.addi": 1}
    new = ir_gate.op_counts(TTGIR.replace("%1 = arith.addi %0, %0", '%1:2 = "tt.foo.bar"(%0)')
                            + "  ttg.local_load %0 : i32\n  ttg.local_load %1 : i32\n")
    assert new["tt.foo.bar"] == 1 and new["ttg.local_load"] == 2
    diff = ir_gate.op_diff({"ops": base}, {"ops": new})
    assert diff == "ttg.local_load:+2 arith.addi:-1 tt.foo.bar:+1"
    assert ir_gate.op_diff(None, {"ops": new}) == ""
