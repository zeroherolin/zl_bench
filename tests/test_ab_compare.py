import csv
import json

import ab_compare
import ab_stats

HEADER = "Operator: softmax  (dtype=torch.bfloat16, mode=kernel, level=core)\n"


def _log(lats, failed=()):
    lines = [HEADER]
    for i, lat in enumerate(lats):
        if i in failed:
            lines.append("FAILED   N/A   N/A   N/A  [torch.Size([8])]\n")
        else:
            lines.append(f"SUCCESS  0.010000  {lat:.6f}  1.000  [torch.Size([{i + 1}])]\n")
    return "".join(lines)


def test_load_rows_keeps_positions_across_failed_rows(tmp_path):
    (tmp_path / "run.log").write_text(_log([0.1, 0.2, 0.3], failed={1}))
    rows = ab_compare.load_rows(tmp_path)
    assert sorted(rows) == [("softmax", "torch.bfloat16", 0), ("softmax", "torch.bfloat16", 2)]
    assert rows[("softmax", "torch.bfloat16", 2)][0] == 0.3


def test_nocg_tables():
    text = ("BENCHMARK_DIRECT_NO_CUDAGRAPH op=softmax phase=host_sync\n" + _log([0.1])
            + HEADER.replace("bfloat16", "float16"))
    assert ab_compare.nocg_tables(text) == {("softmax", "torch.bfloat16")}


def test_parse_range():
    assert ab_compare.parse_range(None) is None
    assert ab_compare.parse_range("7-12") == (7, 12)
    assert ab_compare.parse_range("3") == (3, 3)
    assert ab_compare.parse_range("-6") == (1, 6)
    assert ab_compare.parse_range("7-")[0] == 7


def _ab_dir(tmp_path, rounds, a_lat, b_lat, skip_b=()):
    """An ab_run.py output dir with one op of two rows; row 1 is 10% faster on B."""
    out = tmp_path / "ab"
    for k in range(1, rounds + 1):
        for side, lat in (("a", a_lat), ("b", b_lat)):
            if side == "b" and k in skip_b:
                continue
            run = out / f"r{k}_{side}" / "main" / "softmax"
            run.mkdir(parents=True)
            jitter = 1 + 0.001 * k
            (run / "run.log").write_text(_log([0.1 * jitter, lat * jitter]))
            (out / f"r{k}_{side}" / "ops_status.csv").write_text(
                "repo,op,op_file,run_name,status\nmain,softmax,softmax,softmax,PASS\n")
    (out / "ab.json").write_text(json.dumps({"record": "rec", "sides": {}}))
    return out


def test_analyze_pairs_rounds_and_applies_the_gate(tmp_path):
    out = _ab_dir(tmp_path, 6, 0.2, 0.2 / 1.1, skip_b={3})
    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / "rows.tsv").write_text(
        "run\top\top_name\tdtype\trow\tshape\tchanged\tchanged_kernels\n"
        "main/softmax\tmain:softmax@softmax\tsoftmax\ttorch.bfloat16\t0\t1\t0\t\n"
        "main/softmax\tmain:softmax@softmax\tsoftmax\ttorch.bfloat16\t1\t2\t1\tH1\n")
    res = {r.row.row: r for r in ab_compare.analyze(out, gate)}
    assert res[0].row.rounds == [1, 2, 4, 5, 6]  # round 3 has no B side
    assert res[0].gate == "unchanged" and res[1].gate == "changed"
    assert res[1].stats.verdict == "undecided"  # 5 rounds < MIN_ROUNDS
    res = {r.row.row: r for r in ab_compare.analyze(out, gate, min_rounds=5)}
    assert res[0].stats.verdict == "same"
    assert res[1].stats.verdict == "faster" and abs(res[1].stats.gap - 0.1) < 1e-4
    only_late = ab_compare.analyze(out, gate, (4, 6))
    assert only_late[0].row.rounds == [4, 5, 6]
    assert ab_compare.op_labels(out) == {"main/softmax": "main:softmax@softmax"}


def _result(verdict="faster", p=0.002, p_paired=0.031):
    """One analysed row, enough for write_csv."""
    row = ab_compare.Row(run="vllm/mm", op_name="mm", dtype="torch.bfloat16", row=0,
                    size="[torch.Size([8])]", a=[1.0] * 6, b=[0.9] * 6,
                    rounds=list(range(1, 7)), nocg=False, fallback=False)
    stats = ab_stats.RowStats(6, 6, 1.0, 0.9, 0.111, p, verdict, p_paired)
    return ab_compare.Result(row, stats, "changed")


def test_write_csv_header_matches_every_row(tmp_path):
    out = tmp_path / "rows.csv"
    ab_compare.write_csv(out, [_result(), _result("same", 0.4, 1.0)], {})
    rows = list(csv.reader(out.open()))
    assert len(rows) == 3
    assert len(set(len(r) for r in rows)) == 1, "a row does not match the header"


def test_write_csv_carries_the_paired_p(tmp_path):
    # the README points readers at this column to see why a row was decided
    out = tmp_path / "rows.csv"
    ab_compare.write_csv(out, [_result(p_paired=0.031)], {})
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["p_paired"] == "0.031"


def test_write_csv_leaves_a_missing_paired_p_empty(tmp_path):
    out = tmp_path / "rows.csv"
    ab_compare.write_csv(out, [_result(p_paired=float("nan"))], {})
    assert list(csv.DictReader(out.open()))[0]["p_paired"] == ""
