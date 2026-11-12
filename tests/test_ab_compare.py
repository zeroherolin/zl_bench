import csv
import json

import pytest

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
    assert len({len(r) for r in rows}) == 1, "a row does not match the header"


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


def _rounds(out, a, b, first=1, gpus=None, marker="", ran=None, status=("PASS", "PASS")):
    """Rounds first, first+1, ... of one op with one row; gpus[i] = (GPU of A, GPU of B),
    ran = (binaries of A, binaries of B) as the run's rows.tsv records them, status =
    (status of A, status of B) in ops_status.csv, None for a side without a row."""
    for i, (lat_a, lat_b) in enumerate(zip(a, b, strict=True)):
        for side, lat, gpu in (("a", lat_a, 0), ("b", lat_b, 1)):
            batch = out / f"r{first + i}_{side}"
            run = batch / "main" / "softmax"
            run.mkdir(parents=True)
            (run / "run.log").write_text(marker + _log([lat]))
            if ran:
                (run / "rows.tsv").write_text(
                    "op_name\tdtype\trow\tshape\tkernels\tscope\tbinaries\n"
                    f"softmax\ttorch.bfloat16\t0\t1\tH1,H2\ttimed\t{ran[gpu]}\n")
            used = gpus[i][gpu] if gpus else "3"
            if status[gpu] is not None:
                (batch / "ops_status.csv").write_text(
                    "repo,op,op_file,run_name,gpu,status\n"
                    f"main,softmax,softmax,softmax,{used},{status[gpu]}\n")
    (out / "ab.json").write_text(json.dumps({"record": "rec", "sides": {}}))
    return out


# B is 2% faster in every round, on GPUs a few percent apart
SPREAD = [1.000, 1.030, 0.985, 1.040, 1.012, 0.970]
FASTER = [x * 0.98 for x in SPREAD]
RULE = {"min_rounds": 6, "threshold": 0.005, "alpha": 0.01, "clear": 0.1, "alpha_paired": 0.05}


def test_a_round_split_over_two_gpus_is_not_a_pair(tmp_path):
    gpus = [("3", "3")] * 5 + [("3", "5")]  # the last round's B side was rerun elsewhere
    out = _rounds(tmp_path / "ab", SPREAD, FASTER, gpus=gpus)
    assert ab_compare.split_rounds(out) == {("main/softmax", 6)}
    (row,) = ab_compare.load_samples(out)
    assert row.rounds == [1, 2, 3, 4, 5]


def test_a_replay_fallback_run_has_no_controls(tmp_path):
    gate = {("main/softmax", "softmax", "torch.bfloat16", 0): "0"}
    out = _rounds(tmp_path / "ab", SPREAD, FASTER)
    (row,) = ab_compare.load_samples(out)
    assert ab_compare.gate_flag(gate, row) == "unchanged"
    # the gate compiled the recorded configs; this run tuned one of its own
    out = _rounds(tmp_path / "fb", SPREAD, FASTER, marker="AUTOTUNE_REPLAY_FALLBACK reason=key_missing\n")
    (row,) = ab_compare.load_samples(out)
    assert row.fallback and ab_compare.gate_flag(gate, row) == "unknown"


def test_recheck_accepts_the_pairing():
    # The verdict came from the pairing, so the fresh rounds overlap just as much: the
    # rank test alone could never confirm it.
    outcome, new = ab_compare.recheck("faster", SPREAD[:4], FASTER[:4], 0.005, 0.05)
    # four agreeing rounds are a one-sided p = 0.0625: not shown, but not contradicted
    assert outcome == "too_few_rounds" and new.p >= 0.05 and new.p_paired == 0.0625
    outcome, new = ab_compare.recheck("faster", SPREAD[:5], FASTER[:5], 0.005, 0.05)
    assert outcome == "confirmed" and new.p >= 0.05 and new.p_paired == 0.03125
    # the same rounds do not confirm the opposite direction
    assert ab_compare.recheck("slower", SPREAD[:5], FASTER[:5], 0.005, 0.05)[0] == "not_reproduced"
    # one fresh round disagreeing: neither test shows it
    b = FASTER[:5]
    b[2] = SPREAD[2] * 1.01
    assert ab_compare.recheck("faster", SPREAD[:5], b, 0.005, 0.05)[0] == "not_reproduced"


def test_recheck_with_too_few_rounds_says_so():
    # three rounds cannot reach p < 0.05 however clear they are (the best is exactly 0.05)
    outcome, new = ab_compare.recheck("faster", [1.10, 1.11, 1.12], [1.00, 1.01, 1.02], 0.005, 0.05)
    assert outcome == "too_few_rounds" and new.p == 0.05
    assert ab_compare.recheck("faster", [1.10, 1.11, 1.12, 1.13], [1.0, 1.01, 1.02, 1.03],
                              0.005, 0.05)[0] == "confirmed"


def test_confirm_keeps_drops_or_leaves_a_verdict(tmp_path):
    out = _rounds(tmp_path / "ab", SPREAD, FASTER)
    (judged,) = ab_compare.analyze(out, **RULE)
    assert (judged.stats.verdict, judged.stats.basis) == ("faster", "pairing")
    assert ab_compare.is_weak(judged, 0.0)  # resting on the pairing alone is weak evidence

    _rounds(out, SPREAD[:5], FASTER[:5], first=7)
    (r,) = ab_compare.confirm(out, None, (7, 11), 0.05, **RULE)
    assert r.confirm == "confirmed" and r.stats.verdict == "faster"
    assert r.stats.n_a == 11 and r.new.n_a == 5  # the gap is reported over all rounds

    out = _rounds(tmp_path / "ab2", SPREAD, FASTER)
    _rounds(out, SPREAD[:5], SPREAD[:5], first=7)  # no difference in the fresh rounds
    (r,) = ab_compare.confirm(out, None, (7, 11), 0.05, **RULE)
    assert r.confirm == "not_reproduced" and r.stats.verdict == "undecided"
    assert r.stats.n_a == 6 and r.stats.basis == ""

    out = _rounds(tmp_path / "ab3", SPREAD, FASTER)
    _rounds(out, SPREAD[:3], FASTER[:3], first=7)
    (r,) = ab_compare.confirm(out, None, (7, 9), 0.05, **RULE)
    assert r.confirm == "too_few_rounds" and r.stats.verdict == "faster"


def test_write_csv_names_the_basis_and_the_confirm_columns(tmp_path):
    out = _rounds(tmp_path / "ab", SPREAD, FASTER)
    _rounds(out, SPREAD[:5], FASTER[:5], first=7)
    results = ab_compare.confirm(out, None, (7, 11), 0.05, **RULE)
    ab_compare.write_csv(tmp_path / "rows.csv", results, {})
    (row,) = csv.DictReader((tmp_path / "rows.csv").open())
    assert list(row) == ab_compare.CSV_FIELDS
    assert (row["verdict"], row["basis"], row["confirm"]) == ("faster", "pairing", "confirmed")
    assert row["confirm_rounds"] == "5" and row["confirm_p_paired"] == "0.03125"


def test_parse_range_rejects_garbage():
    with pytest.raises(SystemExit, match="round range"):
        ab_compare.parse_range("7..12")


CLEARED = {("main/softmax", "softmax", "torch.bfloat16", 0): "0"}
FALLBACK_LINE = "AUTOTUNE_REPLAY_FALLBACK reason=key_missing\n"


def _gate_of(tmp_path, name, **kw):
    (row,) = ab_compare.load_samples(_rounds(tmp_path / name, SPREAD, FASTER, **kw))
    return row.binaries, ab_compare.gate_flag(CLEARED, row)


def test_a_cleared_row_whose_sides_ran_other_binaries_diverges(tmp_path):
    # The gate compiled the recorded variant on both builds and found one binary. The
    # B side launched something else (the operator library sets a compile option only
    # where the compiler has it): not a control, a result the gate could not see.
    assert _gate_of(tmp_path, "d", ran=("f1,f2", "f1,f9")) == ("differ", "diverged")
    assert _gate_of(tmp_path, "s", ran=("f2,f1", "f1,f2")) == ("same", "unchanged")  # order is free
    assert _gate_of(tmp_path, "n") == ("", "unchanged")  # runs that recorded no binaries
    # a run that could not tell what a row launched: not a control
    assert _gate_of(tmp_path, "q", ran=("f1,?", "f1,f2")) == ("unknown", "unknown")
    assert _gate_of(tmp_path, "k", ran=("-", "-")) == ("same", "unchanged")  # no kernel on either side
    assert _gate_of(tmp_path, "m", ran=("-", "f1")) == ("differ", "diverged")  # a kernel on one side only
    # a replay fallback: unknown without fingerprints, a control when they agree
    assert _gate_of(tmp_path, "fn", marker=FALLBACK_LINE) == ("", "unknown")
    assert _gate_of(tmp_path, "fs", marker=FALLBACK_LINE, ran=("f1", "f1")) == ("same", "unchanged")
    assert _gate_of(tmp_path, "fd", marker=FALLBACK_LINE, ran=("f1", "f2")) == ("differ", "diverged")


def test_one_round_with_other_binaries_is_enough_to_diverge(tmp_path):
    out = _rounds(tmp_path / "ab", SPREAD[:5], FASTER[:5], ran=("f1", "f1"))
    _rounds(out, SPREAD[5:], FASTER[5:], first=6, ran=("f1", "f2"))
    (row,) = ab_compare.load_samples(out)
    assert row.rounds == [1, 2, 3, 4, 5, 6] and row.binaries == "differ"


def test_diverged_rows_are_results_and_are_reported(tmp_path, capsys):
    out = _rounds(tmp_path / "ab", SPREAD, FASTER, ran=("f1", "f2"))
    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / "rows.tsv").write_text(
        "run\top\top_name\tdtype\trow\tshape\tchanged\tchanged_kernels\n"
        "main/softmax\tmain:softmax@softmax\tsoftmax\ttorch.bfloat16\t0\t1\t0\t\n")
    results = ab_compare.analyze(out, gate, **RULE)
    assert [r.gate for r in results] == ["diverged"]
    ab_compare.report(results, ab_compare.op_labels(out), {"record": "rec"}, False, True)
    text = capsys.readouterr().out
    assert "== rows with a changed kernel: 1; faster 1" in text  # counted as a result
    assert "1 of them DIVERGED" in text and "main:softmax@softmax" in text
    assert "controls (identical binaries): 0" in text
    ab_compare.write_csv(tmp_path / "rows.csv", results, {})
    (row,) = csv.DictReader((tmp_path / "rows.csv").open())
    assert (row["gate"], row["binaries"]) == ("diverged", "differ")


def _write_gate(tmp_path):
    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / "rows.tsv").write_text(
        "run\top\top_name\tdtype\trow\tshape\tchanged\tchanged_kernels\n"
        "main/softmax\tmain:softmax@softmax\tsoftmax\ttorch.bfloat16\t0\t1\t0\t\n")
    return gate


def test_confirm_tells_what_a_row_is_from_all_its_rounds(tmp_path):
    # The rounds judged first ran one binary on both sides; in the confirm rounds the
    # B side launched another. The row is no control, whichever rounds are reported.
    out = _rounds(tmp_path / "ab", SPREAD, FASTER, ran=("f1", "f1"))
    _rounds(out, SPREAD[:5], FASTER[:5], first=7, ran=("f1", "f2"))
    gate = _write_gate(tmp_path)
    (r,) = ab_compare.confirm(out, gate, (7, 11), 0.05, **RULE)
    assert (r.confirm, r.gate, r.row.binaries) == ("confirmed", "diverged", "differ")
    assert r.row.rounds == list(range(1, 12))
    # three fresh rounds are too few: the row keeps the rounds it was judged on, and
    # is still not a control
    out = _rounds(tmp_path / "ab2", SPREAD, FASTER, ran=("f1", "f1"))
    _rounds(out, SPREAD[:3], FASTER[:3], first=7, ran=("f1", "f2"))
    (r,) = ab_compare.confirm(out, gate, (7, 9), 0.05, **RULE)
    assert (r.confirm, r.gate) == ("too_few_rounds", "diverged")
    assert r.row.rounds == [1, 2, 3, 4, 5, 6] and r.stats.n_a == 6
    # a row with samples in the confirm rounds only has nothing to confirm
    assert ab_compare.confirm(out, gate, (1, 9), 0.05, **RULE) == []


def test_diverged_list_without_a_gate_fails_before_the_report(tmp_path, capsys, monkeypatch):
    out = _rounds(tmp_path / "ab", SPREAD, FASTER)
    monkeypatch.setattr("sys.argv", ["ab_compare.py", str(out), "--diverged-list", str(tmp_path / "d.txt")])
    with pytest.raises(SystemExit, match="--diverged-list needs --gate"):
        ab_compare.main()
    assert capsys.readouterr().out == "" and not (tmp_path / "d.txt").exists()


def test_a_round_cut_short_is_not_a_sample(tmp_path):
    # ab_run.py writes a side's status row when its run ends. The sixth round was cut
    # short on B: its run dir holds a log (of this attempt or an earlier one), and no row.
    out = _rounds(tmp_path / "ab", SPREAD[:5], FASTER[:5])
    _rounds(out, SPREAD[5:], FASTER[5:], first=6, status=("PASS", None))
    (row,) = ab_compare.load_samples(out)
    assert row.rounds == [1, 2, 3, 4, 5] and not row.failed


def test_a_row_measured_before_its_side_failed_counts_and_is_marked(tmp_path):
    # the op failed on B in one round, after this row was measured
    out = _rounds(tmp_path / "ab", SPREAD[:5], FASTER[:5])
    _rounds(out, SPREAD[5:], FASTER[5:], first=6, status=("PASS", "FAIL"))
    (r,) = ab_compare.analyze(out, **RULE)
    assert r.row.rounds == [1, 2, 3, 4, 5, 6] and r.row.failed
    assert r.stats.verdict == "faster" and ab_compare.is_weak(r, 0.0)
    ab_compare.write_csv(tmp_path / "rows.csv", [r], {})
    (line,) = csv.DictReader((tmp_path / "rows.csv").open())
    assert line["flags"] == "failed_side"


def test_recheck_does_not_confirm_on_fewer_rounds_than_can_show_it():
    # run.log prints six decimals, so microsecond kernels tie; with ties the rank test
    # is a normal approximation, which gives p = 0.036 where three rounds cannot show
    # anything below 0.05
    a, b = [0.001030, 0.001030, 0.001040], [0.001010, 0.001010, 0.001020]
    outcome, new = ab_compare.recheck("faster", a, b, 0.005, 0.05)
    assert outcome == "too_few_rounds" and new.p < 0.05
    assert ab_compare.recheck("faster", a + [0.001030], b + [0.001010], 0.005, 0.05)[0] == "confirmed"


def test_a_damaged_rows_tsv_does_not_break_the_comparison(tmp_path):
    out = _rounds(tmp_path / "ab", SPREAD, FASTER, ran=("f1", "f1"))
    damaged = out / "r6_b" / "main" / "softmax" / "rows.tsv"
    damaged.write_text(damaged.read_text() + "softmax\ttorch.bfl")  # cut off mid-line
    (row,) = ab_compare.load_samples(out)
    assert row.binaries == "same" and len(row.rounds) == 6
