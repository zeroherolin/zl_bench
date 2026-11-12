import argparse
import threading

import pytest

import ab_run
import run_batch

CASES = [run_batch.Case("main", "softmax", "softmax", "softmax"),
         run_batch.Case("vllm", "mm", "mm", "mm")]


def _args(tmp_path, **kw):
    base = {"out": tmp_path, "rounds": 6, "add_rounds": 0, "from_round": 0}
    base.update(kw)
    return argparse.Namespace(**base)


def _finished(out, rounds, sides="ab", run="main/softmax", status="PASS"):
    """ops_status.csv of the given rounds and sides for one op."""
    repo, name = run.split("/")
    for k in rounds:
        for side in sides:
            d = out / f"r{k}_{side}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "ops_status.csv").write_text(
                "repo,op,op_file,run_name,gpu,status,elapsed_ms\n"
                f"{repo},{name},{name},{name},0,{status},1000\n")


def test_plan_runs_the_rounds_that_are_not_done_on_both_sides(tmp_path):
    _finished(tmp_path, [1, 2])
    _finished(tmp_path, [3], sides="a")  # B never ran: the round is not done
    _finished(tmp_path, [4], status="FAIL")
    done = ab_run.done_rounds(tmp_path)
    assert sorted(done["main/softmax"]) == [1, 2]
    items = ab_run.plan(CASES, _args(tmp_path, rounds=4), done)
    assert [(c.run_name, k) for c, k in items] == [
        ("softmax", 3), ("softmax", 4), ("mm", 1), ("mm", 2), ("mm", 3), ("mm", 4)]


def test_add_rounds_starts_after_the_highest_round(tmp_path):
    _finished(tmp_path, range(1, 7))
    args = _args(tmp_path, add_rounds=3)
    assert ab_run.fresh_range(args) == (7, 9)
    items = ab_run.plan(CASES[:1], args, ab_run.done_rounds(tmp_path))
    assert [k for _, k in items] == [7, 8, 9]


def test_from_round_finishes_an_interrupted_range(tmp_path):
    # rounds 7-9 were asked for; 7 finished, 8 was cut short. A rerun without
    # --from-round would number three more rounds after 8 and leave 8 half done.
    _finished(tmp_path, range(1, 8))
    _finished(tmp_path, [8], sides="a")
    assert ab_run.fresh_range(_args(tmp_path, add_rounds=3)) == (9, 11)
    args = _args(tmp_path, add_rounds=3, from_round=7)
    assert ab_run.fresh_range(args) == (7, 9)
    items = ab_run.plan(CASES[:1], args, ab_run.done_rounds(tmp_path))
    assert [k for _, k in items] == [8, 9]  # round 7 is kept, 8 is rerun on both sides
    # everything done: nothing to run, and the range is still the one to confirm
    _finished(tmp_path, [8, 9])
    assert ab_run.plan(CASES[:1], args, ab_run.done_rounds(tmp_path)) == []
    assert ab_run.fresh_range(args) == (7, 9)


def test_a_gate_dir_without_rows_is_rejected_before_any_round(tmp_path, monkeypatch):
    record, gate = tmp_path / "record", tmp_path / "gate"
    record.mkdir()
    gate.mkdir()
    argv = ["ab_run.py", "--record", str(record), "--ops", "softmax", "--gate", str(gate),
            "--out", str(tmp_path / "ab")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match=r"--gate takes an ir_gate\.py output dir"):
        ab_run.parse_args()
    (gate / "rows.tsv").write_text("run\top\top_name\tdtype\trow\tshape\tchanged\tchanged_kernels\n")
    assert ab_run.parse_args().gate == gate


def _runner(tmp_path, monkeypatch, outcome):
    """A Runner whose runs are stand-ins: outcome(case, side) gives the status of a run,
    or raises run_batch.Interrupted."""
    args = _args(tmp_path, dtypes="", op_timeout=0, shape_file=None, record=tmp_path, use_flagtune=0,
                 busy_gpu="ignore", sides={"a": {"env": {}, "pythonpath": ""},
                                               "b": {"env": {}, "pythonpath": ""}})

    def run_case(case, batch_dir, gpu, side_args, previous=None):
        status = outcome(case, batch_dir.name[-1])
        return {"repo": case.repo, "op": case.op, "op_file": case.op_file, "run_name": case.run_name,
                    "gpu": gpu, "status": status, "return_code": "0", "elapsed_ms": "1000", "rows_total": "1",
                    "rows_success": "1", "notes": "", "out_dir": ""}

    monkeypatch.setattr(run_batch, "run_case", run_case)
    monkeypatch.setattr(run_batch, "_stop", threading.Event())
    return ab_run.Runner(args)


def test_a_round_cut_short_on_rerun_is_not_done(tmp_path, monkeypatch):
    # Round 1 ended with A failed and B passed. The rerun does both sides again; it is
    # interrupted after A. B's row of the first invocation must not make the round done:
    # that run was made at another time, and its run dir now holds a killed run.
    _finished(tmp_path, [1], sides="a", status="FAIL")
    _finished(tmp_path, [1], sides="b")
    assert ab_run.done_rounds(tmp_path) == {"main/softmax": {}}

    def outcome(case, side):
        if side == "b":
            raise run_batch.Interrupted
        return "PASS"

    runner = _runner(tmp_path, monkeypatch, outcome)
    runner.run([(CASES[0], 1)], ["0"], {})
    assert ab_run.done_rounds(tmp_path).get("main/softmax", {}) == {}
    assert ab_run.plan(CASES[:1], _args(tmp_path, rounds=1), ab_run.done_rounds(tmp_path)) == [(CASES[0], 1)]
    # the other op's row in the same round dir is left alone
    _finished(tmp_path, [2], run="vllm/mm")
    runner = _runner(tmp_path, monkeypatch, lambda case, side: "PASS")
    runner.run([(CASES[0], 2)], ["0"], {})
    assert set(ab_run.done_rounds(tmp_path)) == {"main/softmax", "vllm/mm"}
    assert ab_run.done_rounds(tmp_path)["main/softmax"] == {2: 2.0}


def test_an_op_that_never_finished_a_round_has_last_round_zero(tmp_path):
    # it passed on one side only, every round: done_rounds has an empty entry for it,
    # and --until-decided still has to number its next round
    _finished(tmp_path, [1, 2], sides="a")
    _finished(tmp_path, [1, 2], sides="b", status="FAIL")
    done = ab_run.done_rounds(tmp_path)
    assert done == {"main/softmax": {}}
    assert ab_run.last_round(done, "main/softmax") == 0 and ab_run.last_round(done, "vllm/mm") == 0
    _finished(tmp_path, [3])
    assert ab_run.last_round(ab_run.done_rounds(tmp_path), "main/softmax") == 3


def test_a_step_below_one_is_rejected(tmp_path, monkeypatch):
    record = tmp_path / "record"
    record.mkdir()
    monkeypatch.setattr("sys.argv", ["ab_run.py", "--record", str(record), "--ops", "softmax",
                                     "--until-decided", "--step", "0"])
    with pytest.raises(SystemExit, match="--step must be at least 1"):
        ab_run.parse_args()
