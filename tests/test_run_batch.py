import argparse
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

import run_batch


def _row(repo, op, op_file, collected="yes", marked_skip=""):
    return {"repo": repo, "op": op, "op_file": op_file,
            "collected": collected, "marked_skip": marked_skip}


INVENTORY = {
    "main": [
        _row("main", "abs", "abs"),
        _row("main", "perf_x", "a"),
        _row("main", "perf_x", "FLA/b"),
        _row("main", "skipped", "skipped", marked_skip="yes"),
        _row("main", "npu_only", "npu_only", collected="no"),
        _row("main", "dismissed", "dismissed"),
    ],
    "vllm": [_row("vllm", "abs", "abs")],
}


def _args(tmp_path, **kw):
    dismiss = tmp_path / "dismiss.txt"
    dismiss.write_text("# comment\ndismissed\nvllm:nothing\n")
    base = {"all": False, "ops": None, "ops_file": None, "repos": None, "dismiss": dismiss,
                "include_marked_skip": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_load_dismiss(tmp_path):
    assert run_batch.load_dismiss(_args(tmp_path).dismiss) == {
        ("main", "dismissed"), ("vllm", "nothing")}


def test_select_all_filters_and_names_duplicates(tmp_path):
    cases = run_batch.resolve_selection(_args(tmp_path, all=True), INVENTORY)
    assert [c.key for c in cases] == [
        "main:abs@abs", "main:perf_x@a", "main:perf_x@FLA/b", "vllm:abs@abs"]
    assert [c.run_name for c in cases] == ["abs", "perf_x@a", "perf_x@FLA_b", "abs"]


def test_select_specs(tmp_path):
    cases = run_batch.resolve_selection(
        _args(tmp_path, ops=["abs", "vllm:abs", "perf_x@FLA/b"]), INVENTORY)
    assert [c.key for c in cases] == ["main:abs@abs", "vllm:abs@abs", "main:perf_x@FLA/b"]


def test_an_op_named_twice_is_selected_once(tmp_path):
    # changed_ops.txt and a diverged list joined into one file name some ops twice
    ops = tmp_path / "ops.txt"
    ops.write_text("main:abs@abs\nvllm:abs\nabs\n# a comment\nmain:abs@abs\n")
    cases = run_batch.resolve_selection(_args(tmp_path, ops_file=ops, ops=["abs"]), INVENTORY)
    assert [c.key for c in cases] == ["main:abs@abs", "vllm:abs@abs"]


@pytest.mark.parametrize("spec", ["perf_x", "nope", "other:abs", "abs@nope"])
def test_select_rejects_bad_specs(tmp_path, spec):
    with pytest.raises(SystemExit):
        run_batch.resolve_selection(_args(tmp_path, ops=[spec]), INVENTORY)


def test_partition_is_stable_and_complete(tmp_path):
    cases = run_batch.resolve_selection(_args(tmp_path, all=True), INVENTORY)
    first = run_batch.partition(cases, ["0", "1", "2"])
    again = run_batch.partition(list(reversed(cases)), ["0", "1", "2"])
    assert {g: sorted(c.key for c in cs) for g, cs in first.items()} == \
        {g: sorted(c.key for c in cs) for g, cs in again.items()}
    assert sorted(c.key for cs in first.values() for c in cs) == sorted(c.key for c in cases)


SMI_GPUS = "0, GPU-aaa, 3\n1, GPU-bbb, 1034\n2, GPU-ccc, 20480\n"
SMI_APPS = "GPU-bbb, 4242, python, 1024\nGPU-bbb, 4243, /usr/bin/foo, 10\n"


def test_gpu_parsing():
    gpus = run_batch.parse_gpus(SMI_GPUS)
    assert gpus["0"] == gpus["GPU-aaa"] == ("GPU-aaa", 3)
    assert run_batch.parse_compute_apps(SMI_APPS, "GPU-aaa") == []
    assert run_batch.parse_compute_apps(SMI_APPS, "GPU-bbb") == [
        "pid 4242 python (1024 MiB)", "pid 4243 /usr/bin/foo (10 MiB)"]


def test_gpu_users(monkeypatch):
    out = {"--query-gpu=index,uuid,memory.used": SMI_GPUS}
    monkeypatch.setattr(run_batch, "_smi", lambda args: out.get(args[0], SMI_APPS))
    assert run_batch.gpu_users("0") == []
    assert len(run_batch.gpu_users("1")) == 2
    assert run_batch.gpu_users("2") == ["20480 MiB in use by processes not visible here"]
    assert run_batch.gpu_users("9") is None


def test_wait_for_idle_gpu_policies(monkeypatch):
    monkeypatch.setattr(run_batch, "gpu_users", lambda gpu: ["pid 1 x (1 MiB)"])
    assert run_batch.wait_for_idle_gpu("0", "ignore") == ""
    assert run_batch.wait_for_idle_gpu("0", "warn") == "gpu_busy"
    calls = iter([["pid 1 x (1 MiB)"], []])
    monkeypatch.setattr(run_batch, "gpu_users", lambda gpu: next(calls))
    monkeypatch.setattr(run_batch, "BUSY_POLL_S", 0)
    assert run_batch.wait_for_idle_gpu("0", "wait").startswith("gpu_wait=")


def test_gpu_users_asks_about_one_gpu(monkeypatch):
    # a query over every GPU of a busy box takes seconds, and one is made before each run
    asked = []
    monkeypatch.setattr(run_batch, "_smi", lambda args: asked.append(args) or "")
    run_batch.gpu_users("GPU-bbb")
    assert len(asked) == 2 and all(args[-2:] == ["-i", "GPU-bbb"] for args in asked)


@pytest.fixture
def fresh_stop(monkeypatch):
    """The driver-wide stop flag and child registry, reset for one test."""
    monkeypatch.setattr(run_batch, "_stop", threading.Event())
    monkeypatch.setattr(run_batch, "_children", set())
    monkeypatch.setattr(run_batch, "TERM_GRACE_S", 2)


def _sleeper(seconds=60):
    """A stand-in for a run: its own session, with a child that would outlive it."""
    return subprocess.Popen(["bash", "-c", f"sleep {seconds} & wait"], start_new_session=True)


def _group_gone(pgid, within=3.0):
    """The process group has no member left. Its orphans are reaped by init a moment
    after they are killed, so give it a short while."""
    deadline = time.time() + within
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


def _stubborn(seconds=60):
    """A run whose leader dies on SIGTERM while a process it started ignores the signal,
    like a python stuck in a CUDA call under bash."""
    child = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             f"print('up', flush=True); time.sleep({seconds})")
    proc = subprocess.Popen(["bash", "-c", f"{sys.executable} -c \"{child}\" & wait"],
                            start_new_session=True, stdout=subprocess.PIPE)
    assert proc.stdout.readline() == b"up\n"  # the child ignores SIGTERM from here on
    return proc


def test_a_run_that_ignores_sigterm_is_killed(fresh_stop):
    proc = _stubborn()
    t0 = time.time()
    assert run_batch._wait(proc, 1) == (124, True)
    # the leader was gone at once; the group is not left to the child that stays
    assert 1 + run_batch.TERM_GRACE_S <= time.time() - t0 < 1 + run_batch.TERM_GRACE_S + 3
    assert _group_gone(proc.pid, within=1.0)


def test_timeout_kills_the_whole_process_group(fresh_stop):
    proc = _sleeper()
    assert run_batch._wait(proc, 1) == (124, True)
    assert _group_gone(proc.pid)


def test_stop_all_ends_the_runs_under_way(fresh_stop):
    proc = _sleeper()
    seen = []
    waiter = threading.Thread(target=lambda: seen.append(run_batch._wait(proc, 0)))
    waiter.start()
    while not run_batch._children:  # until _wait has registered the run
        time.sleep(0.01)
    t0 = time.time()
    run_batch.stop_all()
    waiter.join(timeout=10)
    assert not waiter.is_alive() and time.time() - t0 < 5
    assert seen[0][1] is False and seen[0][0] != 0  # ended by the signal, not a timeout
    assert _group_gone(proc.pid) and run_batch.stopping()


def test_attempt_gives_no_result_once_the_driver_stops(fresh_stop, tmp_path, monkeypatch):
    case = run_batch.Case("main", "abs", "abs", "abs")
    args = argparse.Namespace(busy_gpu="ignore", op_timeout=0)
    (tmp_path / ".triton_cache").mkdir()
    monkeypatch.setattr(run_batch, "_spawn", lambda *a: _sleeper(0))
    assert run_batch._attempt(case, tmp_path, "0", args, "") == (0, False, "")
    assert not (tmp_path / ".triton_cache").exists()  # left behind only by a killed run
    run_batch._stop.set()
    with pytest.raises(run_batch.Interrupted):
        run_batch._attempt(case, tmp_path, "0", args, "")


def test_waiting_for_a_busy_gpu_can_be_interrupted(fresh_stop, monkeypatch):
    monkeypatch.setattr(run_batch, "gpu_users", lambda gpu: ["pid 1 x (1 MiB)"])
    threading.Timer(0.2, run_batch._stop.set).start()
    t0 = time.time()
    with pytest.raises(run_batch.Interrupted):
        run_batch.wait_for_idle_gpu("0", "wait")  # BUSY_POLL_S is 30 s
    assert time.time() - t0 < 5


@pytest.fixture
def default_signals():
    """SIGINT, SIGTERM and SIGHUP as a foreground process has them, whatever pytest was
    started under (nohup, a background job). Yields their handlers."""
    wanted = {signal.SIGINT: signal.default_int_handler, signal.SIGTERM: signal.SIG_DFL,
              signal.SIGHUP: signal.SIG_DFL}
    previous = {sig: signal.signal(sig, handler) for sig, handler in wanted.items()}
    yield list(wanted.values())
    for sig, handler in previous.items():
        signal.signal(sig, handler if handler is not None else signal.SIG_DFL)


def _handlers():
    return [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)]


each_signal = pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP],
                                      ids=["SIGINT", "SIGTERM", "SIGHUP"])


@each_signal
@pytest.mark.parametrize("receiver", ["process", "another thread"])
def test_run_workers_stops_on_a_signal(fresh_stop, default_signals, sig, receiver):
    started, done = threading.Event(), []

    def job():
        for i in range(3):
            if run_batch.stopping():
                return
            proc = _sleeper(10)
            started.set()
            run_batch._wait(proc, 0)
            if run_batch.stopping():
                raise run_batch.Interrupted
            done.append(i)

    def interrupt():
        started.wait()
        if receiver == "process":
            os.kill(os.getpid(), sig)
        else:
            # The kernel may hand a signal to any thread, and one that another thread
            # receives does not wake a main thread blocked in join().
            signal.pthread_kill(threading.get_ident(), sig)

    threading.Thread(target=interrupt).start()
    t0 = time.time()
    assert run_batch.run_workers([job, job]) is False
    assert time.time() - t0 < 8 and done == []  # no run went on after the interrupt
    assert _handlers() == default_signals  # put back


def test_a_signal_while_the_jobs_are_started_stops_them_too(fresh_stop, default_signals):
    # The first job runs, and the signal arrives, before the last thread is started: the
    # main thread is then inside Thread.start(), not waiting for the jobs yet.
    def first():
        os.kill(os.getpid(), signal.SIGTERM)
        while not run_batch.stopping():
            time.sleep(0.01)
        raise run_batch.Interrupted

    t0 = time.time()
    assert run_batch.run_workers([first] + [lambda: None] * 50) is False
    assert time.time() - t0 < 5 and _handlers() == default_signals


def test_a_second_ctrl_c_does_not_leave_runs_behind(fresh_stop, default_signals, monkeypatch):
    procs, started = [], threading.Event()
    stop_all = run_batch.stop_all

    def impatient_stop_all():
        os.kill(os.getpid(), signal.SIGINT)  # Ctrl-C again, while the runs are being stopped
        stop_all()

    def job():
        procs.append(_sleeper())
        started.set()
        run_batch._wait(procs[0], 0)
        raise run_batch.Interrupted

    def ctrl_c():
        started.wait()
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(run_batch, "stop_all", impatient_stop_all)
    interrupter = threading.Thread(target=ctrl_c)
    interrupter.start()
    assert run_batch.run_workers([job]) is False
    interrupter.join()
    assert _group_gone(procs[0].pid)


@each_signal
def test_a_signal_the_caller_ignores_stays_ignored(fresh_stop, default_signals, sig):
    # under nohup, or as a background job, the driver is meant to go on
    signal.signal(sig, signal.SIG_IGN)
    seen = []
    assert run_batch.run_workers([lambda: seen.append(signal.getsignal(sig))]) is True
    assert seen == [signal.SIG_IGN] and signal.getsignal(sig) is signal.SIG_IGN


def test_run_workers_reports_a_job_error(fresh_stop):
    def broken():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_batch.run_workers([broken, lambda: None])
    assert run_batch.run_workers([lambda: None]) is True


def _status_row(case, status="PASS", **kw):
    row = {"repo": case.repo, "op": case.op, "op_file": case.op_file, "run_name": case.run_name, "gpu": "3",
               "status": status, "return_code": "0", "elapsed_ms": "41000", "rows_total": "1",
               "rows_success": "1", "notes": "", "out_dir": ""}
    row.update(kw)
    return row


def _finished(batch_dir, case, log="1 passed in 1.0s\n"):
    out = batch_dir / case.repo / case.run_name
    out.mkdir(parents=True)
    (out / "run.log").write_text(log)
    (out / ".complete").write_text("PASS\n")
    return out


def test_resume_keeps_the_row_of_a_finished_run(tmp_path, monkeypatch):
    # resuming on other GPUs must not rewrite where and how long a finished run ran
    case = run_batch.Case("main", "abs", "abs", "abs")
    _finished(tmp_path, case)
    monkeypatch.setattr(run_batch, "_spawn", lambda *a: pytest.fail("a finished op was rerun"))
    args = argparse.Namespace(resume=True)
    before = _status_row(case, notes="dtype_fallback;retry_rc=1->0", out_dir="/moved/away")
    row = run_batch.run_case(case, tmp_path, "0", args, before)
    assert (row["gpu"], row["elapsed_ms"]) == ("3", "41000")
    assert row["notes"] == "resumed;dtype_fallback;retry_rc=1->0"
    assert row["out_dir"] == str(tmp_path / "main" / "abs")  # the batch dir may have moved
    # resumed twice: the note is not repeated
    assert run_batch.run_case(case, tmp_path, "0", args, row)["notes"] == row["notes"]
    # no row to go by (a hand-made batch dir): classified from the log
    row = run_batch.run_case(case, tmp_path, "0", args, None)
    assert (row["gpu"], row["elapsed_ms"], row["notes"]) == ("0", "0", "resumed")


def test_status_rows_describe_the_batch_dir_not_the_invocation(tmp_path):
    a, b, c, d = (run_batch.Case("main", op, op, op) for op in "abcd")
    for case in (a, c):
        _finished(tmp_path, case)
    previous = {x.key: _status_row(x) for x in (a, b, c)}
    previous[b.key]["status"] = "FAIL"
    results = {a.key: _status_row(a, elapsed_ms="5")}
    # selected: a (rerun now), b (failed before, not reached this time), d (not reached);
    # c is not selected and keeps its row
    rows = run_batch.status_rows([a, b, d], results, previous, tmp_path)
    assert [(r["op"], r["status"], r["elapsed_ms"]) for r in rows] == [
        ("a", "PASS", "5"), ("b", "FAIL", "41000"), ("c", "PASS", "41000")]
    # a's run was started over and cut short: its old PASS row describes a run that is
    # gone, while b's FAIL row still says what is known about b
    (tmp_path / "main" / "a" / ".complete").unlink()
    rows = run_batch.status_rows([a, b, d], {}, previous, tmp_path)
    assert [r["op"] for r in rows] == ["b", "c"]


def test_parse_env():
    assert run_batch.parse_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(SystemExit):
        run_batch.parse_env(["novalue"])


def test_replay_dir(tmp_path):
    case = run_batch.Case("vllm", "abs", "abs", "abs")
    args = argparse.Namespace(replay_root=tmp_path)
    assert run_batch.replay_dir(case, args) == tmp_path / "vllm" / "abs"
    (tmp_path / "run.log").write_text("")
    assert run_batch.replay_dir(case, args) == tmp_path  # a single-op record
    assert run_batch.replay_dir(case, argparse.Namespace(replay_root=None)) is None
