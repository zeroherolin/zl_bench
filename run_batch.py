#!/usr/bin/env python3
"""Batch driver: run run_pytest.sh once per op from the inventory, sharded across GPUs.

Each op runs in its own subprocess (a CUDA crash only loses that op) with its
own output dir <batch>/<repo>/<op>/, laid out exactly like a single-op
run_pytest.sh run (run.log / shapes.yaml / autotune_records / ttgir / ttir /
rows.tsv).
This layer only orchestrates:

- sharding: ops are hashed to GPUs by name; each GPU runs its ops serially,
  timed exactly like a single-op run. Parallel GPUs couple through power and
  thermals, so treat batch numbers as a screen and confirm suspects with a
  repeated-measures A/B (ab_run.py);
- timeout: --op-timeout kills a hung op (TIMEOUT, no retry, batch goes on);
- retries: pytest rc=1 (failure/flaky) and signal kills (rc>=128) are retried
  up to twice, stopping early when the same rc repeats. Under --dtypes there
  are two fallbacks: upstream's "can't be supported by this op" reruns without
  the filter (dtype_fallback), and any other failure with no SUCCESS row gets
  one last run without it (dtype_rescue);
- resume: --resume reuses ops with a .complete marker (written for PASS/SKIP
  only, so FAIL/TIMEOUT rerun) and keeps their rows of ops_status.csv;
- replay: --replay-root points at an earlier batch dir and passes
  REPLAY_FROM=<root>/<repo>/<op> per op, same semantics as single-op replay;
- idle GPUs: before each op starts, nvidia-smi must show no other process on
  its GPU (--busy-gpu wait, the default; warn or ignore to relax it), since a
  neighbour on the same GPU skews both the timing and the autotune choices.
  GPU numbers follow nvidia-smi (CUDA_DEVICE_ORDER=PCI_BUS_ID).

Outputs: <batch>/ops_status.csv (one row per op the batch dir holds, also those
an earlier invocation ran) and summary.csv (one row per result-table row,
written by batch_summary.py).

Examples:
  # Full screen (every collectable op of both repos, minus marked_skip and dismiss)
  python run_batch.py --all --gpus 0,1,2,3,4,5,6,7 --dtypes bfloat16

  # A subset (repo prefix defaults to main; @op_file disambiguates duplicates)
  python run_batch.py --ops softmax vllm:fused_marlin_moe_w4a16_mxfp4 nextafter_@nextafter_

  # Re-measure with pinned configs (same semantics as single-op REPLAY_FROM)
  python run_batch.py --ops-file my_ops.txt --replay-root runs/batch_xxx
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import batch_summary
from gen_inventory import OPS_DIR, REPOS, ZL_BENCH

DEFAULT_SHAPE_FILE = ZL_BENCH / "shape" / "all_ops_single.yaml"
LOG = "[batch]"
MAX_RETRIES = 2
BUSY_POLL_S = 30
BUSY_REPORT_S = 300
# Memory in use that marks a GPU busy even when nvidia-smi lists no process on it
# (processes of other containers or PID namespaces are not listed). An idle GPU
# shows a few MiB.
BUSY_MEMORY_MIB = 1024

STATUS_FIELDS = [
    "repo", "op", "op_file", "run_name", "gpu", "status", "return_code",
    "elapsed_ms", "rows_total", "rows_success", "notes", "out_dir",
]

DTYPE_UNSUPPORTED_MARK = "can't be supported by this op"
TERM_GRACE_S = 10  # between SIGTERM and SIGKILL when a run is stopped
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)  # stop the driver and its runs (run_workers)
SIGNAL_POLL_S = 0.2  # how often run_workers looks for a signal while its jobs run

# Every run is a session of its own, so that a timeout can kill the whole process group.
# That also means the runs do not get the terminal's Ctrl-C and outlive a driver that
# simply dies: the driver has to stop them itself (stop_all).
_stop = threading.Event()
_children: Set[subprocess.Popen] = set()
_children_lock = threading.Lock()


class Interrupted(Exception):
    """The driver is stopping (Ctrl-C): the op under way was cut short and has no result."""


# ---------------- inventory and selection ----------------

class Case(NamedTuple):
    repo: str
    op: str
    op_file: str
    run_name: str  # output dir name under <batch>/<repo>/

    @property
    def key(self) -> str:
        return f"{self.repo}:{self.op}@{self.op_file}"


def load_inventory() -> Dict[str, List[Dict[str, str]]]:
    inv: Dict[str, List[Dict[str, str]]] = {}
    for label in REPOS:
        path = OPS_DIR / f"inventory_{label}.csv"
        if path.is_file():
            with path.open(newline="") as f:
                inv[label] = list(csv.DictReader(f))
    if not inv:
        raise SystemExit(f"{LOG} no ops/inventory_*.csv; run gen_inventory.py first")
    return inv


def _run_name(row: Dict[str, str], dup_ops: Set[Tuple[str, str]]) -> str:
    # The dir name is the op, plus @op_file when the op name repeats within the
    # repo. Duplicates are judged on the whole inventory, not the selection, so
    # resume/replay paths stay stable. op_file may contain a subdir (test_FLA/),
    # so "/" becomes "_".
    op = row["op"]
    if (row["repo"], op) not in dup_ops:
        return op
    return f"{op}@{row['op_file'].replace('/', '_')}"


def _dup_ops(inv: Dict[str, List[Dict[str, str]]]) -> Set[Tuple[str, str]]:
    dup = set()
    for label, rows in inv.items():
        counts = Counter(r["op"] for r in rows)
        dup |= {(label, op) for op, n in counts.items() if n > 1}
    return dup


def load_dismiss(path: Path) -> Set[Tuple[str, str]]:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repo, _, op = line.rpartition(":")
        out.add((repo or "main", op))
    return out


def resolve_selection(args: argparse.Namespace,
                      inv: Dict[str, List[Dict[str, str]]]) -> List[Case]:
    dup = _dup_ops(inv)
    index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    by_op: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for label, rows in inv.items():
        for r in rows:
            index[(label, r["op"], r["op_file"])] = r
            by_op.setdefault((label, r["op"]), []).append(r)

    dismissed = load_dismiss(args.dismiss)

    def make_case(row: Dict[str, str]) -> Case:
        return Case(row["repo"], row["op"], row["op_file"], _run_name(row, dup))

    if args.all:
        cases = []
        for label, rows in inv.items():
            if args.repos and label not in args.repos:
                continue
            for r in rows:
                if r.get("collected") == "no":
                    continue
                if r.get("marked_skip") == "yes" and not args.include_marked_skip:
                    continue
                if (label, r["op"]) in dismissed:
                    continue
                cases.append(make_case(r))
        return cases

    specs: List[str] = list(args.ops or [])
    if args.ops_file:
        for line in args.ops_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                specs.append(line)
    if not specs:
        raise SystemExit(f"{LOG} no ops selected: use --all, --ops or --ops-file")

    cases, errors = [], []
    for spec in specs:
        repo, _, rest = spec.rpartition(":")
        repo = repo or "main"
        op, _, op_file = rest.partition("@")
        if repo not in inv:
            errors.append(f"{spec}: unknown repo {repo!r}")
            continue
        if op_file:
            row = index.get((repo, op, op_file))
            if row is None:
                errors.append(f"{spec}: no such inventory entry")
                continue
            cases.append(make_case(row))
            continue
        rows = by_op.get((repo, op)) or []
        if not rows:
            errors.append(f"{spec}: no such op in the inventory")
        elif len(rows) > 1:
            files = ", ".join(r["op_file"] for r in rows)
            errors.append(f"{spec}: ambiguous, add @op_file (candidates: {files})")
        else:
            cases.append(make_case(rows[0]))
    if errors:
        for e in errors:
            print(f"{LOG} error: {e}", file=sys.stderr)
        raise SystemExit(1)
    # An op named twice (two op lists joined into one --ops-file) is one op: run twice,
    # its two runs would write into the same dir at the same time.
    return list(dict.fromkeys(cases))


def add_selection_args(p: argparse.ArgumentParser) -> None:
    """Op selection flags, shared with ab_run.py."""
    sel = p.add_argument_group("op selection")
    sel.add_argument("--all", action="store_true",
                     help="the whole inventory (minus collected=no / marked_skip / dismiss)")
    sel.add_argument("--ops", nargs="*", default=None, metavar="[repo:]op[@op_file]",
                     help="ops to run (repo defaults to main; @op_file disambiguates duplicates)")
    sel.add_argument("--ops-file", type=Path, default=None,
                     help="file with one [repo:]op[@op_file] per line, # comments")
    sel.add_argument("--repos", nargs="*", default=None, choices=list(REPOS),
                     help="restrict --all to these repos (default: all)")
    sel.add_argument("--dismiss", type=Path, default=OPS_DIR / "dismiss.txt",
                     help="exclusion list (default ops/dismiss.txt; --all only)")
    sel.add_argument("--include-marked-skip", action="store_true",
                     help="keep ops with an unconditional skip mark under --all")


def parse_env(pairs: Optional[Sequence[str]]) -> Dict[str, str]:
    """K=V strings -> dict."""
    env = {}
    for kv in pairs or ():
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise SystemExit(f"expected K=V, got {kv!r}")
        env[k] = v
    return env


# ---------------- GPUs ----------------

def visible_gpus(arg: Optional[str]) -> List[str]:
    if arg:
        return [g.strip() for g in arg.split(",") if g.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], text=True,
                             capture_output=True, timeout=30).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        if n:
            return [str(i) for i in range(n)]
    except Exception:
        pass
    return ["0"]


def _smi(args: List[str]) -> Optional[str]:
    try:
        return subprocess.run(["nvidia-smi", *args], text=True, capture_output=True,
                              timeout=30, check=True).stdout
    except Exception:
        return None


def parse_gpus(text: str) -> Dict[str, Tuple[str, int]]:
    """`nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits`
    -> {index: (uuid, MiB in use)}, also keyed by uuid."""
    out = {}
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[1]:
            used = int(parts[2]) if parts[2].isdigit() else 0
            out[parts[0]] = out[parts[1]] = (parts[1], used)
    return out


def parse_compute_apps(text: str, uuid: str) -> List[str]:
    """Processes of `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory
    --format=csv,noheader,nounits` that run on the GPU with this uuid."""
    out = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4 and parts[0] == uuid:
            out.append(f"pid {parts[1]} {parts[2]} ({parts[3]} MiB)")
    return out


def gpu_users(gpu: str) -> Optional[List[str]]:
    """What is using gpu (an nvidia-smi index or a GPU uuid): its processes, or the
    memory in use when no process is listed; empty when idle, None if nvidia-smi
    cannot tell."""
    # -i asks about this GPU only: a query over all GPUs of a busy 8-GPU box takes
    # several seconds, and it is made before every run.
    gpus = _smi(["--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits", "-i", gpu])
    apps = _smi(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader,nounits", "-i", gpu])
    info = parse_gpus(gpus).get(gpu) if gpus else None
    if info is None or apps is None:
        return None
    uuid, used = info
    procs = parse_compute_apps(apps, uuid)
    if not procs and used >= BUSY_MEMORY_MIB:
        procs = [f"{used} MiB in use by processes not visible here"]
    return procs


_smi_missing_reported = threading.Event()


def wait_for_idle_gpu(gpu: str, policy: str) -> str:
    """Hold until no other process uses gpu (policy "wait"), or only report it ("warn").
    Returns a note for ops_status.csv: "", gpu_busy or gpu_wait=<seconds>s."""
    if policy == "ignore":
        return ""
    procs = gpu_users(gpu)
    if procs is None:
        if not _smi_missing_reported.is_set():
            _smi_missing_reported.set()
            print(f"{LOG} warning: nvidia-smi cannot list GPU processes; not checking that "
                  "GPUs are idle", flush=True)
        return ""
    if not procs:
        return ""
    if policy == "warn":
        print(f"{LOG} warning: gpu{gpu} is in use by {'; '.join(procs)}; measuring anyway",
              flush=True)
        return "gpu_busy"
    t0 = last = time.time()
    print(f"{LOG} gpu{gpu} is in use by {'; '.join(procs)}; waiting for it to be idle", flush=True)
    while procs:
        if _stop.wait(BUSY_POLL_S):
            raise Interrupted
        procs = gpu_users(gpu) or []
        if procs and time.time() - last >= BUSY_REPORT_S:
            last = time.time()
            print(f"{LOG} gpu{gpu} still in use ({int(last - t0)} s): {'; '.join(procs)}", flush=True)
    return f"gpu_wait={int(time.time() - t0)}s"


def partition(cases: List[Case], gpus: List[str]) -> Dict[str, List[Case]]:
    """Stable hash of (repo, run_name): independent of selection order, so a
    resumed op lands on the same GPU."""
    parts: Dict[str, List[Case]] = {g: [] for g in gpus}
    for c in cases:
        digest = hashlib.sha256(f"{c.repo}/{c.run_name}".encode()).digest()
        parts[gpus[int.from_bytes(digest[:8], "big") % len(gpus)]].append(c)
    return parts


# ---------------- per-op execution ----------------

def _spawn(case: Case, out_dir: Path, gpu: str, args: argparse.Namespace,
           dtypes: str) -> subprocess.Popen:
    env = os.environ.copy()
    # The batch arguments win: same-named variables from the caller's shell must
    # not leak into run_pytest.sh. PARALLEL_WARMUP_GPUS: the batch already
    # shards across GPUs, so no per-op parallelism.
    for key in ("PARALLEL_WARMUP_GPUS", "REPLAY_FROM", "SHAPE_FILE", "DTYPES"):
        env.pop(key, None)
    # GPU numbers as nvidia-smi shows them, which the idle check relies on.
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.update(getattr(args, "env", None) or {})
    pythonpath = getattr(args, "pythonpath", None)
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [pythonpath, env.get("PYTHONPATH")]))
    env.update({
        "OP": case.op,
        "OP_FILE": case.op_file,
        "FLAGGEMS_DIR": str(REPOS[case.repo]),
        "OUT_DIR": str(out_dir),
        "CUDA_VISIBLE_DEVICES": gpu,
        "FLAGGEMS_PERF_COLOR": "never",
        "NO_COLOR": "1",
        "USE_FLAGTUNE": str(args.use_flagtune),
    })
    if args.shape_file:  # unset under replay: run_pytest.sh reuses the A side's archive
        env["SHAPE_FILE"] = str(args.shape_file)
    if dtypes:
        env["DTYPES"] = dtypes
    replay = replay_dir(case, args)
    if replay:
        env["REPLAY_FROM"] = str(replay)
    return subprocess.Popen(
        ["bash", str(ZL_BENCH / "run_pytest.sh")],
        cwd=str(ZL_BENCH), env=env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def replay_dir(case: Case, args: argparse.Namespace) -> Optional[Path]:
    """The recorded run an op replays: <replay_root>/<repo>/<run_name>, or a
    single-op record given as replay_root itself (it has a run.log)."""
    root = getattr(args, "replay_root", None)
    if not root:
        return None
    return root if (root / "run.log").is_file() else root / case.repo / case.run_name


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal a run and everything it started (it leads its own process group)."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass  # the group is gone already


def _group_alive(proc: subprocess.Popen) -> bool:
    """The run's process group still has a member. The leader's exit does not settle
    it: bash dies on SIGTERM at once, while the python it started may be in a CUDA call
    that does not return for a while."""
    try:
        os.killpg(proc.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # a member that changed its user: there, and not ours to signal
        return True
    return True


def _terminate(procs: Sequence[subprocess.Popen]) -> None:
    """SIGTERM the runs' process groups, SIGKILL what is left of them after TERM_GRACE_S."""
    live = [p for p in procs if p.poll() is None or _group_alive(p)]
    for p in live:
        _signal_group(p, signal.SIGTERM)
    deadline = time.time() + TERM_GRACE_S
    for p in live:
        try:
            p.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            pass
    # The group of a reaped leader keeps its id while it has members, so the id cannot
    # have been given to another process.
    while any(_group_alive(p) for p in live) and time.time() < deadline:
        time.sleep(0.05)
    for p in live:
        if _group_alive(p):
            _signal_group(p, signal.SIGKILL)
        p.wait()


def stop_all() -> None:
    """Let no further run start and stop the ones under way (Ctrl-C, SIGTERM)."""
    _stop.set()
    with _children_lock:
        procs = list(_children)
    _terminate(procs)


def stopping() -> bool:
    return _stop.is_set()


def _wait(proc: subprocess.Popen, timeout: int) -> Tuple[int, bool]:
    """(rc, timed_out); on timeout the whole process group gets SIGTERM, then SIGKILL."""
    with _children_lock:
        _children.add(proc)
    try:
        if _stop.is_set():  # stop_all ran before this run was registered
            _terminate([proc])
        try:
            return proc.wait(timeout=timeout or None), False
        except subprocess.TimeoutExpired:
            _terminate([proc])
            return 124, True
    finally:
        with _children_lock:
            _children.discard(proc)


def run_case(case: Case, batch_dir: Path, gpu: str, args: argparse.Namespace,
             previous: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Run one op, or with --resume reuse its finished run. previous is the op's row in
    the batch dir's ops_status.csv, if it has one."""
    out_dir = batch_dir / case.repo / case.run_name
    marker = out_dir / ".complete"

    if args.resume and marker.is_file() and (out_dir / "run.log").is_file():
        if previous and previous.get("status") in ("PASS", "SKIP"):
            # the run's own row: the GPU it ran on, its time and its retries still hold
            notes = [n for n in previous.get("notes", "").split(";") if n and n != "resumed"]
            return dict(previous, notes=_join("resumed", *notes), out_dir=str(out_dir))
        row = _classify(case, out_dir, gpu, 0, 0, "", timed_out=False)
        row["notes"] = _join("resumed", row["notes"])
        return row

    out_dir.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)

    dtypes = args.dtypes or ""
    t0 = time.time()
    rcs: List[int] = []
    notes = ""
    timed_out = False
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            # keep the previous attempt's log
            log = out_dir / "run.log"
            if log.is_file():
                log.rename(out_dir / f"run.attempt{attempt - 1}.log")
        rc, timed_out, note = _attempt(case, out_dir, gpu, args, dtypes)
        notes = _join(notes, note)
        rcs.append(rc)
        if timed_out:
            break
        log_text = _read_log(out_dir)
        # Unsupported dtype: rerun without --dtypes (once, outside the retry budget).
        if rc != 0 and dtypes and DTYPE_UNSUPPORTED_MARK in log_text:
            dtypes = ""
            notes = _join(notes, "dtype_fallback")
            continue
        if rc == 0 or not (rc == 1 or rc >= 128):
            break
        if len(rcs) >= 2 and rcs[-1] == rcs[-2]:
            break  # deterministic: same rc twice in a row

    # dtype rescue: the filtered run failed without a single SUCCESS row, so try
    # once more without the filter. Covers baselines that fail to compile for
    # the filtered dtype (torch jiterator special functions etc.) with an error
    # other than upstream's "can't be supported by this op".
    if dtypes and not timed_out:
        probe = _classify(case, out_dir, gpu, rcs[-1], 0, "", False)
        if probe["status"] == "FAIL" and probe["rows_success"] == "0":
            log = out_dir / "run.log"
            if log.is_file():
                log.rename(out_dir / f"run.attempt{len(rcs) - 1}.log")
            rc, timed_out, note = _attempt(case, out_dir, gpu, args, "")
            notes = _join(notes, "dtype_rescue", note)
            rcs.append(rc)
    elapsed_ms = int((time.time() - t0) * 1000)

    if len(rcs) > 1:
        notes = _join(notes, "retry_rc=" + "->".join(map(str, rcs)))
    row = _classify(case, out_dir, gpu, rcs[-1], elapsed_ms, notes, timed_out)
    if row["status"] in ("PASS", "SKIP"):
        tmp = marker.with_name(".complete.tmp")
        tmp.write_text(row["status"] + "\n")
        os.replace(tmp, marker)
    return row


def _attempt(case: Case, out_dir: Path, gpu: str, args: argparse.Namespace,
             dtypes: str) -> Tuple[int, bool, str]:
    """One run of the op, once its GPU is idle: (rc, timed_out, note). Raises
    Interrupted when the driver is stopping, before or during the run."""
    if stopping():
        raise Interrupted
    note = wait_for_idle_gpu(gpu, getattr(args, "busy_gpu", "ignore"))
    try:
        rc, timed_out = _wait(_spawn(case, out_dir, gpu, args, dtypes), args.op_timeout)
    finally:
        # run_pytest.sh deletes its Triton cache itself, unless it was killed (a
        # timeout, Ctrl-C): hundreds of MB per op would pile up in the batch dir.
        shutil.rmtree(out_dir / ".triton_cache", ignore_errors=True)
    if stopping():
        raise Interrupted  # the run was cut short: its exit code means nothing
    return rc, timed_out, note


def run_workers(jobs: Sequence[Callable[[], None]]) -> bool:
    """Run the jobs, one thread each, until they return. On Ctrl-C, SIGTERM or SIGHUP,
    stop the runs under way (stop_all), wait for the jobs to wind down and return False.

    The jobs must check stopping() between runs and let Interrupted pass. Any other
    exception of a job is raised here once all jobs are done.
    """
    errors: List[BaseException] = []
    received: List[int] = []  # the signals on_signal took note of

    def guarded(job: Callable[[], None]) -> None:
        try:
            job()
        except Interrupted:
            pass
        except BaseException as exc:  # reported by the caller's thread, below
            errors.append(exc)

    def on_signal(signum, frame):
        received.append(signum)

    # SIGTERM and a closed terminal (SIGHUP) stop the driver the way Ctrl-C does: left to
    # their defaults they kill it, and its runs, sessions of their own, go on without it.
    # A signal the caller ignores (nohup, a background job) stays ignored.
    #
    # The handler only takes note, and the loop below acts on it. A handler that raises
    # does so wherever the main thread happens to be, between starting two threads as
    # well. And the main thread polls: the kernel may hand a signal to any thread, which
    # does not wake a main thread blocked in join().
    saved: Dict[int, object] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in STOP_SIGNALS:
            if signal.getsignal(sig) is not signal.SIG_IGN:
                saved[sig] = signal.signal(sig, on_signal)
    threads = [threading.Thread(target=guarded, args=(job,)) for job in jobs]
    started: List[threading.Thread] = []
    try:
        for t in threads:
            if received:
                break
            t.start()
            started.append(t)
        alive = started
        while alive and not received:
            alive[0].join(SIGNAL_POLL_S)
            alive = [t for t in alive if t.is_alive()]
        if received:
            # Further signals are only noted: stopping the runs must not be cut short,
            # or they would be left behind. It takes TERM_GRACE_S at most.
            print(f"{LOG} interrupted: stopping the runs under way", flush=True)
            stop_all()
        for t in started:
            t.join()
    finally:
        for sig, handler in saved.items():
            # None: a handler that was not installed from Python cannot be put back
            signal.signal(sig, handler if handler is not None else signal.SIG_DFL)
    if errors:
        raise errors[0]
    return not received


def _read_log(out_dir: Path) -> str:
    log = out_dir / "run.log"
    return log.read_text(errors="replace") if log.is_file() else ""


def _join(*notes: str) -> str:
    return ";".join(n for n in notes if n)


def _classify(case: Case, out_dir: Path, gpu: str, rc: int, elapsed_ms: int,
              notes: str, timed_out: bool) -> Dict[str, str]:
    text = _read_log(out_dir)
    rows = batch_summary.parse_run_log(text)
    if timed_out:
        status = "TIMEOUT"
    else:
        status = batch_summary.classify(rc, rows, text)
    n_skipped = batch_summary.pytest_skipped_count(text)
    if status == "PASS" and n_skipped:
        notes = _join(notes, f"partial_skip={n_skipped}")
    return {
        "repo": case.repo,
        "op": case.op,
        "op_file": case.op_file,
        "run_name": case.run_name,
        "gpu": gpu,
        "status": status,
        "return_code": str(rc),
        "elapsed_ms": str(elapsed_ms),
        "rows_total": str(len(rows)),
        "rows_success": str(sum(1 for r in rows if r["row_status"] == "SUCCESS")),
        "notes": _join(notes, batch_summary.marker_notes(text)),
        "out_dir": str(out_dir),
    }


def status_rows(cases: Sequence[Case], results: Dict[str, Dict[str, str]],
                previous: Dict[str, Dict[str, str]], batch_dir: Path) -> List[Dict[str, str]]:
    """The rows of ops_status.csv: the selected ops in selection order, then the other
    ops the batch dir holds. results are this invocation's rows, previous the ones an
    earlier invocation left, both keyed by Case.key.

    The file describes the batch dir, not one invocation, so an op that was not run this
    time (not selected, or not reached before an interrupt) keeps its row. Except a
    PASS/SKIP row whose run was started over and has not finished: that run dir no
    longer holds the result the row describes (.complete is removed when a run starts).
    """
    def standing(case: Case) -> Optional[Dict[str, str]]:
        row = previous.get(case.key)
        if row and row.get("status") in ("PASS", "SKIP") \
                and not (batch_dir / case.repo / case.run_name / ".complete").is_file():
            return None
        return row

    rows = [results.get(c.key) or standing(c) for c in cases]
    selected = {c.key for c in cases}
    return [r for r in rows if r] + [r for key, r in previous.items() if key not in selected]


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(p)

    run = p.add_argument_group("execution")
    run.add_argument("--gpus", default=None,
                     help="comma-separated GPU list (default: CUDA_VISIBLE_DEVICES or all)")
    run.add_argument("--batch-dir", type=Path, default=None,
                     help="output dir (default runs/batch_<timestamp>; required with --resume)")
    run.add_argument("--op-timeout", type=int, default=1800,
                     help="per-op timeout in seconds, 0 = none (default %(default)s)")
    run.add_argument("--resume", action="store_true",
                     help="reuse ops in --batch-dir that carry a .complete marker")
    run.add_argument("--replay-root", type=Path, default=None,
                     help="earlier batch dir; each op replays <root>/<repo>/<op>")
    run.add_argument("--shape-file", type=Path, default=None,
                     help=f"shape set (default {DEFAULT_SHAPE_FILE.relative_to(ZL_BENCH)}; "
                          "shape/all_ops_multi.yaml for a deep run). With --replay-root "
                          "the default is each op's archived shapes.yaml from the A side")
    run.add_argument("--dtypes", default="",
                     help='dtype whitelist, space separated (e.g. "bfloat16"); empty = upstream sweep')
    run.add_argument("--use-flagtune", type=int, default=0, choices=(0, 1),
                     help="passed through as USE_FLAGTUNE (default 0; must match on both A/B sides)")
    run.add_argument("--env", action="append", default=[], metavar="K=V",
                     help="extra environment variable for every op, e.g. a compiler switch "
                          "(repeatable)")
    run.add_argument("--busy-gpu", choices=("wait", "warn", "ignore"), default="wait",
                     help="when another process uses an op's GPU: wait until it is idle "
                          "(default), warn and run, or ignore")
    args = p.parse_args()
    args.env = parse_env(args.env)

    if args.resume and not args.batch_dir:
        raise SystemExit(f"{LOG} --resume needs an explicit --batch-dir")
    if args.shape_file is None and not args.replay_root:
        args.shape_file = DEFAULT_SHAPE_FILE
    if args.shape_file:
        args.shape_file = args.shape_file.resolve()
        if not args.shape_file.is_file():
            raise SystemExit(f"{LOG} no shape file {args.shape_file}")
    if args.replay_root:
        args.replay_root = args.replay_root.resolve()
        if not args.replay_root.is_dir():
            raise SystemExit(f"{LOG} no replay dir {args.replay_root}")
    return args


def main() -> None:
    args = parse_args()
    inv = load_inventory()
    cases = resolve_selection(args, inv)
    if not cases:
        raise SystemExit(f"{LOG} the selection is empty")

    gpus = visible_gpus(args.gpus)
    batch_dir = (args.batch_dir or
                 ZL_BENCH / "runs" / f"batch_{datetime.now():%Y%m%d-%H%M%S}")
    batch_dir = batch_dir.resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)

    parts = {g: cs for g, cs in partition(cases, gpus).items() if cs}
    print(f"{LOG} {len(cases)} ops -> {len(parts)} GPUs "
          f"(shards: {[len(cs) for cs in parts.values()]})  out={batch_dir}")
    if args.replay_root:
        print(f"{LOG} replay mode: REPLAY_FROM={args.replay_root}/<repo>/<op>")

    lock = threading.Lock()
    results: Dict[str, Dict[str, str]] = {}
    status_csv = batch_dir / "ops_status.csv"
    previous = {f"{r['repo']}:{r['op']}@{r['op_file']}": r
                    for r in batch_summary.read_table(status_csv)}
    wname = max((len(c.run_name) for c in cases), default=8)

    def flush_status() -> None:
        rows = status_rows(cases, results, previous, batch_dir)
        tmp = status_csv.with_name(status_csv.name + ".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, status_csv)

    def run_partition(gpu: str, part: List[Case]) -> None:
        for case in part:
            if stopping():
                return
            try:
                row = run_case(case, batch_dir, gpu, args, previous.get(case.key))
            except Interrupted:
                return
            except Exception as exc:  # a harness error must not take the shard down
                row = {
                    "repo": case.repo, "op": case.op, "op_file": case.op_file,
                    "run_name": case.run_name, "gpu": gpu, "status": "FAIL",
                    "return_code": "EXC", "elapsed_ms": "0", "rows_total": "0",
                    "rows_success": "0", "notes": f"harness_exc={exc}",
                    "out_dir": str(batch_dir / case.repo / case.run_name),
                }
            with lock:
                results[case.key] = row
                flush_status()
                print(f"  ({len(results):>4}/{len(cases)})  {row['status']:<7} "
                      f"{row['elapsed_ms']:>8}ms  rows={row['rows_success']}/"
                      f"{row['rows_total']:<4} {case.repo}:{case.run_name:<{wname}} "
                      f"[gpu{gpu}]"
                      + (f"  {row['notes']}" if row["notes"] else ""),
                      flush=True)

    completed = run_workers([lambda g=g, p=p: run_partition(g, p) for g, p in parts.items()])

    flush_status()  # also when no op finished, so the paths printed below exist
    summary = batch_summary.summarize(batch_dir)
    rows = [results[c.key] for c in cases if c.key in results]
    counts = {s: sum(1 for r in rows if r["status"] == s)
              for s in ("PASS", "SKIP", "FAIL", "TIMEOUT")}
    print(f"{LOG} done: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"{LOG} status:  {status_csv}")
    print(f"{LOG} summary: {summary}")
    if not completed:
        print(f"{LOG} interrupted with {len(rows)} of {len(cases)} ops done; to go on: "
              f"--batch-dir {batch_dir} --resume")
        sys.exit(130)
    bad = [r for r in rows if r["status"] in ("FAIL", "TIMEOUT")]
    if bad:
        print(f"{LOG} failed: " +
              ", ".join(f"{r['repo']}:{r['run_name']}" for r in bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
