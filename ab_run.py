#!/usr/bin/env python3
"""Repeated-measures A/B: replay one record on two compiler setups, interleaved on one
GPU per op, over several rounds.

A side is a compiler setup: a PYTHONPATH (a FlagTree python/ dir) and/or environment
variables (compile switches). Both sides replay the same record, so they run the same
autotune configs on the same shapes. Each (op, round) is one work item: the op's A and
B runs back to back on one GPU, A first in odd rounds and B first in even rounds, so
per-GPU bias and slow drift cancel within the item and order effects cancel across
rounds. Items share one queue (longest first) over all GPUs; each run is a fresh
process, so rounds are independent samples. ab_compare.py turns the rounds into
per-row verdicts.

  # 6 rounds of the ops the gate selected, one build with a compile switch
  python ab_run.py --record runs/base_multi --ops-file runs/gate_x/changed_ops.txt \\
      --a-env FLAGTREE_XXX=0 --b-env FLAGTREE_XXX=1 --out runs/ab_x --rounds 6
  # two builds
  python ab_run.py --record runs/base_multi --ops softmax \\
      --a-pythonpath /path/base/python --b-pythonpath /path/dev/python --out runs/ab_y
  # then: add rounds to ops with undecided rows, up to 14
  python ab_run.py ... --out runs/ab_x --until-decided --max-rounds 14
  # fresh rounds for an independent check (judge them with ab_compare.py --confirm);
  # if they are interrupted or a run fails, finish the same range with --from-round
  python ab_run.py ... --out runs/ab_x --ops-file recheck.txt --add-rounds 6

Output: <out>/r<k>_a and <out>/r<k>_b, each laid out like a run_batch.py batch dir
(ops_status.csv, summary.csv, <repo>/<run_name>/), and <out>/ab.json with the sides.
Rerunning with the same --out resumes: rounds finished on both sides are kept; a round
with a failed or missing side is rerun on both, so that its sides stay a pair. Ctrl-C
(or SIGTERM, or a closed terminal) stops the runs under way and exits with 130.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import statistics
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import ab_compare
import batch_summary
import run_batch
from gen_inventory import ZL_BENCH

LOG = "[ab]"
SIDES = ("a", "b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    run_batch.add_selection_args(p)
    s = p.add_argument_group("sides")
    s.add_argument("--record", type=Path, required=True,
                   help="record to replay: a run_batch.py batch dir, or a single-op run dir")
    for side in SIDES:
        S = side.upper()
        s.add_argument(f"--{side}-pythonpath", default="",
                       help=f"PYTHONPATH prepended for side {S} (a FlagTree python/ dir)")
        s.add_argument(f"--{side}-env", action="append", default=[], metavar="K=V",
                       help=f"environment variable for side {S} (repeatable)")
        s.add_argument(f"--{side}-name", default=S, help=f"display name of side {S}")

    r = p.add_argument_group("rounds")
    r.add_argument("--out", type=Path, default=None,
                   help="output dir (default runs/ab_<timestamp>); an existing one is resumed")
    r.add_argument("--rounds", type=int, default=6,
                   help="make sure every selected op has rounds 1..N (default %(default)s)")
    r.add_argument("--add-rounds", type=int, default=0, metavar="N",
                   help="instead run N new rounds per selected op, numbered after the highest "
                        "round in --out, for a check on fresh data")
    r.add_argument("--from-round", type=int, default=0, metavar="K",
                   help="with --add-rounds: the rounds are K..K+N-1 instead; finishes a range "
                        "that was interrupted or had failures (rounds already done are kept)")
    r.add_argument("--until-decided", action="store_true",
                   help="afterwards add rounds to ops with undecided rows (ab_compare.py's "
                        "rule) until every row is decided or --max-rounds is reached")
    r.add_argument("--max-rounds", type=int, default=14,
                   help="round cap for --until-decided (default %(default)s)")
    r.add_argument("--step", type=int, default=2,
                   help="rounds added per --until-decided pass (default %(default)s)")
    r.add_argument("--gate", type=Path, default=None,
                   help="ir_gate.py output dir: --until-decided extends only ops with an "
                        "undecided row that is not a control (changed, diverged or unknown); "
                        "controls ride along on those ops")
    ab_compare.rule_args(p)

    e = p.add_argument_group("execution")
    e.add_argument("--gpus", default=None,
                   help="comma-separated GPU list (default: CUDA_VISIBLE_DEVICES or all)")
    e.add_argument("--dtypes", default="", help='dtype whitelist, e.g. "bfloat16 float16"')
    e.add_argument("--shape-file", type=Path, default=None,
                   help="shape set (default: the record's archived shapes.yaml per op)")
    e.add_argument("--op-timeout", type=int, default=3600,
                   help="per-run timeout in seconds, 0 = none (default %(default)s)")
    e.add_argument("--use-flagtune", type=int, default=0, choices=(0, 1))
    e.add_argument("--busy-gpu", choices=("wait", "warn", "ignore"), default="wait",
                   help="when another process uses a GPU: wait (default), warn, or ignore")
    args = p.parse_args()

    if args.from_round and not args.add_rounds:
        raise SystemExit(f"{LOG} --from-round goes with --add-rounds")
    if args.step < 1:
        raise SystemExit(f"{LOG} --step must be at least 1")  # 0 would never reach --max-rounds
    if args.add_rounds and args.until_decided:
        raise SystemExit(f"{LOG} --add-rounds and --until-decided do not mix: run them one after the other")
    args.record = args.record.resolve()
    if not args.record.is_dir():
        raise SystemExit(f"{LOG} no record dir {args.record}")
    if args.gate and not (args.gate / "rows.tsv").is_file():
        # now, not after the rounds have run and --until-decided first reads it
        raise SystemExit(f"{LOG} no {args.gate / 'rows.tsv'}: --gate takes an ir_gate.py output dir")
    if args.shape_file:
        args.shape_file = args.shape_file.resolve()
    args.out = (args.out or ZL_BENCH / "runs" / f"ab_{datetime.now():%Y%m%d-%H%M%S}").resolve()
    # absolute paths, so that a resumed run with a relative path matches ab.json
    args.sides = {side: {"name": getattr(args, f"{side}_name"),
                         "pythonpath": os.path.abspath(getattr(args, f"{side}_pythonpath"))
                         if getattr(args, f"{side}_pythonpath") else "",
                         "env": run_batch.parse_env(getattr(args, f"{side}_env"))}
                  for side in SIDES}
    if args.sides["a"]["pythonpath"] == args.sides["b"]["pythonpath"] and \
            args.sides["a"]["env"] == args.sides["b"]["env"]:
        print(f"{LOG} note: both sides are identical, this measures the noise floor", flush=True)
    return args


# ---------------- manifest ----------------

def manifest(args: argparse.Namespace) -> dict:
    return {"record": str(args.record), "sides": args.sides, "dtypes": args.dtypes,
            "shape_file": str(args.shape_file or ""), "use_flagtune": args.use_flagtune}


def check_manifest(args: argparse.Namespace) -> None:
    """Write ab.json, or make sure a resumed dir was run with the same setup."""
    path = args.out / ab_compare.MANIFEST
    want = manifest(args)
    if path.is_file():
        have = json.loads(path.read_text())
        diff = [k for k in want if have.get(k) != want[k]]
        if diff:
            raise SystemExit(f"{LOG} {args.out} was run with a different {', '.join(diff)}; "
                             "use another --out")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(want, indent=1) + "\n")


# ---------------- rounds and cost ----------------

def done_rounds(out: Path) -> Dict[str, Dict[int, float]]:
    """{<repo>/<run_name>: {round: seconds of both sides}} of the finished rounds."""
    per: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for k, side, d in ab_compare.round_dirs(out):
        for r in batch_summary.read_table(d / "ops_status.csv"):
            if r["status"] in ("PASS", "SKIP"):
                per[f"{r['repo']}/{r['run_name']}"][k][side] = int(r["elapsed_ms"] or 0) / 1000
    return {run: {k: s["a"] + s["b"] for k, s in rs.items() if len(s) == 2}
            for run, rs in per.items()}


def record_cost(record: Path) -> Dict[str, float]:
    """Seconds per op of the record run (a proxy until the first round is done)."""
    out = {}
    for r in batch_summary.read_table(record / "ops_status.csv"):
        out[f"{r['repo']}/{r['run_name']}"] = 2 * int(r["elapsed_ms"] or 0) / 1000
    return out


def last_round(done: Dict[str, Dict[int, float]], run: str) -> int:
    """The highest finished round of an op; 0 if it has none (also when every round so
    far failed on a side, which leaves the op with an empty entry)."""
    return max(done.get(run) or [0])


def fresh_range(args: argparse.Namespace) -> Tuple[int, int]:
    """(first, last) of the rounds --add-rounds runs: after the highest round in --out,
    or from --from-round, which names the same range again to finish it after an
    interruption or a failure (a rerun without it would start yet another range)."""
    first = args.from_round or max((k for k, _, _ in ab_compare.round_dirs(args.out)), default=0) + 1
    return first, first + args.add_rounds - 1


def plan(cases, args, done) -> List[Tuple[run_batch.Case, int]]:
    """(case, round) items of this invocation, before any --until-decided pass: the
    rounds asked for that are not finished on both sides."""
    first, last = fresh_range(args) if args.add_rounds else (1, args.rounds)
    items = []
    for c in cases:
        have = done.get(f"{c.repo}/{c.run_name}", {})
        items += [(c, k) for k in range(first, last + 1) if k not in have]
    return items


# ---------------- execution ----------------

class Runner:
    """Runs (case, round) items over the GPUs; keeps the per-round ops_status.csv."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.status: Dict[str, Dict[str, Dict[str, str]]] = {}  # round dir -> {case key: row}
        self.touched = set()
        self.failed = []
        self.interrupted = False

    def _status(self, batch: str) -> Dict[str, Dict[str, str]]:
        if batch not in self.status:
            rows = batch_summary.read_table(self.args.out / batch / "ops_status.csv")
            self.status[batch] = {f"{r['repo']}:{r['op']}@{r['op_file']}": r for r in rows}
        return self.status[batch]

    def _flush(self, batch: str) -> None:
        path = self.args.out / batch / "ops_status.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=run_batch.STATUS_FIELDS)
            w.writeheader()
            w.writerows(self.status[batch].values())
        os.replace(tmp, path)

    def _forget(self, case: run_batch.Case, k: int) -> None:
        """Drop the rows an earlier invocation left for a round that is run again. The
        round is done once both sides have a row again: if this invocation is cut short
        after one side, the other must not pass for done on the strength of a run that
        was made at another time, and whose run dir the rerun may have overwritten."""
        with self.lock:
            for side in SIDES:
                batch = f"r{k}_{side}"
                if self._status(batch).pop(case.key, None) is not None:
                    self._flush(batch)
                    self.touched.add(batch)

    def side_args(self, side: str) -> argparse.Namespace:
        # resume=False: an item is a round that is not finished on both sides, and a
        # round is a pair only if both sides run now, back to back on this GPU. Keeping
        # the side an earlier invocation finished would pair it with a run made at
        # another time, most likely on another GPU.
        a, cfg = self.args, self.args.sides[side]
        return argparse.Namespace(
            resume=False, dtypes=a.dtypes, op_timeout=a.op_timeout, shape_file=a.shape_file,
            replay_root=a.record, use_flagtune=a.use_flagtune, busy_gpu=a.busy_gpu,
            env=cfg["env"], pythonpath=cfg["pythonpath"])

    def run(self, items: List[Tuple[run_batch.Case, int]], gpus: List[str],
            cost: Dict[str, float]) -> None:
        if not items:
            return
        items = sorted(items, key=lambda it: (-cost.get(f"{it[0].repo}/{it[0].run_name}", 60.0), it[1]))
        work: "queue.Queue" = queue.Queue()
        for it in items:
            work.put(it)
        done = [0]
        t0 = time.time()

        def worker(gpu: str) -> None:
            while not run_batch.stopping():
                try:
                    case, k = work.get_nowait()
                except queue.Empty:
                    return
                line = []
                self._forget(case, k)
                for side in (SIDES if k % 2 else SIDES[::-1]):
                    batch = f"r{k}_{side}"
                    try:
                        row = run_batch.run_case(case, self.args.out / batch, gpu,
                                                 self.side_args(side))
                    except run_batch.Interrupted:
                        return  # the round stays unfinished and is rerun on resume
                    except Exception as exc:  # a harness error must not take the worker down
                        row = {"repo": case.repo, "op": case.op, "op_file": case.op_file,
                               "run_name": case.run_name, "gpu": gpu, "status": "FAIL",
                               "return_code": "EXC", "elapsed_ms": "0", "rows_total": "0",
                               "rows_success": "0", "notes": f"harness_exc={exc}",
                               "out_dir": str(self.args.out / batch / case.repo / case.run_name)}
                    with self.lock:
                        self._status(batch)[case.key] = row
                        self._flush(batch)
                        self.touched.add(batch)
                        if row["status"] in ("FAIL", "TIMEOUT"):
                            self.failed.append(f"{case.repo}:{case.run_name} r{k}{side}")
                    line.append(f"{side}:{row['status']}/{int(row['elapsed_ms']) // 1000}s"
                                + (f"({row['notes']})" if row["notes"] else ""))
                with self.lock:
                    done[0] += 1
                    print(f"  ({done[0]:>4}/{len(items)}) r{k:<3} [gpu{gpu}] "
                          f"{case.repo}:{case.run_name}  " + "  ".join(line), flush=True)

        if not run_batch.run_workers([lambda g=g: worker(g) for g in gpus[:len(items)]]):
            self.interrupted = True
        print(f"{LOG} {done[0]} of {len(items)} items in {time.time() - t0:.0f} s", flush=True)

    def summarize(self) -> None:
        for batch in sorted(self.touched):
            batch_summary.summarize(self.args.out / batch)


def undecided_runs(args: argparse.Namespace) -> set:
    """<repo>/<run_name> of the ops with an undecided row (with --gate: a row that is not
    a control, i.e. changed, diverged or unknown)."""
    return {r.row.run for r in ab_compare.analyze(args.out, args.gate, **ab_compare.rule_of(args))
            if r.stats.verdict == "undecided" and r.gate != "unchanged"}


def main() -> None:
    args = parse_args()
    cases = run_batch.resolve_selection(args, run_batch.load_inventory())
    if not cases:
        raise SystemExit(f"{LOG} the selection is empty")
    check_manifest(args)
    gpus = run_batch.visible_gpus(args.gpus)
    print(f"{LOG} {len(cases)} ops, {len(gpus)} GPUs, out={args.out}", flush=True)
    for side in SIDES:
        cfg = args.sides[side]
        print(f"{LOG} {side.upper()} = {cfg['name']}: PYTHONPATH={cfg['pythonpath'] or '(inherited)'} "
              + " ".join(f"{k}={v}" for k, v in cfg["env"].items()), flush=True)

    runner = Runner(args)
    done = done_rounds(args.out)
    cost = record_cost(args.record)
    cost.update({run: statistics.median(rs.values()) for run, rs in done.items() if rs})
    # before any round is run: afterwards "the highest round in --out" is another one
    fresh_range_at_start = fresh_range(args) if args.add_rounds else None
    items = plan(cases, args, done)
    runner.run(items, gpus, cost)

    by_run = {f"{c.repo}/{c.run_name}": c for c in cases}
    finished = sum(map(len, done.values()))
    while args.until_decided and not runner.interrupted:
        done = done_rounds(args.out)
        if runner.failed and sum(map(len, done.values())) <= finished:
            print(f"{LOG} the last pass completed no round; stopping", flush=True)
            break
        finished = sum(map(len, done.values()))
        cost.update({run: statistics.median(rs.values()) for run, rs in done.items() if rs})
        need = {run for run in undecided_runs(args)
                if run in by_run and last_round(done, run) < args.max_rounds}
        if not need:
            break
        extra = []
        for run in sorted(need):
            last = last_round(done, run)
            extra += [(by_run[run], k) for k in range(last + 1, min(last + args.step, args.max_rounds) + 1)]
        print(f"{LOG} {len(need)} ops with undecided rows: {len(extra)} more items", flush=True)
        runner.run(extra, gpus, cost)

    runner.summarize()
    confirm = again = ""
    if args.add_rounds:
        first, last = fresh_range_at_start
        confirm = f" --confirm {first}-{last}"
        again = "" if args.from_round else f", adding --from-round {first}"
    if runner.interrupted:
        print(f"{LOG} interrupted; rerun the same command{again} to finish the rounds", flush=True)
        sys.exit(130)
    print(f"{LOG} done: python ab_compare.py {args.out}"
          + (f" --gate {args.gate}" if args.gate else "") + confirm, flush=True)
    if runner.failed:
        print(f"{LOG} failed runs: {', '.join(runner.failed)}; rerun the same command{again} "
              "to retry their rounds", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
