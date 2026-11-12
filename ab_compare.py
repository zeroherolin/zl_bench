#!/usr/bin/env python3
"""A/B comparison of Gems latency, per result row.

Repeated-measures A/B (the output dir of ab_run.py): every row has one sample per
round and side, and gets a verdict from both samples (ab_stats.py):

  python ab_compare.py runs/ab_x [--gate runs/gate_x] [--rounds 7-12] [--csv rows.csv] [-v]

  verdict  faster / slower: |median gap| > --threshold, shown by the rank test
           (Mann-Whitney p < --alpha) or by the pairing (every round agrees and the
           sign test is below --alpha-paired)
           same: within the threshold, or no detectable shift (p >= --clear)
           undecided: fewer than --min-rounds samples, or in between (add rounds)
  gap      median(A) / median(B) - 1, > 0 means B is faster

With --gate (an ir_gate.py output dir), rows whose kernels all compiled to the same
binary on both sides are controls: they cannot differ, so any verdict other than
same/undecided among them is a false positive, and their rate is the measured
false-positive rate of the rule. The other rows count as results: those with a changed
kernel, those the gate cleared although the two sides launched different binaries
(diverged), and those that cannot be attributed (unknown). --rounds restricts the
analysis to a range of rounds.

Independent recheck of weak results, on fresh rounds:

  python ab_compare.py runs/ab_x --gate G --recheck-list recheck.txt   # ops with weak rows
  python ab_run.py ... --out runs/ab_x --ops-file recheck.txt --add-rounds 6   # say rounds 15-20
  python ab_compare.py runs/ab_x --gate G --confirm 15-20

A faster/slower row (not a control) is weak if |gap| < --weak-gap (2%), |gap| < 2x
the larger coefficient of variation of its two samples, its table was timed by plain
do_bench, or only the pairing shows it. With --confirm, every row is judged on the
other rounds first; each faster/slower row with samples in the confirm rounds is then
confirmed only if those rounds alone show a gap beyond the threshold in the same
direction, one-sided at --confirm-alpha (0.05): by the rank test, or by the pairing
when every confirm round agrees. Confirmed rows report the gap over all rounds; rows
that are not reproduced become undecided. Confirm rounds too few to show anything at
0.05 are reported as too_few_rounds and the row keeps its verdict: fewer than four,
or four that all agree while their samples overlap (the pairing needs five).

A round counts only if both of its sides ran to their end, on the same GPU. With
--gate, a row the gate cleared is a control only if both sides launched the same
binaries in every round (each run records them in its rows.tsv): if they differ the
row is diverged, and if a run could not tell, or had a replay fallback and recorded
none, it is unknown. The gate compared the recorded configs with the recorded
options, not what such a run launched.

Single comparison of two run dirs or two run_batch.py batch dirs (one sample per
row, no verdict; B usually replayed A's record):

  python ab_compare.py runs/softmax_A runs/softmax_B [--min 0.03] [--quiet]
  python ab_compare.py runs/batch_A runs/batch_B

Ratio = A/B Gems latency (>1 means B is faster). Rows are paired by (table op_name,
dtype, row index) and count only if SUCCESS on both sides. A single comparison is a
screen: run-to-run noise and per-GPU bias are a few percent; use ab_run.py for
conclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import ab_stats
from batch_summary import OPERATOR_HEADER_RE, parse_run_log, read_table

NOCG = "BENCHMARK_DIRECT_NO_CUDAGRAPH"
FALLBACK = "AUTOTUNE_REPLAY_FALLBACK"
MANIFEST = "ab.json"

Key = Tuple[str, str, int]  # (table op_name, dtype, row index)


# ---------------- run.log ----------------

def load_rows(run_dir: Path) -> Dict[Key, Tuple[float, str]]:
    """{(op_name, dtype, row): (Gems latency, size detail)} of the SUCCESS rows. The
    row index counts every row of the table, failed ones included, so that rows pair
    by position even when one side fails a row."""
    log = run_dir / "run.log"
    if not log.is_file():
        return {}
    out: Dict[Key, Tuple[float, str]] = {}
    seq: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in parse_run_log(log.read_text(errors="replace")):
        key = (r["fg_op_name"], r["dtype"])
        i = seq[key]
        seq[key] += 1
        if r["row_status"] != "SUCCESS":
            continue
        try:
            lat = float(r["gems_lat"])
        except ValueError:
            continue
        if lat > 0:
            out[key + (i,)] = (lat, r["size_detail"])
    return out


def nocg_tables(text: str) -> set:
    """(op_name, dtype) of the tables with a row timed by plain do_bench: the fallback
    marker is printed while the table's rows are measured, before its header."""
    out, pending = set(), 0
    for line in text.splitlines():
        s = line.strip()
        if NOCG in s:
            pending += 1
            continue
        h = OPERATOR_HEADER_RE.match(s)
        if h:
            if pending:
                out.add((h.group("name"), h.group("dtype").strip()))
            pending = 0
    return out


def _fallback(run_dir: Path) -> bool:
    log = run_dir / "run.log"
    return log.is_file() and FALLBACK in log.read_text(errors="replace")


# ---------------- single comparison ----------------

def compare_pair(a_dir: Path, b_dir: Path, label: str, thr: float,
                 verbose: bool) -> Optional[Tuple[float, int]]:
    a, b = load_rows(a_dir), load_rows(b_dir)
    keys = [k for k in a if k in b]
    if not keys:
        return None
    ratios = [a[k][0] / b[k][0] for k in keys]
    gm = ab_stats.geomean(ratios)
    flag = "  [replay_fallback]" if _fallback(b_dir) else ""
    print(f"{label:60s} rows={len(keys):4d} geomean={gm:.3f} "
          f"min={min(ratios):.3f} max={max(ratios):.3f}{flag}")
    if verbose:
        for k, r in zip(keys, ratios, strict=True):
            if abs(r - 1) >= thr:
                print(f"    {r:6.3f}  {a[k][0]:.6f} -> {b[k][0]:.6f}  {k[0]} {k[1]} "
                      f"#{k[2]} {a[k][1][:100]}")
    return gm, len(keys)


def single(args: argparse.Namespace) -> None:
    if (args.b / "run.log").is_file():
        if compare_pair(args.a, args.b, args.b.name, args.min, not args.quiet) is None:
            sys.exit("no SUCCESS rows to pair")
        return
    results = []
    for b_dir in sorted(d for d in args.b.glob("*/*") if (d / "run.log").is_file()):
        a_dir = args.a / b_dir.parent.name / b_dir.name
        r = compare_pair(a_dir, b_dir, f"{b_dir.parent.name}:{b_dir.name}", args.min,
                         not args.quiet)
        if r:
            results.append(r)
    if not results:
        sys.exit("no ops to pair")
    gm = ab_stats.geomean([r[0] for r in results])
    print(f"== {len(results)} ops, geomean of per-op geomeans {gm:.4f}; "
          f"<0.97: {sum(r[0] < 0.97 for r in results)}, "
          f">1.03: {sum(r[0] > 1.03 for r in results)}")


# ---------------- repeated-measures A/B ----------------

def parse_range(text: Optional[str]) -> Optional[Tuple[int, int]]:
    """"7-12" / "7-" / "-6" / "3" -> inclusive (lo, hi); None = every round."""
    if not text:
        return None
    lo, sep, hi = text.partition("-")
    try:
        if not sep:
            return int(lo), int(lo)
        return int(lo) if lo else 1, int(hi) if hi else 10 ** 9
    except ValueError:
        raise SystemExit(f"expected a round range like 7-12, 7-, -6 or 3, got {text!r}") from None


def round_dirs(ab_dir: Path, rng: Optional[Tuple[int, int]] = None,
               exclude: Optional[Tuple[int, int]] = None) -> List[Tuple[int, str, Path]]:
    """[(round, side, dir)] of r<k>_a / r<k>_b, ordered by round; rng keeps, exclude drops
    an inclusive range of rounds."""
    out = []
    for d in ab_dir.glob("r*_[ab]"):
        head, _, side = d.name.partition("_")
        if d.is_dir() and head[1:].isdigit():
            k = int(head[1:])
            if rng is not None and not rng[0] <= k <= rng[1]:
                continue
            if exclude is not None and exclude[0] <= k <= exclude[1]:
                continue
            out.append((k, side, d))
    return sorted(out)


def _status_rows(ab_dir: Path):
    """(round, side, row) of every ops_status.csv row of the rounds."""
    for k, side, d in round_dirs(ab_dir):
        for r in read_table(d / "ops_status.csv"):
            yield k, side, r


def op_labels(ab_dir: Path) -> Dict[str, str]:
    """<repo>/<run_name> -> repo:op@op_file, from the rounds' ops_status.csv."""
    return {f"{r['repo']}/{r['run_name']}": f"{r['repo']}:{r['op']}@{r['op_file']}"
            for _, _, r in _status_rows(ab_dir)}


def round_status(ab_dir: Path) -> Dict[Tuple[str, int], Dict[str, Dict[str, str]]]:
    """(<repo>/<run_name>, round) -> side -> its ops_status.csv row."""
    out: Dict[Tuple[str, int], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for k, side, r in _status_rows(ab_dir):
        out[(f"{r['repo']}/{r['run_name']}", k)][side] = r
    return out


def split_rounds(ab_dir: Path) -> Set[Tuple[str, int]]:
    """(<repo>/<run_name>, round) whose two sides ran on different GPUs.

    A round is a pair because its sides run back to back on one GPU. A side that failed
    and was rerun by a later invocation may have landed on another GPU; such a round
    compares two GPUs as much as two compilers, so it is not used.
    """
    return {key for key, sides in round_status(ab_dir).items()
               if len({r["gpu"] for r in sides.values() if r.get("gpu")}) > 1}


class Row(NamedTuple):
    run: str          # <repo>/<run_name>
    op_name: str
    dtype: str
    row: int
    size: str
    a: List[float]
    b: List[float]
    rounds: List[int]  # rounds with a sample on both sides
    nocg: bool         # its table had a sample timed by plain do_bench (marked per table)
    fallback: bool     # a side did not replay every recorded config
    binaries: str = ""  # both sides launched the same binaries: same / differ / unknown / "" (_same_binaries)
    failed: bool = False  # a side of one of its rounds failed or timed out after measuring the row


def load_samples(ab_dir: Path, rng: Optional[Tuple[int, int]] = None,
                 exclude: Optional[Tuple[int, int]] = None) -> List[Row]:
    """Every row with its per-round samples.

    A round counts for a row only if both of its sides ran to their end as one pair, on
    the same GPU (split_rounds), and measured the row. ab_run.py writes a side's row of
    ops_status.csv when its run ends, and drops both rows when it runs a round again:
    a round with a row on one side only was cut short, and what its run dirs hold is
    half of it, or the log of an earlier attempt. A side that failed or timed out did
    end: the rows it measured before that count, and the row is marked (Row.failed).
    """
    per: Dict[Tuple[str, str, str, int], Dict[int, Dict[str, float]]] = defaultdict(dict)
    size: Dict[Tuple[str, str, str, int], str] = {}
    nocg, fallback, failed = set(), set(), set()
    binaries: Dict[Tuple[str, str, str, int], Dict[int, Dict[str, str]]] = defaultdict(dict)
    status = round_status(ab_dir)
    split = split_rounds(ab_dir)
    for k, side, d in round_dirs(ab_dir, rng, exclude):
        for run_dir in sorted(p for p in d.glob("*/*") if (p / "run.log").is_file()):
            run = f"{run_dir.parent.name}/{run_dir.name}"
            sides = status.get((run, k), {})
            if (run, k) in split or len(sides) < 2:
                continue
            if any(r.get("status") in ("FAIL", "TIMEOUT") for r in sides.values()):
                failed.add((run, k))
            text = (run_dir / "run.log").read_text(errors="replace")
            nocg |= {(run,) + t for t in nocg_tables(text)}
            if FALLBACK in text:
                fallback.add(run)
            ran = load_binaries(run_dir)
            for (op_name, dtype, i), (lat, sz) in load_rows(run_dir).items():
                key = (run, op_name, dtype, i)
                per[key].setdefault(k, {})[side] = lat
                binaries[key].setdefault(k, {})[side] = ran.get((op_name, dtype, i), "")
                size.setdefault(key, sz)
    rows = []
    for key, by_round in sorted(per.items()):
        rounds = sorted(k for k, s in by_round.items() if len(s) == 2)
        if not rounds:
            continue
        rows.append(Row(*key, size[key], [by_round[k]["a"] for k in rounds],
                        [by_round[k]["b"] for k in rounds], rounds,
                        key[:3] in nocg, key[0] in fallback,
                        _same_binaries([binaries[key][k] for k in rounds]),
                        any((key[0], k) in failed for k in rounds)))
    return rows


def load_binaries(run_dir: Path) -> Dict[Key, str]:
    """(op_name, dtype, row) -> the binaries the row's measurement launched, from the
    run's rows.tsv: the sorted fingerprints (_binary_id) of its kernels, "-" for a row
    that launched none, "?" where the run could not tell (a row it did not track, a
    kernel without a binary in the cache). Rows of a run that recorded none, or cut
    off in a damaged file, are left out."""
    out: Dict[Key, str] = {}
    for r in read_table(run_dir / "rows.tsv", "\t"):
        ids = [x for x in (r.get("binaries") or "").split(",") if x]
        if ids and (r.get("row") or "").isdigit():
            out[(r["op_name"], r["dtype"], int(r["row"]))] = "?" if "?" in ids else ",".join(sorted(ids))
    return out


def _same_binaries(per_round: List[Dict[str, str]]) -> str:
    """"differ" when the two sides ran different binaries in some round, "same" when
    they ran the same ones in every round, "unknown" when a run recorded that it could
    not tell, "" when the runs recorded no binaries."""
    known = [{side: ids for side, ids in s.items() if ids and ids != "?"} for s in per_round]
    if any(len(s) == 2 and s["a"] != s["b"] for s in known):
        return "differ"
    if any("?" in s.values() for s in per_round):
        return "unknown"
    return "same" if all(len(s) == 2 for s in known) else ""


def load_gate(gate_dir: Path) -> Dict[Tuple[str, str, str, int], str]:
    """{(run, op_name, dtype, row): changed 1/0/?} from ir_gate.py's rows.tsv."""
    path = gate_dir / "rows.tsv"
    if not path.is_file():
        raise SystemExit(f"no {path}: run ir_gate.py on a record made with row tracking")
    out = {}
    for r in read_table(path, "\t"):
        out[(r["run"], r["op_name"], r["dtype"], int(r["row"]))] = r["changed"]
    return out


def gate_flag(gate: Dict[Tuple[str, str, str, int], str], row: Row) -> str:
    """changed / unchanged / diverged / unknown for a row; a single-op record has run ""
    in the gate.

    The gate compiled the recorded variants with the recorded options on both builds.
    Whether the two sides of the A/B launched those is a separate question: a replay
    fallback tunes a config on the spot, and an operator library may set a compile
    option only where the compiler has it. The rounds answer it directly, since every
    run records the binaries its rows launched:

    - unchanged: the gate cleared the row, and the sides ran the same binaries (or the
      runs recorded no binaries and had no replay fallback): a control;
    - diverged: the gate cleared the row, yet the sides ran different binaries. The
      row is a result like a changed one, of a difference the gate could not see;
    - unknown: not attributable by the gate; or cleared by it while a run could not
      tell which binaries the row launched, or had a replay fallback and recorded none.
    """
    flag = gate.get((row.run, row.op_name, row.dtype, row.row))
    if flag is None:
        flag = gate.get(("", row.op_name, row.dtype, row.row))
    if flag == "0":
        if row.binaries == "differ":
            return "diverged"
        if row.binaries == "unknown" or (row.fallback and not row.binaries):
            return "unknown"
        return "unchanged"
    return "changed" if flag == "1" else "unknown"


class Result(NamedTuple):
    row: Row
    stats: ab_stats.RowStats
    gate: str  # changed / unchanged / diverged / unknown (gate_flag), or "" without --gate
    confirm: str = ""  # --confirm: confirmed / not_reproduced / too_few_rounds / "" (no fresh rounds)
    # Statistics of the confirm rounds alone; p and p_paired are one-sided, in the
    # direction of the verdict under recheck.
    new: Optional[ab_stats.RowStats] = None


def analyze(ab_dir: Path, gate_dir: Optional[Path] = None, rng: Optional[Tuple[int, int]] = None,
            exclude: Optional[Tuple[int, int]] = None, **rule) -> List[Result]:
    gate = load_gate(gate_dir) if gate_dir else None
    out = []
    for row in load_samples(ab_dir, rng, exclude):
        out.append(Result(row, ab_stats.compare(row.a, row.b, **rule),
                          gate_flag(gate, row) if gate is not None else ""))
    return out


def is_weak(r: Result, weak_gap: float) -> bool:
    """A faster/slower row whose evidence is thin: a small gap, a gap within twice the
    samples' coefficient of variation, timing by plain do_bench, a round in which a
    side failed after measuring the row, or a verdict that rests on the pairing alone
    (the sign test's level is looser than the rank test's)."""
    if r.stats.verdict not in ("faster", "slower"):
        return False
    cv = max(statistics.pstdev(x) / statistics.fmean(x) for x in (r.row.a, r.row.b))
    return (abs(r.stats.gap) < weak_gap or abs(r.stats.gap) < 2 * cv or r.row.nocg
            or r.row.failed or r.stats.basis == "pairing")


def recheck(verdict: str, a: List[float], b: List[float], threshold: float,
            confirm_alpha: float) -> Tuple[str, ab_stats.RowStats]:
    """Whether fresh rounds alone reproduce a faster/slower verdict.

    Reproduced: the fresh gap is beyond the threshold in the same direction, and a
    one-sided test shows it at confirm_alpha: the rank test, or the pairing when every
    fresh round agrees. One-sided, because the direction was fixed before the fresh
    rounds existed. Either test may confirm, whichever gave the verdict: a row decided
    by the pairing is one the rank test cannot separate, on fresh rounds as before.

    Rounds too few to reach confirm_alpha say nothing, which is not the same as failing
    to reproduce: three rounds at 0.05 whatever they show, and four when they all agree
    but overlap, since four agreeing rounds are a sign test at 0.0625.
    """
    alternative = "greater" if verdict == "faster" else "less"
    p = ab_stats.mann_whitney_u(a, b, alternative)[1]
    p_paired = ab_stats.sign_test(a, b, alternative)[2]
    gap = statistics.median(a) / statistics.median(b) - 1
    stats = ab_stats.RowStats(len(a), len(b), statistics.median(a), statistics.median(b), gap, p,
                              "", p_paired)
    beyond = gap > threshold if verdict == "faster" else gap < -threshold
    agree = ab_stats.pairs_agree(a, b, alternative)
    # First: with tied samples the rank test is a normal approximation, which can go
    # below what so few rounds can show at all.
    if ab_stats.smallest_p(len(a)) >= confirm_alpha:
        return "too_few_rounds", stats
    if beyond and (p < confirm_alpha or (agree and p_paired < confirm_alpha)):
        return "confirmed", stats
    if beyond and agree:
        return "too_few_rounds", stats
    return "not_reproduced", stats


def _only(row: Row, keep: Callable[[int], bool]) -> Optional[Row]:
    """The row with the rounds keep() selects; None if it has none of them."""
    idx = [i for i, k in enumerate(row.rounds) if keep(k)]
    if not idx:
        return None
    return row._replace(a=[row.a[i] for i in idx], b=[row.b[i] for i in idx],
                        rounds=[row.rounds[i] for i in idx])


def confirm(ab_dir: Path, gate_dir: Optional[Path], rng: Tuple[int, int], confirm_alpha: float,
            **rule) -> List[Result]:
    """Judge every row on the rounds outside rng, then recheck its faster/slower verdict on
    the rounds inside rng alone (recheck). A confirmed row reports all its rounds; one
    that is not reproduced becomes undecided; one with too few fresh rounds is left as
    judged. What a row is (a control or a result, how it was timed) is told by all its
    rounds, the confirm rounds included."""
    gate = load_gate(gate_dir) if gate_dir else None
    out = []
    for whole in load_samples(ab_dir):
        row = _only(whole, lambda k: not rng[0] <= k <= rng[1])
        if row is None:
            continue
        flag = gate_flag(gate, whole) if gate is not None else ""
        stats = ab_stats.compare(row.a, row.b, **rule)
        new = _only(whole, lambda k: rng[0] <= k <= rng[1])
        if stats.verdict not in ("faster", "slower") or new is None:
            out.append(Result(row, stats, flag))
            continue
        outcome, stats_new = recheck(stats.verdict, new.a, new.b, rule["threshold"], confirm_alpha)
        if outcome == "confirmed":
            all_stats = ab_stats.compare(whole.a, whole.b, **rule)
            all_stats = all_stats._replace(verdict=stats.verdict,
                                           basis=all_stats.basis or stats.basis)
            out.append(Result(whole, all_stats, flag, outcome, stats_new))
        elif outcome == "not_reproduced":
            out.append(Result(row, stats._replace(verdict="undecided", basis=""), flag,
                              outcome, stats_new))
        else:
            out.append(Result(row, stats, flag, outcome, stats_new))
    return out


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _p(x: float) -> str:
    return "" if math.isnan(x) else f"{x:.4g}"


def _evidence(s: ab_stats.RowStats) -> str:
    """The p-values of a row for the report, with the test that decided it marked."""
    text = f"p={s.p:.2g}"
    if not math.isnan(s.p_paired):
        text += f" p_paired={s.p_paired:.2g}"
    return text + (" [pairing]" if s.basis == "pairing" else "")


def report(results: List[Result], labels: Dict[str, str], manifest: dict,
           verbose: bool, quiet: bool) -> None:
    sides = manifest.get("sides", {})
    for s in ("a", "b"):
        cfg = sides.get(s, {})
        what = " ".join(([f"PYTHONPATH={cfg['pythonpath']}"] if cfg.get("pythonpath") else [])
                        + [f"{k}={v}" for k, v in (cfg.get("env") or {}).items()])
        print(f"{s.upper()} = {cfg.get('name', s.upper())}: {what or '(caller environment)'}")
    gated = any(r.gate for r in results)
    tested = [r for r in results if not gated or r.gate != "unchanged"]
    controls = [r for r in results if gated and r.gate == "unchanged"]
    if results:
        n = [len(r.row.rounds) for r in results]
        print(f"record {manifest.get('record', '?')}; rounds per row {min(n)}-{max(n)}; "
              "gap = median(A)/median(B) - 1, > 0: B faster")

    by_op: Dict[str, List[Result]] = defaultdict(list)
    for r in tested:
        by_op[r.row.run].append(r)
    if not quiet and by_op:
        print(f"\n{'op':56s} {'rows':>5} {'faster':>7} {'slower':>7} {'same':>6} {'undec':>6}"
              f" {'fastest':>8} {'slowest':>8}")
        for run, rs in sorted(by_op.items()):
            v = defaultdict(int)
            for r in rs:
                v[r.stats.verdict] += 1
            fast = max((r.stats.gap for r in rs if r.stats.verdict == "faster"), default=None)
            slow = min((r.stats.gap for r in rs if r.stats.verdict == "slower"), default=None)
            flags = []
            if any(r.row.nocg for r in rs):
                flags.append("do_bench")
            if any(r.row.fallback for r in rs):
                flags.append("replay_fallback")
            if any(r.row.failed for r in rs):
                flags.append("failed_side")
            fast_s = _pct(fast) if fast is not None else "-"
            slow_s = _pct(slow) if slow is not None else "-"
            print(f"{labels.get(run, run):56s} {len(rs):5d} {v['faster']:7d} {v['slower']:7d}"
                  f" {v['same']:6d} {v['undecided']:6d} {fast_s:>8} {slow_s:>8}"
                  + (f"  [{', '.join(flags)}]" if flags else ""))
            if verbose:
                for r in rs:
                    if r.stats.verdict in ("faster", "slower", "undecided"):
                        print(f"    {r.stats.verdict:9s} {_pct(r.stats.gap):>8} {_evidence(r.stats)} "
                              f"n={r.stats.n_a} {r.row.op_name} {r.row.dtype} #{r.row.row} "
                              f"{r.row.size[:90]}")
    v = defaultdict(int)
    for r in tested:
        v[r.stats.verdict] += 1
    what = "rows with a changed kernel" if gated else "rows"
    print(f"\n== {what}: {len(tested)}; faster {v['faster']}, slower {v['slower']}, "
          f"same {v['same']}, undecided {v['undecided']}")
    by_pairing = sum(1 for r in tested if r.stats.basis == "pairing")
    if by_pairing:
        print(f"   {by_pairing} of the faster/slower rows rest on the pairing alone: every round "
              "agrees, but the samples of the two sides overlap")
    rechecked = Counter(r.confirm for r in tested if r.confirm)
    if rechecked:
        few = rechecked["too_few_rounds"]
        print(f"   rechecked on fresh rounds: {sum(rechecked.values()) - few}; confirmed "
              f"{rechecked['confirmed']}, not reproduced {rechecked['not_reproduced']} (now undecided)"
              + (f"; {few} more have too few fresh rounds to tell and stay as judged" if few else ""))
    if gated:
        unknown = sum(1 for r in tested if r.gate == "unknown")
        if unknown:
            print(f"   ({unknown} of them unattributed: no rows.tsv, a kernel that failed to "
                  "recompile, or binaries a run could not tell)")
        diverged = [r for r in tested if r.gate == "diverged"]
        if diverged:
            ops = sorted({labels.get(r.row.run, r.row.run) for r in diverged})
            print(f"   {len(diverged)} of them DIVERGED: the gate cleared their kernels, but the two "
                  f"sides launched different binaries\n   (a compile option or config that is not "
                  f"in the record). Ops: {', '.join(ops[:8])}{' ...' if len(ops) > 8 else ''}")
        same_changed = sum(1 for r in tested if r.gate == "changed" and r.row.binaries == "same")
        if same_changed:
            print(f"   ({same_changed} rows the gate calls changed ran the same binaries on both "
                  "sides: is this the A/B the gate was made for?)")
        fp = [r for r in controls if r.stats.verdict in ("faster", "slower")]
        decided = [r for r in controls if r.stats.verdict != "undecided"]
        rate = f" ({len(fp) / len(decided):.2%} of the {len(decided)} decided)" if decided else ""
        print(f"== controls (identical binaries): {len(controls)}; called faster/slower: "
              f"{len(fp)}{rate}, undecided {len(controls) - len(decided)}")
        if fp:
            print("   a control only controls when both sides also ran the same tuning "
                  "config; under independently tuned records some of these are real "
                  "differences between two configs, not false positives")
        for r in fp[:10]:
            print(f"    false positive: {labels.get(r.row.run, r.row.run)} {r.row.op_name} "
                  f"{r.row.dtype} #{r.row.row} {_pct(r.stats.gap)} {_evidence(r.stats)}")


CSV_FIELDS = ["op", "run", "op_name", "dtype", "row", "size_detail", "gate", "binaries", "rounds",
              "median_a_ms", "median_b_ms", "gap", "p", "p_paired", "verdict", "basis", "flags",
              "confirm", "confirm_rounds", "confirm_gap", "confirm_p", "confirm_p_paired"]


def write_csv(path: Path, results: List[Result], labels: Dict[str, str]) -> None:
    """Every row's statistics. p and p_paired are two-sided; confirm_p and
    confirm_p_paired are one-sided over the confirm rounds, in the verdict's direction.
    basis says which test showed a faster/slower verdict (rank or pairing)."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            flags = ";".join(x for x, on in (("do_bench", r.row.nocg),
                                             ("replay_fallback", r.row.fallback),
                                             ("failed_side", r.row.failed)) if on)
            new = r.new
            w.writerow({
                "op": labels.get(r.row.run, r.row.run), "run": r.row.run,
                "op_name": r.row.op_name, "dtype": r.row.dtype, "row": r.row.row,
                "size_detail": r.row.size, "gate": r.gate, "binaries": r.row.binaries,
                "rounds": r.stats.n_a,
                "median_a_ms": f"{r.stats.median_a:.6g}", "median_b_ms": f"{r.stats.median_b:.6g}",
                "gap": f"{r.stats.gap:.6f}", "p": _p(r.stats.p), "p_paired": _p(r.stats.p_paired),
                "verdict": r.stats.verdict, "basis": r.stats.basis, "flags": flags,
                "confirm": r.confirm, "confirm_rounds": new.n_a if new else "",
                "confirm_gap": f"{new.gap:.6f}" if new else "",
                "confirm_p": _p(new.p) if new else "",
                "confirm_p_paired": _p(new.p_paired) if new else "",
            })


def rule_args(p: argparse.ArgumentParser) -> None:
    """Verdict rule flags, shared with ab_run.py."""
    g = p.add_argument_group("verdict rule (repeated-measures A/B)")
    g.add_argument("--min-rounds", type=int, default=ab_stats.MIN_ROUNDS,
                   help="samples per side before a row gets a verdict (default %(default)s)")
    g.add_argument("--threshold", type=float, default=ab_stats.THRESHOLD,
                   help="smallest |gap| that counts as a difference (default %(default)s)")
    g.add_argument("--alpha", type=float, default=ab_stats.ALPHA,
                   help="Mann-Whitney p below which a gap is real (default %(default)s)")
    g.add_argument("--alpha-paired", type=float, default=ab_stats.ALPHA_PAIRED,
                   help="sign-test p below which a gap is real when every round points the "
                        "same way; the rounds are pairs, both sides on one GPU "
                        "(default %(default)s; 0 turns the pairing off)")
    g.add_argument("--clear", type=float, default=ab_stats.CLEAR,
                   help="Mann-Whitney p at or above which a row is called same, unless the "
                        "pairing shows a gap (default %(default)s)")


def rule_of(args: argparse.Namespace) -> dict:
    return {"min_rounds": args.min_rounds, "threshold": args.threshold, "alpha": args.alpha,
            "clear": args.clear, "alpha_paired": args.alpha_paired}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", type=Path, help="ab_run.py output dir, or the A side: run dir or batch dir")
    p.add_argument("b", type=Path, nargs="?", help="B side of a single comparison")
    p.add_argument("--gate", type=Path, default=None,
                   help="ir_gate.py output dir: unchanged rows become controls")
    p.add_argument("--rounds", default=None, metavar="LO-HI",
                   help="only these rounds, e.g. 7-12 (default: all)")
    p.add_argument("--csv", type=Path, default=None, help="write every row's statistics here")
    p.add_argument("--recheck-list", type=Path, default=None, metavar="FILE",
                   help="write the ops with a weak faster/slower row (an --ops-file for "
                        "ab_run.py --add-rounds)")
    p.add_argument("--diverged-list", type=Path, default=None, metavar="FILE",
                   help="with --gate: write the ops with a row the gate cleared although the "
                        "two sides launched different binaries (an --ops-file for ab_run.py)")
    p.add_argument("--weak-gap", type=float, default=0.02,
                   help="|gap| below which a decided row is weak (default %(default)s)")
    p.add_argument("--confirm", default=None, metavar="LO-HI",
                   help="recheck rounds: judge on the other rounds, confirm on these alone")
    p.add_argument("--confirm-alpha", type=float, default=0.05,
                   help="one-sided p a recheck must reach, by the rank test or the pairing "
                        "(default %(default)s; takes 4 confirm rounds, 5 for the pairing)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="list each faster / slower / undecided row")
    p.add_argument("--min", type=float, default=0.03,
                   help="single comparison: per-row detail threshold (default %(default)s)")
    p.add_argument("--quiet", action="store_true", help="summary lines only")
    rule_args(p)
    args = p.parse_args()

    if args.b is not None:
        if args.gate or args.rounds or args.confirm or args.recheck_list or args.csv \
                or args.diverged_list:
            sys.exit("--gate, --rounds, --confirm, --recheck-list, --diverged-list and --csv "
                     "need an ab_run.py dir")
        single(args)
        return
    manifest_path = args.a / MANIFEST
    if not manifest_path.is_file():
        sys.exit(f"{args.a} has no {MANIFEST}: give an ab_run.py output dir, or two dirs to compare")
    if args.diverged_list and not args.gate:
        sys.exit("--diverged-list needs --gate")
    if args.confirm:
        if args.rounds:
            sys.exit("--confirm and --rounds do not mix")
        results = confirm(args.a, args.gate, parse_range(args.confirm), args.confirm_alpha,
                          **rule_of(args))
    else:
        results = analyze(args.a, args.gate, parse_range(args.rounds), **rule_of(args))
    if not results:
        sys.exit("no row has a sample on both sides")
    labels = op_labels(args.a)
    report(results, labels, json.loads(manifest_path.read_text()), args.verbose, args.quiet)
    if args.csv:
        write_csv(args.csv, results, labels)
        print(f"== per-row statistics: {args.csv}")
    if args.diverged_list:
        ops = sorted({labels.get(r.row.run, r.row.run) for r in results if r.gate == "diverged"})
        args.diverged_list.write_text("".join(f"{op}\n" for op in ops))
        print(f"== {len(ops)} ops with diverged rows: {args.diverged_list}")
    if args.recheck_list:
        weak = [r for r in results
                if r.gate != "unchanged" and r.confirm in ("", "too_few_rounds")
                and is_weak(r, args.weak_gap)]
        ops = sorted({labels.get(r.row.run, r.row.run) for r in weak})
        args.recheck_list.write_text("".join(f"{op}\n" for op in ops))
        print(f"== {len(weak)} weak rows in {len(ops)} ops: {args.recheck_list}")


if __name__ == "__main__":
    main()
