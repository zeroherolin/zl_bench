#!/usr/bin/env python3
"""A/B comparison of Gems latency, per result row.

Repeated-measures A/B (the output dir of ab_run.py): every row has one sample per
round and side, and gets a verdict from both samples (ab_stats.py):

  python ab_compare.py runs/ab_x [--gate runs/gate_x] [--rounds 7-12] [--csv rows.csv] [-v]

  verdict  faster / slower: |median gap| > --threshold and Mann-Whitney p < --alpha
           same: within the threshold, or no detectable shift (p >= --clear)
           undecided: fewer than --min-rounds samples, or in between (add rounds)
  gap      median(A) / median(B) - 1, > 0 means B is faster

With --gate (an ir_gate.py output dir), rows whose kernels all compiled to the same
binary on both sides are controls: they cannot differ, so any verdict other than
same/undecided among them is a false positive, and their rate is the measured
false-positive rate of the rule. Only rows with a changed (or unknown) kernel count
as results. --rounds restricts the analysis to a range of rounds.

Independent recheck of weak results, on fresh rounds:

  python ab_compare.py runs/ab_x --gate G --recheck-list recheck.txt   # ops with weak rows
  python ab_run.py ... --out runs/ab_x --ops-file recheck.txt --add-rounds 6   # say rounds 15-20
  python ab_compare.py runs/ab_x --gate G --confirm 15-20

A faster/slower row (not a control) is weak if |gap| < --weak-gap (2%), |gap| < 2x
the larger coefficient of variation of its two samples, or its table was timed by
plain do_bench. With
--confirm, every row is judged on the other rounds first; each faster/slower row with
samples in the confirm rounds is then confirmed only if those rounds alone show a gap
beyond the threshold in the same direction with a one-sided Mann-Whitney
p < --confirm-alpha (0.05). Confirmed rows report the gap over all rounds; rows that are
not reproduced become undecided.

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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import ab_stats
from batch_summary import OPERATOR_HEADER_RE, parse_run_log

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
    if not sep:
        return int(lo), int(lo)
    return int(lo) if lo else 1, int(hi) if hi else 10 ** 9


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


def op_labels(ab_dir: Path) -> Dict[str, str]:
    """<repo>/<run_name> -> repo:op@op_file, from the rounds' ops_status.csv."""
    out = {}
    for _, _, d in round_dirs(ab_dir):
        status = d / "ops_status.csv"
        if status.is_file():
            for r in csv.DictReader(status.open(newline="")):
                out[f"{r['repo']}/{r['run_name']}"] = f"{r['repo']}:{r['op']}@{r['op_file']}"
    return out


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


def load_samples(ab_dir: Path, rng: Optional[Tuple[int, int]] = None,
                 exclude: Optional[Tuple[int, int]] = None) -> List[Row]:
    """Every row with its per-round samples. A round counts for a row only if both of
    its sides measured the row."""
    per: Dict[Tuple[str, str, str, int], Dict[int, Dict[str, float]]] = defaultdict(dict)
    size: Dict[Tuple[str, str, str, int], str] = {}
    nocg, fallback = set(), set()
    for k, side, d in round_dirs(ab_dir, rng, exclude):
        for run_dir in sorted(p for p in d.glob("*/*") if (p / "run.log").is_file()):
            run = f"{run_dir.parent.name}/{run_dir.name}"
            text = (run_dir / "run.log").read_text(errors="replace")
            nocg |= {(run,) + t for t in nocg_tables(text)}
            if FALLBACK in text:
                fallback.add(run)
            for (op_name, dtype, i), (lat, sz) in load_rows(run_dir).items():
                key = (run, op_name, dtype, i)
                per[key].setdefault(k, {})[side] = lat
                size.setdefault(key, sz)
    rows = []
    for key, by_round in sorted(per.items()):
        rounds = sorted(k for k, s in by_round.items() if len(s) == 2)
        if not rounds:
            continue
        rows.append(Row(*key, size[key], [by_round[k]["a"] for k in rounds],
                        [by_round[k]["b"] for k in rounds], rounds,
                        key[:3] in nocg, key[0] in fallback))
    return rows


def load_gate(gate_dir: Path) -> Dict[Tuple[str, str, str, int], str]:
    """{(run, op_name, dtype, row): changed 1/0/?} from ir_gate.py's rows.tsv."""
    path = gate_dir / "rows.tsv"
    if not path.is_file():
        raise SystemExit(f"no {path}: run ir_gate.py on a record made with row tracking")
    out = {}
    for r in csv.DictReader(path.open(newline=""), delimiter="\t"):
        out[(r["run"], r["op_name"], r["dtype"], int(r["row"]))] = r["changed"]
    return out


def gate_flag(gate: Dict[Tuple[str, str, str, int], str], row: Row) -> str:
    """changed / unchanged / unknown for a row; a single-op record has run "" in the gate."""
    flag = gate.get((row.run, row.op_name, row.dtype, row.row))
    if flag is None:
        flag = gate.get(("", row.op_name, row.dtype, row.row))
    return {"1": "changed", "0": "unchanged"}.get(flag or "?", "unknown")


class Result(NamedTuple):
    row: Row
    stats: ab_stats.RowStats
    gate: str          # changed / unchanged / unknown, or "" without --gate
    confirm: str = ""  # --confirm: confirmed / not_reproduced / "" (not rechecked)
    new: Optional[ab_stats.RowStats] = None  # statistics of the confirm rounds alone


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
    samples' coefficient of variation, or timing by plain do_bench."""
    if r.stats.verdict not in ("faster", "slower"):
        return False
    cv = max(statistics.pstdev(x) / statistics.fmean(x) for x in (r.row.a, r.row.b))
    return abs(r.stats.gap) < weak_gap or abs(r.stats.gap) < 2 * cv or r.row.nocg


def confirm(ab_dir: Path, gate_dir: Optional[Path], rng: Tuple[int, int], confirm_alpha: float,
            **rule) -> List[Result]:
    """Judge every row on the rounds outside rng, then recheck its faster/slower verdict on
    the rounds inside rng alone (one-sided, in the original direction)."""
    main = analyze(ab_dir, gate_dir, exclude=rng, **rule)
    new = {(r.run, r.op_name, r.dtype, r.row): r for r in load_samples(ab_dir, rng)}
    out = []
    for r in main:
        n = new.get((r.row.run, r.row.op_name, r.row.dtype, r.row.row))
        if r.stats.verdict not in ("faster", "slower") or n is None or len(n.rounds) < 3:
            out.append(r)
            continue
        alternative = "greater" if r.stats.verdict == "faster" else "less"
        p = ab_stats.mann_whitney_u(n.a, n.b, alternative)[1]
        gap = statistics.median(n.a) / statistics.median(n.b) - 1
        ok = p < confirm_alpha and (gap > rule["threshold"] if r.stats.verdict == "faster"
                            else gap < -rule["threshold"])
        stats_new = ab_stats.RowStats(len(n.a), len(n.b), statistics.median(n.a),
                                      statistics.median(n.b), gap, p, "")
        if ok:
            both = r.row._replace(a=r.row.a + n.a, b=r.row.b + n.b, rounds=r.row.rounds + n.rounds)
            all_stats = ab_stats.compare(both.a, both.b, **rule)._replace(verdict=r.stats.verdict)
            out.append(Result(both, all_stats, r.gate, "confirmed", stats_new))
        else:
            out.append(Result(r.row, r.stats._replace(verdict="undecided"), r.gate,
                              "not_reproduced", stats_new))
    return out


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


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
            fast_s = _pct(fast) if fast is not None else "-"
            slow_s = _pct(slow) if slow is not None else "-"
            print(f"{labels.get(run, run):56s} {len(rs):5d} {v['faster']:7d} {v['slower']:7d}"
                  f" {v['same']:6d} {v['undecided']:6d} {fast_s:>8} {slow_s:>8}"
                  + (f"  [{', '.join(flags)}]" if flags else ""))
            if verbose:
                for r in rs:
                    if r.stats.verdict in ("faster", "slower", "undecided"):
                        print(f"    {r.stats.verdict:9s} {_pct(r.stats.gap):>8} p={r.stats.p:.2g} "
                              f"n={r.stats.n_a} {r.row.op_name} {r.row.dtype} #{r.row.row} "
                              f"{r.row.size[:90]}")
    v = defaultdict(int)
    for r in tested:
        v[r.stats.verdict] += 1
    what = "rows with a changed kernel" if gated else "rows"
    print(f"\n== {what}: {len(tested)}; faster {v['faster']}, slower {v['slower']}, "
          f"same {v['same']}, undecided {v['undecided']}")
    rechecked = [r for r in tested if r.confirm]
    if rechecked:
        kept = sum(r.confirm == "confirmed" for r in rechecked)
        print(f"   rechecked on fresh rounds: {len(rechecked)}; confirmed {kept}, not reproduced "
              f"{len(rechecked) - kept} (now undecided)")
    if gated:
        unknown = sum(1 for r in tested if r.gate == "unknown")
        if unknown:
            print(f"   ({unknown} of them unattributed: no rows.tsv or a kernel failed to recompile)")
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
                  f"{r.row.dtype} #{r.row.row} {_pct(r.stats.gap)} p={r.stats.p:.2g}")


def write_csv(path: Path, results: List[Result], labels: Dict[str, str]) -> None:
    fields = ["op", "run", "op_name", "dtype", "row", "size_detail", "gate", "rounds",
              "median_a_ms", "median_b_ms", "gap", "p", "p_paired", "verdict", "flags", "confirm",
              "confirm_rounds", "confirm_gap", "confirm_p"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            flags = ";".join(x for x, on in (("do_bench", r.row.nocg),
                                             ("replay_fallback", r.row.fallback)) if on)
            w.writerow([labels.get(r.row.run, r.row.run), r.row.run, r.row.op_name, r.row.dtype,
                        r.row.row, r.row.size, r.gate, r.stats.n_a, f"{r.stats.median_a:.6g}",
                        f"{r.stats.median_b:.6g}", f"{r.stats.gap:.6f}",
                        "" if math.isnan(r.stats.p) else f"{r.stats.p:.4g}",
                    "" if math.isnan(r.stats.p_paired) else f"{r.stats.p_paired:.4g}",
                    r.stats.verdict, flags,
                        r.confirm, r.new.n_a if r.new else "",
                        f"{r.new.gap:.6f}" if r.new else "", f"{r.new.p:.4g}" if r.new else ""])


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
                  help="sign-test level over the paired rounds: a row the unpaired test "
                   "leaves undecided is decided when every round agrees and the gap "
                   "clears --threshold (default %(default)s)")
    g.add_argument("--clear", type=float, default=ab_stats.CLEAR,
                   help="p at or above which a row is called same (default %(default)s)")


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
    p.add_argument("--weak-gap", type=float, default=0.02,
                   help="|gap| below which a decided row is weak (default %(default)s)")
    p.add_argument("--confirm", default=None, metavar="LO-HI",
                   help="recheck rounds: judge on the other rounds, confirm on these alone")
    p.add_argument("--confirm-alpha", type=float, default=0.05,
                   help="one-sided p a recheck must reach (default %(default)s)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="list each faster / slower / undecided row")
    p.add_argument("--min", type=float, default=0.03,
                   help="single comparison: per-row detail threshold (default %(default)s)")
    p.add_argument("--quiet", action="store_true", help="summary lines only")
    rule_args(p)
    args = p.parse_args()

    if args.b is not None:
        if args.gate or args.rounds or args.confirm or args.recheck_list or args.csv:
            sys.exit("--gate, --rounds, --confirm, --recheck-list and --csv need an ab_run.py dir")
        single(args)
        return
    manifest_path = args.a / MANIFEST
    if not manifest_path.is_file():
        sys.exit(f"{args.a} has no {MANIFEST}: give an ab_run.py output dir, or two dirs to compare")
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
    if args.recheck_list:
        weak = [r for r in results
                if r.gate != "unchanged" and not r.confirm and is_weak(r, args.weak_gap)]
        ops = sorted({labels.get(r.row.run, r.row.run) for r in weak})
        args.recheck_list.write_text("".join(f"{op}\n" for op in ops))
        print(f"== {len(weak)} weak rows in {len(ops)} ops: {args.recheck_list}")


if __name__ == "__main__":
    main()
