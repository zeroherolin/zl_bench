#!/usr/bin/env python3
"""Collect the results of a run_batch.py batch dir into summary.csv.

The source is each op's run.log (ANSI-stripped by run_pytest.sh):
- each table title `Operator: <op_name> (dtype=..., mode=..., level=...)`,
  in both the _pretty_report_plugin and the upstream format;
- each SUCCESS/FAILED row: status, then the first three numeric columns are
  torch_lat / gems_lat / speedup (extra columns such as tflops/gbps go to
  extra_metrics), the rest is size_detail;
- merged with ops_status.csv from run_batch.py (per-op status and notes).

Standalone use: python batch_summary.py runs/batch_xxx [-o summary.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Both title formats:
#   pretty:   Operator: softmax  (dtype=torch.float16, mode=kernel, level=core)
#   upstream: Operator: softmax  Performance Test (dtype=torch.float16, mode=kernel,level=core)
OPERATOR_HEADER_RE = re.compile(
    r"^Operator:\s+(?P<name>\S+)\s+(?:Performance Test\s*)?"
    r"\(dtype=(?P<dtype>[^,]+),\s*mode=(?P<mode>[^,]+),\s*level=(?P<level>[^)]+)\)"
)
ROW_RE = re.compile(r"^(?P<status>SUCCESS|FAILED)\s+(?P<rest>\S.*)$")
NUM_TOKEN_RE = re.compile(
    r"^(?:N/A|nan|-?inf"
    r"|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
PYTEST_SKIPPED_RE = re.compile(r"(?P<n>\d+)\s+(?:skipped|xfailed)\b")
PYTEST_SUMMARY_RE = re.compile(r"^=+ (.*\bin [0-9.]+s\b.*) =+$", re.M)
# Markers the plugins print into run.log, as (note for ops_status.csv, pattern). A note
# with a count reads name=<n>; the others say that the marker occurred at all.
COUNTED_MARKERS = (
    ("no_cudagraph_rows", re.compile(r"BENCHMARK_DIRECT_NO_CUDAGRAPH\b")),
    ("replay_fallback", re.compile(r"AUTOTUNE_REPLAY_FALLBACK\b")),
    ("cudagraph_oom_retry", re.compile(r"CUDAGRAPH_OOM_RETRY\b")),
)
FLAG_MARKERS = (
    ("shape_keys_unused", re.compile(r"SHAPE_KEYS_UNUSED\b")),
    ("ir_dump_failed", re.compile(r"\b(?:IR_DUMP_FAILED|ROWS_TSV_FAILED)\b")),
)

SUMMARY_FIELDS = [
    "repo", "op", "fg_op_name", "dtype", "case_id", "row_status",
    "torch_lat", "gems_lat", "speedup", "extra_metrics", "size_detail",
    "op_status", "notes",
]


def parse_run_log(text: str) -> List[Dict[str, str]]:
    """All result-table rows of a run.log, across the per-dtype tables."""
    rows: List[Dict[str, str]] = []
    cur_op, cur_dtype = "", ""
    for line in text.splitlines():
        stripped = line.strip()
        header = OPERATOR_HEADER_RE.match(stripped)
        if header:
            cur_op = header.group("name")
            cur_dtype = header.group("dtype").strip()
            continue
        m = ROW_RE.match(stripped)
        if not m:
            continue
        tokens = m.group("rest").split()
        metrics: List[str] = []
        while tokens and len(metrics) < 7 and NUM_TOKEN_RE.match(tokens[0]):
            metrics.append(tokens.pop(0))
        # The first three numeric columns are torch_lat, gems_lat, speedup (upstream column order).
        while len(metrics) < 3:
            metrics.append("")
        rows.append({
            "fg_op_name": cur_op,
            "dtype": cur_dtype,
            "row_status": m.group("status"),
            "torch_lat": metrics[0],
            "gems_lat": metrics[1],
            "speedup": metrics[2],
            "extra_metrics": " ".join(metrics[3:]),
            "size_detail": " ".join(tokens),
        })
    return rows


def pytest_skipped_count(text: str) -> int:
    """Skipped + xfailed count: both mean no measurement, for the SKIP verdict.

    Read from pytest's final summary line ("== 1 skipped, 2 warnings in 0.2s =="). The
    line is not at the very end of run.log: the IR dump prints after it, one line per
    shape, so a fixed window over the tail of the log would miss it on a long shape set
    and turn a SKIP into a FAIL. Without such a line (a killed run) the
    tail is searched.
    """
    summaries = PYTEST_SUMMARY_RE.findall(text)
    scope = summaries[-1] if summaries else text[-4000:]
    return sum(int(m.group("n")) for m in PYTEST_SKIPPED_RE.finditer(scope))


def marker_notes(text: str) -> str:
    """The markers of a run.log for the notes column: how many measurements fell back to
    plain do_bench or to live tuning, and whether the run measured other shapes than the
    shape file's or lost its IR dump. A reader of ops_status.csv sees them without
    opening the log."""
    notes = []
    for name, pattern in COUNTED_MARKERS:
        n = len(pattern.findall(text))
        if n:
            notes.append(f"{name}={n}")
    notes += [name for name, pattern in FLAG_MARKERS if pattern.search(text)]
    return ";".join(notes)


def classify(rc: int, rows: List[Dict[str, str]], log_text: str) -> str:
    """Per-op status: FAIL / SKIP / PASS (run_batch.py decides TIMEOUT itself)."""
    if rc != 0:
        return "FAIL"
    if not rows:
        return "SKIP" if pytest_skipped_count(log_text) else "FAIL"
    if any(r["row_status"] != "SUCCESS" for r in rows):
        return "FAIL"
    return "PASS"


def read_table(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    """The rows of a csv (or tsv) file with a header line; none if there is no such file."""
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _load_ops_status(batch_dir: Path) -> Dict[str, Dict[str, str]]:
    return {f"{r['repo']}/{r['run_name']}": r for r in read_table(batch_dir / "ops_status.csv")}


def summarize(batch_dir: Path, out_path: Optional[Path] = None) -> Path:
    batch_dir = batch_dir.resolve()
    out_path = out_path or (batch_dir / "summary.csv")
    status_by_dir = _load_ops_status(batch_dir)

    all_rows: List[Dict[str, str]] = []
    for repo_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
        repo = repo_dir.name
        for op_dir in sorted(p for p in repo_dir.iterdir() if p.is_dir()):
            log = op_dir / "run.log"
            if not log.is_file():
                continue
            text = log.read_text(errors="replace")
            status_row = status_by_dir.get(f"{repo}/{op_dir.name}", {})
            op = status_row.get("op") or op_dir.name
            table_rows = parse_run_log(text)
            if not table_rows:
                all_rows.append({
                    "repo": repo, "op": op, "fg_op_name": "", "dtype": "",
                    "case_id": "", "row_status": "", "torch_lat": "",
                    "gems_lat": "", "speedup": "", "extra_metrics": "",
                    "size_detail": "",
                    "op_status": status_row.get("status", ""),
                    "notes": status_row.get("notes", ""),
                })
                continue
            case_counter: Dict[Tuple[str, str], int] = {}
            for row in table_rows:
                key = (row["fg_op_name"], row["dtype"])
                idx = case_counter.get(key, 0)
                case_counter[key] = idx + 1
                all_rows.append({
                    "repo": repo, "op": op, "case_id": f"case_{idx:03d}",
                    "op_status": status_row.get("status", ""),
                    "notes": status_row.get("notes", ""),
                    **row,
                })

    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(tmp, out_path)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("batch_dir", type=Path, help="run_batch.py output dir")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output CSV (default <batch_dir>/summary.csv)")
    args = p.parse_args()
    if not args.batch_dir.is_dir():
        raise SystemExit(f"no such dir: {args.batch_dir}")
    out = summarize(args.batch_dir, args.output)
    rows = read_table(out)
    n_ok = sum(1 for r in rows if r["row_status"] == "SUCCESS")
    n_bad = sum(1 for r in rows if r["row_status"] == "FAILED")
    print(f"[summary] {out}: {len(rows)} rows (SUCCESS={n_ok} FAILED={n_bad})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
