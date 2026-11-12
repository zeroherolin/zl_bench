#!/usr/bin/env python3
"""Compile-diff gate: recompile the recorded TTIR with two compiler builds offline and
report which kernels, result rows and ops the change can affect.

Input is the ttir/<cache hash>/{<kernel>.ttir, <kernel>.json} that _ir_dump_plugin
writes into each run dir (only the variants the measurement actually used) and the
run's rows.tsv (which kernels each result row ran). No GPU is needed. A compiler
change usually touches few kernels, so pass this gate first and A/B only what it
selects, one to two orders of magnitude faster than a blind full run.

  # two builds (each value is a FlagTree python/ dir, used as PYTHONPATH)
  python ir_gate.py runs/batch_x --base /path/old/python --new /path/new/python
  # one build, two compile switches: --base-env / --new-env add env variables
  python ir_gate.py runs/batch_x --base $PWD/python --new $PWD/python \\
      --base-env FLAGTREE_XXX=0 --new-env FLAGTREE_XXX=1

Each variant is compiled down to cubin on both sides (--stage cubin, the default).
A variant is unchanged if, and only if, its cubin and the launch metadata Triton
derives (shared memory, warps, ...) are identical: then the measured kernel is the
same binary, whatever the stages before it look like. TTGIR alone decides neither
way: a change in a later stage leaves the TTGIR intact but changes the code ptxas
emits, and a reordered TTGIR may end in the same code. Line info is turned
off on both sides (TRITON_DISABLE_LINE_INFO=1, unless an --*-env sets it), so a
moved source location alone does not count as a change; it does not change the code.

--stage ttgir stops after make_ttgir (faster, for a first look); without --base the
TTGIR recorded at run time is the baseline, so only --stage ttgir is possible and
the batch must have been recorded with the baseline build.

What the gate clears is the recorded variants, compiled with the recorded options. An
A/B side may launch something else: a replay fallback tunes a config on the spot, and
an operator library may set a compile option only where the compiler has it (per
kernel, or as one more dimension of a tuning space), so a record made with a build
that lacks the option does not hold the variants the other build launches. The gate
warns when a build has compile options the record, or the other build, does not have.
Every run records the binaries its rows launched (rows.tsv), and ab_compare.py --gate
holds the gate against them: a cleared row whose two sides ran different binaries is
reported as diverged and counted as a result, not as a control.

Outputs in <out>/:
  variants.tsv     every variant: changed (1 / 0 / ? = a side failed to compile),
                   first stage that differs (in parentheses when the variant still
                   ends in the same cubin), TTGIR verdict (same / order / changed),
                   and the TTGIR ops whose count changed (e.g. "ttg.local_load:+2")
  report.tsv       the changed and unknown variants only
  rows.tsv         every result row of runs that have rows.tsv: changed if any kernel
                   it ran changed (input for ab_compare.py --gate)
  changed_ops.txt  ops with a changed or unknown row (or variant, for runs without
                   rows.tsv), for run_batch.py / ab_run.py --ops-file
  all_ops.txt      every recorded op, for the one-round replay that finds diverged rows
  base/ new/       the recompiled ttgir per <cache hash>/<kernel>.ttgir, for diffing
  errors.log       compile failures
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from _binary_id import LAUNCH_META
from batch_summary import read_table
from run_batch import parse_env

LOG = "[ir-gate]"
STAGES = ("ttgir", "llir", "ptx", "cubin")


def option_drift(base: List[Optional[dict]], new: List[Optional[dict]]) -> List[str]:
    """Compile options that one side has and the record, or the other side, has not.

    Each list holds the digests of one side's variants; a digest names the compile
    options of its build ("options") and those of them the recorded variant was
    compiled without ("unrecorded_options"). The gate compiles the recorded variants
    with the recorded options. An operator library that sets a new option where the
    compiler has it (per kernel, or as one more dimension of a tuning space) launches
    other variants on that side, and those were never recorded: the gate cannot see
    them, and an op it clears may still differ.
    """
    sides = [{o for d in digests if d for o in d.get("options", ())} for digests in (base, new)]
    drift = {o for digests in (base, new) for d in digests if d
                for o in d.get("unrecorded_options", ())}
    if all(sides):
        drift |= sides[0] ^ sides[1]
    return sorted(drift)


# ---------------- collection ----------------

class Variant(NamedTuple):
    ttir: Path                # ttir/<cache hash>/<kernel>.ttir, next to <kernel>.json
    label: str                # first op that used it (variants are shared): repo:op@op_file
    recorded: Optional[Path]  # ttgir compiled at record time (ttgir/<shape>/<kernel>/<file>)

    @property
    def key(self) -> Path:
        return Path(self.ttir.parent.name) / self.ttir.with_suffix(".ttgir").name


class RunDir(NamedTuple):
    path: Path
    label: str  # repo:op@op_file, the run_batch.py op spec (dir name for a single-op run)
    run: str    # <repo>/<run_name> inside a batch, "" for a single-op run dir


def run_dirs(roots: List[Path]) -> List[RunDir]:
    """Batch dirs expand to <repo>/<run_name>/; a single-op run dir is taken as is."""
    out = []
    for root in roots:
        if (root / "ttir").is_dir():
            out.append(RunDir(root, single_op_label(root), ""))
            continue
        spec = {}
        # changed_ops.txt must feed back into run_batch.py: label as repo:op@op_file
        for r in read_table(root / "ops_status.csv"):
            spec[(r["repo"], r["run_name"])] = f"{r['repo']}:{r['op']}@{r['op_file']}"
        for d in sorted(p for p in root.glob("*/*") if (p / "ttir").is_dir()):
            key = (d.parent.name, d.name)
            out.append(RunDir(d, spec.get(key, f"{key[0]}:{key[1]}"), f"{key[0]}/{key[1]}"))
    return out


_SPEC_RE = re.compile(r"^>>> spec=(\S+)$", re.M)


def single_op_label(run_dir: Path) -> str:
    """The op of a single-op run dir as run_batch.py / ab_run.py select it
    (repo:op@op_file), from the spec line run_pytest.sh logs; the dir name for a run
    whose log has no such line, which is not a selectable op."""
    log = run_dir / "run.log"
    m = _SPEC_RE.search(log.read_text(errors="replace")) if log.is_file() else None
    return m.group(1) if m else run_dir.name


def run_hashes(run_dir: Path) -> Set[str]:
    """Cache hashes of the variants a run used (the ones it dumped)."""
    return {p.parent.name for p in (run_dir / "ttir").glob("*/*.ttir")}


def collect(roots: List[Path]) -> List[Variant]:
    """Every recompilable variant, deduplicated by cache hash (ops often share kernels)."""
    seen: Dict[str, Variant] = {}
    for rd in run_dirs(roots):
        recorded: Dict[str, Path] = {}
        for row in read_table(rd.path / "ttgir" / "index.tsv", "\t"):
            if int(row["launches"] or 0) > 0:
                recorded.setdefault(row["cache_hash"], rd.path / "ttgir" / row["shape"]
                                    / row["kernel"] / row["file"])
        for ttir in sorted((rd.path / "ttir").glob("*/*.ttir")):
            h = ttir.parent.name
            if h not in seen and ttir.with_suffix(".json").is_file():
                seen[h] = Variant(ttir, rd.label, recorded.get(h))
    return list(seen.values())


def read_rows(run_dir: Path) -> Optional[List[Dict[str, str]]]:
    """The run's rows.tsv (written by _ir_dump_plugin), None if it has none."""
    path = run_dir / "rows.tsv"
    return read_table(path, "\t") if path.is_file() else None


def judge_rows(rows: List[Dict[str, str]], status: Dict[str, Tuple[str, str, str]],
               used: Set[str]) -> List[Tuple[Dict[str, str], str, str]]:
    """(row, changed 1/0/?, the kernels that are not cleared) for the rows of one run.

    status is compare()'s result per cache hash, used the hashes of the variants the run
    dumped. A row is changed if a kernel it ran changed, unknown if one could not be
    compiled or the row was not tracked ("?"), else unchanged.

    A row without kernels is unchanged only if the rows account for every variant the
    run used: it then launches no Triton kernel (small inputs may not). Otherwise its
    kernels may be the ones no row claims, and it is unknown. A line cut off in a
    damaged file is left out; ab_compare.py finds no verdict for its row.
    """
    named = {k for r in rows for k in (r.get("kernels") or "").split(",") if k and k != "?"}
    stray = bool(used - named)
    out = []
    for r in rows:
        if None in (r.get("op_name"), r.get("dtype"), r.get("row"), r.get("shape")):
            continue
        kernels = r.get("kernels")
        ks = [k for k in kernels.split(",") if k] if kernels not in (None, "?") else None
        if ks is None or (not ks and stray):
            out.append((r, "?", ""))
            continue
        flags = {k: status.get(k, ("?",))[0] for k in ks}
        flag = "1" if "1" in flags.values() else "?" if "?" in flags.values() else "0"
        out.append((r, flag, ",".join(k for k, f in flags.items() if f != "0")))
    return out


# ---------------- comparison helpers ----------------

_LOC_RE = re.compile(r" loc\((?:[^()]|\([^()]*\))*\)")
_SSA_RE = re.compile(r"%[\w#.:-]+")
# The op of an MLIR line: optional results ("%a, %b:2 = "), then the dialect.op name.
_OP_RE = re.compile(r"^\s*(?:%[\w#.:-]+(?:\s*,\s*%[\w#.:-]+)*\s*=\s*)?\"?([a-z_][\w]*\.[\w.]+)")


def _normalize(text: str, exact: bool) -> str:
    """TTGIR without loc info; with exact=False also without SSA names and line order,
    so only the multiset of ops/types/layouts is left (the "order" verdict)."""
    lines = [_LOC_RE.sub("", ln) for ln in text.splitlines() if not ln.startswith("#loc")]
    if exact:
        return "\n".join(lines)
    return "\n".join(sorted(_SSA_RE.sub("%", ln).strip() for ln in lines))


def _digest(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()[:16]


def op_counts(text: str) -> Dict[str, int]:
    """Number of each TTGIR op (dialect.op) in a module."""
    return dict(Counter(m.group(1) for m in map(_OP_RE.match, text.splitlines()) if m))


def op_diff(base: Optional[dict], new: Optional[dict]) -> str:
    """Every op whose count changed, largest change first: "ttg.local_load:+2 ..."."""
    if not base or not new:
        return ""
    a, b = base.get("ops", {}), new.get("ops", {})
    d = {op: b.get(op, 0) - a.get(op, 0) for op in set(a) | set(b)}
    d = sorted(((op, n) for op, n in d.items() if n), key=lambda x: (-abs(x[1]), x[0]))
    return " ".join(f"{op}:{n:+d}" for op, n in d)


def ttgir_digests(text: str) -> dict:
    return {"ttgir": _digest(_normalize(text, True)), "ttgir_norm": _digest(_normalize(text, False)),
            "ops": op_counts(text)}


def compare(base: Optional[dict], new: Optional[dict], stage: str) -> Tuple[str, str, str]:
    """(changed 1/0/?, first differing stage, TTGIR verdict) of one variant.

    Compiled down to the cubin, a variant is changed when the cubin or the launch
    metadata differ, and only then: an earlier stage may differ (a reordering, renamed
    values) and still end in the same binary, which measures the same. The first
    differing stage is reported either way, in parentheses when the binary is the
    same. With --stage ttgir there is no binary to go by, and any TTGIR difference is
    a change.
    """
    if not base or not new:
        return "?", "error", "?"
    if base["ttgir"] == new["ttgir"]:
        ttgir = "same"
    elif base["ttgir_norm"] == new["ttgir_norm"]:
        ttgir = "order"
    else:
        ttgir = "changed"
    stages = STAGES[:STAGES.index(stage) + 1]
    first = next((s for s in stages if base.get(s) != new.get(s)), "")
    if stage != "cubin":
        return ("1" if first else "0"), first or "-", ttgir
    if base.get("cubin") != new.get("cubin"):
        return "1", first, ttgir
    if base.get("meta") != new.get("meta"):
        return "1", first or "meta", ttgir
    return "0", f"({first})" if first else "-", ttgir


# ---------------- compilation (in a subprocess; PYTHONPATH picks the compiler) ----------------

def _init_worker():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)  # MLIR diagnostics flood stderr; failures reach errors.log as tracebacks


def _compile_one(ttir_path: str, dest: str, stage: str) -> Optional[str]:
    """Compile one variant up to `stage`; write <dest>.ttgir and <dest>.digest.json."""
    try:
        from triton._C.libtriton import ir
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import IRSource, make_backend

        ttir = Path(ttir_path)
        meta = json.loads(ttir.with_suffix(".json").read_text())
        t = meta.get("target") or {}
        target = GPUTarget(t.get("backend", "cuda"), t.get("arch", 90), t.get("warp_size", 32))
        backend = make_backend(target)
        fields = backend.parse_options({}).__dict__.keys()
        opts = {k: tuple(v) if isinstance(v, list) else v
                for k, v in meta.items() if k in fields}
        options = backend.parse_options(opts)
        context = ir.context()
        src = IRSource(str(ttir), context, backend)
        stages: dict = {}
        backend.add_stages(stages, options, src.language)
        module = backend.make_ir(src, options, backend.get_codegen_implementation(options),
                                 backend.get_module_map(), context)
        md = {"hash": meta.get("hash", "x"), "target": target, **options.__dict__}
        out: dict = {}
        for name in STAGES[:STAGES.index(stage) + 1]:
            module = stages[name](module, md)
            if name == "ttgir":
                text = str(module)
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest + ".ttgir").write_text(text)
                out.update(ttgir_digests(text))
            else:
                out[name] = _digest(module)
        out["meta"] = {k: md[k] for k in LAUNCH_META if k in md}
        out["options"] = sorted(fields)
        out["unrecorded_options"] = sorted(set(fields) - set(meta))
        Path(dest + ".digest.json").write_text(json.dumps(out, default=str))
        return None
    except Exception:
        return traceback.format_exc(limit=4)


def _worker_main(list_file: str, jobs: int, stage: str) -> None:
    """Subprocess entry: each list_file line is "<ttir>\\t<dest prefix>"."""
    pairs = [ln.split("\t") for ln in Path(list_file).read_text().splitlines() if ln]
    errors = []
    with cf.ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
        futs = {ex.submit(_compile_one, s, d, stage): s for s, d in pairs}
        for f in cf.as_completed(futs):
            err = f.result()
            if err:
                errors.append(f"== {futs[f]}\n{err}")
    print(json.dumps({"compiled": len(pairs) - len(errors), "errors": errors}))


def _dest(out: Path, side: str, v: Variant) -> Path:
    return out / side / v.ttir.parent.name / v.ttir.stem


def compile_side(name: str, items: List[Variant], out: Path, pythonpath: str,
                 extra_env: Dict[str, str], jobs: int, stage: str) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [pythonpath, env.get("PYTHONPATH")]))
    env.setdefault("TRITON_DISABLE_LINE_INFO", "1")
    env.update(extra_env)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for v in items:
            f.write(f"{v.ttir}\t{_dest(out, name, v)}\n")
        list_file = f.name
    proc = subprocess.run([sys.executable, __file__, "_worker", list_file, str(jobs), stage],
                          env=env, text=True, capture_output=True)
    os.unlink(list_file)
    if proc.returncode != 0:
        raise SystemExit(f"{LOG} {name} compile process failed:\n{proc.stderr[-2000:]}")
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    if res["errors"]:
        with (out / "errors.log").open("a") as fh:
            fh.write(f"######## {name}\n" + "\n".join(res["errors"]) + "\n")
    print(f"{LOG} {name}: compiled {res['compiled']}/{len(items)} "
          f"({len(res['errors'])} failed)", flush=True)
    return len(res["errors"])


def _load_digest(path: Path) -> Optional[dict]:
    p = path.with_name(path.name + ".digest.json")
    return json.loads(p.read_text()) if p.is_file() else None


# ---------------- main ----------------

def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        _worker_main(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        return
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="batch dirs or single-op run dirs")
    p.add_argument("--new", required=True,
                   help="PYTHONPATH of the new compiler (a FlagTree python/ dir)")
    p.add_argument("--base", default=None,
                   help="PYTHONPATH of the baseline compiler; without it the recorded ttgir is "
                        "the baseline (--stage ttgir only)")
    p.add_argument("--new-env", action="append", default=[], metavar="K=V")
    p.add_argument("--base-env", action="append", default=[], metavar="K=V")
    p.add_argument("--stage", choices=("cubin", "ttgir"), default="cubin",
                   help="last stage to compile and compare (default %(default)s)")
    p.add_argument("--out", type=Path, default=None, help="output dir (default runs/ir_gate_<timestamp>)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 16)
    args = p.parse_args()

    stage = args.stage
    if not args.base and stage != "ttgir":
        print(f"{LOG} no --base: comparing against the recorded ttgir (--stage ttgir); "
              "changes below TTGIR are not visible", flush=True)
        stage = "ttgir"
    runs = run_dirs(args.runs)
    items = collect(args.runs)
    if not items:
        raise SystemExit(f"{LOG} nothing to recompile: expected ttir/<cache hash>/ in the run dirs")
    out = (args.out or Path(__file__).resolve().parent / "runs" /
           f"ir_gate_{datetime.now():%Y%m%d-%H%M%S}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "errors.log").unlink(missing_ok=True)
    print(f"{LOG} {len(items)} variants of {len(runs)} runs -> {out} (stage {stage})", flush=True)

    compile_side("new", items, out, args.new, parse_env(args.new_env), args.jobs, stage)
    if args.base:
        compile_side("base", items, out, args.base, parse_env(args.base_env), args.jobs, stage)

    # ---- per variant
    status: Dict[str, Tuple[str, str, str]] = {}
    lines = []
    digests: Dict[str, List[Optional[dict]]] = {"base": [], "new": []}
    for v in items:
        new = _load_digest(_dest(out, "new", v))
        digests["new"].append(new)
        if args.base:
            base = _load_digest(_dest(out, "base", v))
            digests["base"].append(base)
        else:
            base = ttgir_digests(v.recorded.read_text()) if v.recorded and v.recorded.is_file() else None
        changed, first, ttgir = compare(base, new, stage)
        h = v.ttir.parent.name
        status[h] = (changed, first, ttgir)
        lines.append([v.label, h, v.ttir.stem, changed, first, ttgir, op_diff(base, new)])
    head = "op\tcache_hash\tkernel\tchanged\tfirst_diff\tttgir\top_count_diff\n"
    lines.sort()
    (out / "variants.tsv").write_text(head + "".join("\t".join(r) + "\n" for r in lines))
    (out / "report.tsv").write_text(head + "".join("\t".join(r) + "\n" for r in lines if r[3] != "0"))

    # ---- per row and per op
    changed_ops, row_lines = [], []
    row_counts: Counter = Counter()
    no_rows = 0
    for rd in runs:
        rows = read_rows(rd.path)
        if rows is None:
            no_rows += 1
            flags = {status.get(h, ("?",))[0] for h in run_hashes(rd.path)}
            if flags & {"1", "?"}:
                changed_ops.append(rd.label)
            continue
        op_changed = False
        for r, flag, which in judge_rows(rows, status, run_hashes(rd.path)):
            row_counts[flag] += 1
            op_changed |= flag != "0"
            row_lines.append([rd.run, rd.label, r["op_name"], r["dtype"], r["row"], r["shape"],
                              flag, which])
        if op_changed:
            changed_ops.append(rd.label)
    (out / "rows.tsv").write_text(
        "run\top\top_name\tdtype\trow\tshape\tchanged\tchanged_kernels\n"
        + "".join("\t".join(r) + "\n" for r in row_lines))
    (out / "changed_ops.txt").write_text("".join(f"{label}\n" for label in sorted(set(changed_ops))))
    (out / "all_ops.txt").write_text("".join(f"{label}\n" for label in sorted({rd.label for rd in runs})))

    # ---- summary
    flags = Counter(s[0] for s in status.values())
    firsts = Counter(s[1] for s in status.values() if s[0] == "1")
    hidden = sum(1 for s in status.values() if s[0] == "1" and s[2] == "same")
    order = sum(1 for s in status.values() if s[0] == "1" and s[2] == "order")
    print(f"{LOG} variants: unchanged {flags['0']}, changed {flags['1']}, unknown {flags['?']} "
          f"(compile failure, see errors.log)")
    same_binary = sum(1 for s in status.values() if s[0] == "0" and s[1] != "-")
    if same_binary:
        print(f"{LOG}   {same_binary} of the unchanged differ in an earlier stage and end in the "
              "same cubin (first_diff in parentheses in variants.tsv)")
    if flags["1"]:
        print(f"{LOG}   first difference at: "
              + ", ".join(f"{k} {n}" for k, n in sorted(firsts.items(), key=lambda kv: -kv[1])))
        if stage == "cubin":
            print(f"{LOG}   changed with identical TTGIR {hidden}, with only reordered TTGIR {order}")
    if row_lines:
        print(f"{LOG} rows: changed {row_counts['1']}, unchanged {row_counts['0']}, "
              f"unknown {row_counts['?']} (of {len(row_lines)})")
    if no_rows:
        print(f"{LOG} {no_rows} runs have no rows.tsv (recorded before row tracking): judged per op")
    print(f"{LOG} {len(set(changed_ops))} ops to A/B: {out / 'changed_ops.txt'}")
    drift = option_drift(digests["base"], digests["new"])
    if drift:
        print(f"{LOG} WARNING: compile options the record, or one of the builds, does not have: "
              f"{', '.join(drift)}.\n"
              f"{LOG}   An operator library that sets them where the compiler has them launches "
              "variants that were\n"
              f"{LOG}   never recorded, so an op cleared here may still differ. Replay every "
              "recorded op once on both\n"
              f"{LOG}   sides and let ab_compare.py compare the binaries they ran:\n"
              f"{LOG}     python ab_run.py --record <record> --ops-file {out / 'all_ops.txt'} "
              f"--rounds 1 --gate {out} <sides> --out <ab>\n"
              f"{LOG}     python ab_compare.py <ab> --gate {out} --diverged-list "
              f"{out / 'diverged_ops.txt'}\n"
              f"{LOG}   then A/B the ops of changed_ops.txt and diverged_ops.txt.", flush=True)


if __name__ == "__main__":
    main()
