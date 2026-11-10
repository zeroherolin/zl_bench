#!/usr/bin/env python3
"""Rebuild the batch op inventory and refresh the shape sets against upstream.

Statically scans test_*.py under each FlagGems repo's benchmark/, including
test_-prefixed subdirs such as test_FLA/ (op_file keeps the relative path minus
prefix and suffix, so run_pytest.sh's `benchmark/test_${OP_FILE}.py` still
resolves; paths without the test_ prefix, e.g. models_benchmark/, are out of
scope):

- one inventory row per module-level test_* function: op (function name minus
  test_, run_pytest.sh's OP) and op_file (file name minus test_ and .py,
  run_pytest.sh's OP_FILE);
- marked_skip: the function carries an unconditional pytest.mark.skip, so it
  always SKIPs and the batch drops it by default;
- op_names: the op_name string constants (op_name= keyword arguments and the
  first positional string of super().__init__ in *Benchmark classes), used to
  validate shape yaml keys;
- with --verify-collect, a pytest --collect-only run confirms each function is
  collectable (seen by the AST != collected by pytest: import failures and
  conditional definitions differ).

Outputs:
  ops/inventory_<label>.csv        # repo,op,op_file,file,marked_skip,collected,op_names
  --refresh-shapes rewrites that shape set in place (see refresh_shapes); keys
  matching no upstream op_name move to <name>.unmatched.yaml next to it

Examples (scanning the two repos in REPOS by default):
  python gen_inventory.py --verify-collect
  # single-shape set: --one-shape keeps one shape for keys taken from core_shapes
  python gen_inventory.py --refresh-shapes shape/all_ops_single.yaml --one-shape
  python gen_inventory.py --refresh-shapes shape/all_ops_multi.yaml
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

ZL_BENCH = Path(__file__).resolve().parent
OPS_DIR = ZL_BENCH / "ops"

# Repo label -> checkout. The label is the inventory's repo column and
# run_batch.py's [repo:]op prefix. Defaults are the checkouts next to this repo.
REPOS = {
    "main": Path(os.environ.get("FLAGGEMS_MAIN_DIR") or ZL_BENCH.parent / "FlagGems").resolve(),
    "vllm": Path(os.environ.get("FLAGGEMS_VLLM_DIR") or ZL_BENCH.parent / "FlagGems-vllm").resolve(),
}

INVENTORY_FIELDS = ["repo", "op", "op_file", "file", "marked_skip", "collected", "op_names"]

# Files under benchmark/ that unit-test the benchmark framework rather than an
# op (they PASS with zero result rows and would only skew the batch counts).
HARNESS_TEST_FILES = {"test_benchmark_case_api.py"}

# Index of the shape kept by --one-shape for keys taken from core_shapes (the
# 3rd, or the last if fewer): matches how the curated single shapes were picked,
# and is the middle of upstream's 5 default shapes ([4096, 4096] for Benchmark).
ONE_SHAPE_INDEX = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", action="append", default=None, metavar="LABEL=PATH",
                   help="repo to scan, repeatable (default: " +
                        ", ".join(f"{k}={v}" for k, v in REPOS.items()) + ")")
    p.add_argument("--verify-collect", action="store_true",
                   help="confirm each test function with pytest --collect-only (slow, "
                        "needs the full runtime environment; fills the collected column)")
    p.add_argument("--out-dir", type=Path, default=OPS_DIR,
                   help="inventory CSV output dir (default ops/)")
    p.add_argument("--refresh-shapes", type=Path, default=None, metavar="YAML",
                   help="refresh this shape set in place "
                        "(drop stale keys, sync the core_shapes base)")
    p.add_argument("--check-shapes", type=Path, action="append", default=None, metavar="YAML",
                  help="report keys of these shape sets that match no current op_name or "
                       "Benchmark class; such a key is silently ignored at run time "
                       "(repeatable). Unlike --refresh-shapes this never rewrites the file")
    p.add_argument("--one-shape", action="store_true",
                   help="single-shape set: keys whose value equals upstream core_shapes "
                        "(not hand-curated) keep one representative shape")
    return p.parse_args()


def _resolve_repos(raw: Optional[List[str]]) -> List[Tuple[str, Path]]:
    if not raw:
        return [(label, path) for label, path in REPOS.items() if path.is_dir()]
    repos = []
    for item in raw:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--repo expects LABEL=PATH, got {item!r}")
        repos.append((label, Path(path)))
    return repos


def _is_unconditional_skip(dec: ast.AST) -> bool:
    """pytest.mark.skip (not skipif), with or without a reason argument."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if not (isinstance(target, ast.Attribute) and target.attr == "skip"):
        return False
    mark = target.value
    return (isinstance(mark, ast.Attribute) and mark.attr == "mark"
            and isinstance(mark.value, ast.Name) and mark.value.id == "pytest")


def _op_names_in(node: ast.AST) -> Set[str]:
    """op_name string constants under node, in the three upstream spellings:
    1. an op_name= keyword argument of any call;
    2. the first positional string of super().__init__ (FusedDeepseekV4...);
    3. the first positional string when instantiating a *Benchmark class
       (ScaledMMBenchmark("scaled_mm", ...), where the name is forwarded
       through a parameter and never reaches super() as a constant).
    A false positive only keeps an unused key in the shape yaml."""
    names: Set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        for kw in sub.keywords:
            if kw.arg == "op_name" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                names.add(kw.value.value)
        first_str = (sub.args[0].value
                     if sub.args and isinstance(sub.args[0], ast.Constant)
                     and isinstance(sub.args[0].value, str) else None)
        if first_str is None:
            continue
        func = sub.func
        if (isinstance(func, ast.Attribute) and func.attr == "__init__"
                and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"):
            names.add(first_str)
        callee = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else "")
        if callee.endswith("Benchmark"):
            names.add(first_str)
    return names


def scan_repo(label: str, root: Path) -> Tuple[List[Dict[str, str]], Set[str]]:
    """(inventory rows, every op_name in the repo)."""
    bench = root / "benchmark"
    rows: List[Dict[str, str]] = []
    op_names_all: Set[str] = set()
    for path in sorted(bench.rglob("test_*.py")):
        rel = path.relative_to(bench)
        # Only test_-prefixed paths (top-level test_x.py and test_XXX/ subdirs),
        # so "benchmark/test_" + op_file + ".py" rebuilds the path.
        if not str(rel).startswith("test_") or rel.name in HARNESS_TEST_FILES:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            print(f"[gen] warning: cannot parse {path}, skipped: {exc}", file=sys.stderr)
            continue
        op_names_all |= _op_names_in(tree)
        op_file = str(rel)[len("test_"):-len(".py")]
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            marked = any(_is_unconditional_skip(d) for d in node.decorator_list)
            func_ops = sorted(_op_names_in(node))
            rows.append({
                "repo": label,
                "op": node.name[len("test_"):],
                "op_file": op_file,
                "file": f"benchmark/{rel}",
                "marked_skip": "yes" if marked else "",
                "collected": "",
                "op_names": ";".join(func_ops),
            })
    return rows, op_names_all


def verify_collect(label: str, root: Path, rows: List[Dict[str, str]]) -> None:
    """Fill the collected column from a pytest --collect-only run."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "--continue-on-collection-errors", "benchmark/"],
        cwd=str(root), text=True, capture_output=True, timeout=1200,
    )
    collected_funcs: Set[Tuple[str, str]] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" not in line or not line.startswith("benchmark/"):
            continue
        file_part, _, rest = line.partition("::")
        func = rest.split("::")[0].split("[")[0]
        collected_funcs.add((file_part, func))
    if not collected_funcs:
        print(f"[gen] warning: {label} collected nothing (rc={proc.returncode}), "
              f"stderr tail: {proc.stderr[-500:]}", file=sys.stderr)
        return
    for row in rows:
        key = (row["file"], f"test_{row['op']}")
        row["collected"] = "yes" if key in collected_funcs else "no"
    n_missing = sum(1 for r in rows if r["collected"] == "no")
    if n_missing:
        print(f"[gen] {label}: {n_missing} functions seen by the AST but not collected "
              "by pytest (import failure / conditional definition): collected=no")


def carry_over_collected(out_dir: Path, label: str,
                         rows: List[Dict[str, str]]) -> None:
    """Without --verify-collect, carry the collected column over from the existing
    CSV so rewriting the inventory keeps the last verification."""
    path = out_dir / f"inventory_{label}.csv"
    if not path.is_file():
        return
    with path.open(newline="") as f:
        prev = {(r["op"], r["op_file"]): r.get("collected", "")
                for r in csv.DictReader(f)}
    for row in rows:
        row["collected"] = prev.get((row["op"], row["op_file"]), "")


def write_inventory(out_dir: Path, label: str, rows: List[Dict[str, str]]) -> Path:
    out = out_dir / f"inventory_{label}.csv"
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out)
    return out


def refresh_shapes(path: Path, valid_op_names: Set[str],
                   repos: List[Tuple[str, Path]], one_shape: bool = False) -> None:
    """Refresh a shape set in place, on top of each repo's core_shapes.yaml.

    Upstream set_shapes falls back op_name key -> MRO class-name key
    (BlasBenchmark etc.) -> base-class DEFAULT_SHAPES (1-D). --shape_file
    replaces core_shapes.yaml instead of layering on it, so without the
    class-name keys BLAS-style ops expecting (B, M, N, K) would fall to the 1-D
    default and fail to unpack. Hence every core_shapes.yaml key missing here
    (class-name and op keys alike) is merged in verbatim; existing entries win,
    main merges before vllm (first one wins). Keys coming from core_shapes are
    never stale, and each refresh picks up upstream's current values.

    one_shape: keys whose value equals upstream's (merged in, or copied
    unchanged) keep only shape ONE_SHAPE_INDEX, so ops falling back to a
    class-name key run one shape too. Curated entries are left alone."""
    data = yaml.safe_load(path.read_text()) or {}
    core_maps = []
    for label, root in repos:
        core = root / "benchmark" / "core_shapes.yaml"
        if core.is_file():
            core_maps.append((label, yaml.safe_load(core.read_text()) or {}))
    core_keys = {k for _, m in core_maps for k in m}
    kept = {k: v for k, v in data.items() if k in valid_op_names}
    dropped = {k: v for k, v in data.items()
               if k not in valid_op_names and k not in core_keys}
    for _, m in core_maps:
        for key, val in m.items():
            kept.setdefault(key, val)
    if one_shape:
        upstream = {}
        for _, m in reversed(core_maps):  # same precedence as the merge: main wins
            upstream.update(m)
        for key, val in kept.items():
            shapes = val.get("shapes") if isinstance(val, dict) else None
            if key in upstream and val == upstream[key] and shapes and len(shapes) > 1:
                kept[key] = {**val, "shapes": [shapes[min(ONE_SHAPE_INDEX, len(shapes) - 1)]]}

    header = ("# Maintained by gen_inventory.py --refresh-shapes.\n"
              "# Top-level keys: FlagGems op_name or Benchmark class name (upstream's\n"
              "# fallback chain); ops with neither use their base-class DEFAULT_SHAPES.\n"
              "# Keys other than the curated ones come from upstream core_shapes.yaml"
              + (" (one shape kept)" if one_shape else "") + ".\n")
    path.write_text(header + yaml.safe_dump(
        kept, sort_keys=True, default_flow_style=None, allow_unicode=True))
    if dropped:
        unmatched = path.with_name(f"{path.stem}.unmatched.yaml")
        unmatched.write_text(
            f"# {len(dropped)} keys of {path.name} match no upstream op_name: map renamed\n"
            "# ops to their new name and move them back, delete the rest.\n"
            + yaml.safe_dump(dropped, sort_keys=True, default_flow_style=None,
                             allow_unicode=True))
        print(f"[gen] shapes: kept {len(kept)}, moved out {len(dropped)} -> {unmatched}")
    else:
        print(f"[gen] shapes: all {len(kept)} keys valid")
    # Coverage note: op_names without an entry run upstream's default shapes.
    uncovered = valid_op_names - set(kept)
    print(f"[gen] shapes: {len(uncovered)} of {len(valid_op_names)} upstream op_names "
          "have no entry (upstream default shapes)")


def check_shapes(paths, valid_op_names, repos) -> int:
    """Report shape-set keys that match no current op_name or Benchmark class.

    Upstream renames an op_name now and then. A set still keyed on the old name
    matches nothing, upstream falls back to the benchmark's own hardcoded shapes
    and the run looks clean, so nothing but a check like this catches the drift.
    Valid keys are the op_names, the Benchmark class names (upstream's set_shapes
    falls back along the MRO) and whatever each repo's core_shapes.yaml defines.
    Returns the number of stale keys; never rewrites a file.
    """
    classes, core_keys = set(), set()
    for _label, root in repos:
        for path in sorted((root / "benchmark").rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except SyntaxError:
                continue
            classes |= set(n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        core = root / "benchmark" / "core_shapes.yaml"
        if core.is_file():
            core_keys |= set(yaml.safe_load(core.read_text()) or dict())
    known = set(valid_op_names) | classes | core_keys
    stale_total = 0
    for path in paths:
        if not path.is_file():
            print(f"[gen] check-shapes: no {path}")
            continue
        try:
            entries = yaml.safe_load(path.read_text()) or dict()
        except yaml.YAMLError as exc:
            print(f"[gen] check-shapes: {path} is not valid YAML: {str(exc).splitlines()[0]}")
            continue
        if not isinstance(entries, dict):
            print(f"[gen] check-shapes: {path} is not a mapping of op_name -> entry; skipped")
            continue
        stale = sorted(k for k in entries if k not in known)
        stale_total += len(stale)
        detail = ": " + ", ".join(stale[:10]) + (" ..." if len(stale) > 10 else "") if stale else ""
        print(f"[gen] check-shapes: {path} {len(entries)} keys, {len(stale)} stale{detail}")
    if stale_total:
        print(f"[gen] check-shapes: {stale_total} key(s) would be ignored at run time; rename them "
              "to the current op_name or drop them")
    return stale_total


def main() -> None:
    args = parse_args()
    repos = _resolve_repos(args.repo)
    if not repos:
        raise SystemExit("no repo to scan")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_op_names: Set[str] = set()
    for label, root in repos:
        rows, op_names = scan_repo(label, root)
        all_op_names |= op_names
        if args.verify_collect:
            verify_collect(label, root, rows)
        else:
            carry_over_collected(args.out_dir, label, rows)
        out = write_inventory(args.out_dir, label, rows)
        n_skip = sum(1 for r in rows if r["marked_skip"])
        print(f"[gen] {label}: {len(rows)} test functions -> {out} "
              f"({n_skip} with an unconditional skip mark)")

    if args.check_shapes:
        check_shapes(args.check_shapes, all_op_names, repos)

    if args.refresh_shapes:
        refresh_shapes(args.refresh_shapes, all_op_names, repos, one_shape=args.one_shape)


if __name__ == "__main__":
    main()
