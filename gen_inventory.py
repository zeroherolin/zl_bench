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
from typing import Dict, List, Optional, Sequence, Set, Tuple

import yaml

from shape_set import SCOPE, dump_shapes, resolve, split_key

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
                        "Benchmark class, or sit on the name of a test instead of its "
                        "op_name; such a key is silently ignored at run time (repeatable). "
                        "Rewrites nothing, not the inventory either unless --verify-collect "
                        "or --refresh-shapes is given too; exits 1 if it finds a problem")
    p.add_argument("--one-shape", action="store_true",
                   help="single-shape set: keys whose value equals upstream core_shapes "
                        "(not hand-curated) keep one representative shape")
    return p.parse_args()


def _resolve_repos(raw: Optional[List[str]]) -> List[Tuple[str, Path]]:
    """The repos to scan. A checkout without benchmark/ is an error, not an empty repo:
    scanning it would write an empty inventory over the committed one."""
    if raw:
        repos = []
        for item in raw:
            label, _, path = item.partition("=")
            if not path:
                raise SystemExit(f"--repo expects LABEL=PATH, got {item!r}")
            repos.append((label, Path(path)))
    else:
        repos = list(REPOS.items())
    missing = [f"{label}={path}" for label, path in repos if not (path / "benchmark").is_dir()]
    if missing:
        raise SystemExit(
            f"[gen] no benchmark/ in {', '.join(missing)}: set FLAGGEMS_MAIN_DIR / "
            "FLAGGEMS_VLLM_DIR to the checkouts, or pass --repo LABEL=PATH for the repos to scan")
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


def _inventory_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_inventory(out_dir: Path, label: str, rows: List[Dict[str, str]]) -> Path:
    out = out_dir / f"inventory_{label}.csv"
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out)
    return out


def _core_shapes(repos: List[Tuple[str, Path]]) -> List[Tuple[str, dict]]:
    """[(repo label, its benchmark/core_shapes.yaml as a dict)], in repo order."""
    out = []
    for label, root in repos:
        core = root / "benchmark" / "core_shapes.yaml"
        if core.is_file():
            out.append((label, yaml.safe_load(core.read_text()) or {}))
    return out


def refresh_shapes(path: Path, valid_op_names: Set[str], repos: List[Tuple[str, Path]],
                   one_shape: bool = False,
                   names_by_repo: Optional[Dict[str, Set[str]]] = None) -> None:
    """Refresh a shape set in place, on top of each repo's core_shapes.yaml.

    Upstream set_shapes falls back op_name key -> MRO class-name key
    (BlasBenchmark etc.) -> base-class DEFAULT_SHAPES (1-D). --shape_file
    replaces core_shapes.yaml instead of layering on it, so without the
    class-name keys BLAS-style ops expecting (B, M, N, K) would fall to the 1-D
    default and fail to unpack. Hence every core_shapes.yaml key missing here
    (class-name and op keys alike) is merged in verbatim; existing entries win,
    main merges before vllm (first one wins). Keys coming from core_shapes are
    never stale, and each refresh picks up upstream's current values.

    The set serves both repos, and a core_shapes key of one repo may mean something
    else in the other: fused_recurrent_gated_delta_rule is a sequence length in
    FlagGems and a case name in FlagGems-vllm, and whichever benchmark is handed the
    other's entry fails to unpack it. With names_by_repo (the op_names and Benchmark
    class names of each repo) a merged key is scoped to its repo, `<repo>:<key>`
    (shape_set.py), when sharing it would change what the other repo runs:

    - the other repo uses the name too, but its core_shapes has no such key;
    - both define the key, with different shapes: the first repo keeps the plain
      key, the other gets a scoped one.

    A key the set already has keeps its form, plain or scoped: only a key merged in
    for the first time is scoped, so a refresh never changes which repo an existing
    entry drives.

    one_shape: keys whose value equals upstream's (merged in, or copied
    unchanged) keep only shape ONE_SHAPE_INDEX, so ops falling back to a
    class-name key run one shape too. Curated entries are left alone."""
    data = yaml.safe_load(path.read_text()) or {}
    core_maps = _core_shapes(repos)
    core = dict(core_maps)
    core_keys = {k for _, m in core_maps for k in m}
    names = names_by_repo or {}

    def valid(key) -> bool:
        scope, name = split_key(key)
        if scope is None:
            return name in valid_op_names
        return scope in core and (name in names.get(scope, valid_op_names) or name in core[scope])

    def from_core(key) -> bool:
        scope, name = split_key(key)
        return name in core_keys if scope is None else name in core.get(scope, ())

    kept = {k: v for k, v in data.items() if valid(k)}
    dropped = {k: v for k, v in data.items() if not valid(k) and not from_core(k)}
    merged: Dict[str, str] = {}  # plain key -> the repo whose core_shapes it was merged from
    for label, m in core_maps:
        others = [o for o, _ in core_maps if o != label]
        for key, val in m.items():
            if key in merged:
                if core[merged[key]][key] != val:
                    kept.setdefault(f"{label}{SCOPE}{key}", val)  # same key, other shapes
            elif key in kept:
                pass  # an entry of the set wins
            elif key not in data and any(key in names.get(o, ()) and key not in core[o]
                                         for o in others):
                kept.setdefault(f"{label}{SCOPE}{key}", val)  # the other repo means something else
            else:
                kept[key] = val
                merged[key] = label
    if one_shape:
        for key, val in kept.items():
            scope, name = split_key(key)
            # same precedence as the merge: the first repo's value is upstream's for a plain key
            upstream = [m[name] for label, m in core_maps if name in m and scope in (None, label)]
            shapes = val.get("shapes") if isinstance(val, dict) else None
            if upstream and val == upstream[0] and shapes and len(shapes) > 1:
                kept[key] = dict(val, shapes=[shapes[min(ONE_SHAPE_INDEX, len(shapes) - 1)]])

    header = ("# Maintained by gen_inventory.py --refresh-shapes.\n"
              "# Top-level keys: FlagGems op_name or Benchmark class name (upstream's\n"
              "# fallback chain); ops with neither use their base-class DEFAULT_SHAPES.\n"
              "# A key `<repo>:<name>` applies to that repo only (main = FlagGems, vllm =\n"
              "# FlagGems-vllm) and beats the plain key there; see shape_set.py.\n"
              "# Keys other than the curated ones come from upstream core_shapes.yaml"
              + (" (one shape kept)" if one_shape else "") + ".\n")
    path.write_text(header + dump_shapes(kept))
    if dropped:
        unmatched = path.with_name(f"{path.stem}.unmatched.yaml")
        unmatched.write_text(
            f"# {len(dropped)} keys of {path.name} match no upstream op_name: map renamed\n"
            "# ops to their new name and move them back, delete the rest.\n"
            + dump_shapes(dropped))
        print(f"[gen] shapes: kept {len(kept)}, moved out {len(dropped)} -> {unmatched}")
    else:
        print(f"[gen] shapes: all {len(kept)} keys valid")
    scoped = sorted(k for k in kept if split_key(k)[0] is not None)
    if scoped:
        print(f"[gen] shapes: {len(scoped)} key(s) apply to one repo only: {', '.join(scoped)}")
    # Coverage note: op_names without an entry run upstream's default shapes.
    uncovered = valid_op_names - {split_key(k)[1] for k in kept}
    print(f"[gen] shapes: {len(uncovered)} of {len(valid_op_names)} upstream op_names "
          "have no entry (upstream default shapes)")


def _benchmark_modules(repos: List[Tuple[str, Path]]):
    """(repo label, parsed module) of every Python file under the repos' benchmark/."""
    for label, root in repos:
        for path in sorted((root / "benchmark").rglob("*.py")):
            try:
                yield label, ast.parse(path.read_text(errors="replace"))
            except SyntaxError:
                continue


def hardcoded_shape_users(repos: List[Tuple[str, Path]]) -> Dict[str, Set[str]]:
    """{Benchmark class that hardcodes its shapes: the op_names it is created with}.

    Such a class overrides set_shapes() without calling the base class, so it reads no
    key of its own accord: _shape_override_plugin feeds it the entry of its op_name,
    and without one it runs its whole hardcoded list. The op_names are the op_name=
    keywords of the calls to the class, over both repos (they share most classes but
    not always the op_name, e.g. fused_marlin_moe_w8a16_int8 / fused_marlin_moe_w8a16).
    """
    hardcoded: Set[str] = set()
    users: Dict[str, Set[str]] = {}
    for _label, tree in _benchmark_modules(repos):
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for f in node.body:
                    if isinstance(f, ast.FunctionDef) and f.name == "set_shapes" \
                            and "super().set_shapes" not in ast.unparse(f):
                        hardcoded.add(node.name)
            elif isinstance(node, ast.Call):
                func = node.func
                callee = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else "")
                for kw in node.keywords:
                    if kw.arg == "op_name" and isinstance(kw.value, ast.Constant) \
                            and isinstance(kw.value.value, str):
                        users.setdefault(callee, set()).add(kw.value.value)
    return {cls: names for cls, names in users.items() if cls in hardcoded}


def check_shapes(paths: Sequence[Path], valid_op_names: Set[str], repos: List[Tuple[str, Path]],
                 tests: Sequence[Dict[str, str]] = ()) -> int:
    """Report what in these shape sets would not drive the benchmark it is meant for.

    Upstream renames an op_name now and then. A set still keyed on the old name
    matches nothing, upstream falls back to the benchmark's own hardcoded shapes
    and the run looks clean, so nothing but a check like this catches the drift.
    Counted as problems (the return value; the file is never rewritten):

    - a file that cannot be read as a mapping of keys;
    - a stale key: no current op_name, Benchmark class name (upstream's set_shapes
      falls back along the MRO) or core_shapes.yaml key;
    - a key on the name of a test whose benchmark uses another op_name and has no
      entry: `tests` are inventory rows, and the op_name is what the benchmark reads.

    Reported as a note, not counted: op_names without an entry that share a
    hardcoded-shape Benchmark class with an op_name that has one. In a set curated
    for a few benchmarks that is usually an omission (one repo's name for the
    benchmark was left out); in a whole-suite set it may be meant.
    """
    classes, core_keys = set(), set()
    for _label, tree in _benchmark_modules(repos):
        classes |= {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for _label, root in repos:
        core = root / "benchmark" / "core_shapes.yaml"
        if core.is_file():
            core_keys |= set(yaml.safe_load(core.read_text()) or {})
    known = set(valid_op_names) | classes | core_keys
    users = hardcoded_shape_users(repos)
    problems = 0
    for path in paths:
        if not path.is_file():
            print(f"[gen] check-shapes: no {path}")
            problems += 1
            continue
        try:
            entries = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            print(f"[gen] check-shapes: {path} is not valid YAML: {str(exc).splitlines()[0]}")
            problems += 1
            continue
        if not isinstance(entries, dict):
            print(f"[gen] check-shapes: {path} is not a mapping of op_name -> entry; skipped")
            problems += 1
            continue
        # A key may be scoped to one repo, `<repo>:<name>` (shape_set.py).
        keys = [split_key(k) for k in entries]
        scopes = set(REPOS) | {label for label, _ in repos}
        stale = sorted(str(k) for k, (scope, name) in zip(entries, keys, strict=True)
                       if name not in known or (scope is not None and scope not in scopes))
        detail = ": " + ", ".join(stale[:10]) + (" ..." if len(stale) > 10 else "") if stale else ""
        print(f"[gen] check-shapes: {path} {len(entries)} keys, {len(stale)} stale{detail}")
        problems += len(stale)
        views = {label: resolve(entries, label) for label in {r["repo"] for r in tests}}
        for row in tests:
            view = views[row["repo"]]
            names = [n for n in row.get("op_names", "").split(";") if n]
            if row["op"] in view and names and not any(n in view for n in names):
                print(f"[gen] check-shapes:   key {row['op']!r} is the name of a test of "
                      f"{row['repo']}, whose benchmark reads op_name {' / '.join(names)}: "
                      "no entry for that")
                problems += 1
        present = {name for _, name in keys}
        for cls, names in sorted(users.items()):
            have, miss = sorted(names & present), sorted(names - present)
            if have and miss:
                print(f"[gen] check-shapes:   note: {cls} hardcodes its shapes and is also "
                      f"created as {', '.join(miss)}, which has no entry like {', '.join(have)}")
    if problems:
        print(f"[gen] check-shapes: {problems} problem(s): a stale or misnamed key is ignored at "
              "run time; rename it to the current op_name or drop it")
    return problems


def main() -> None:
    args = parse_args()
    repos = _resolve_repos(args.repo)
    # Shape keys are valid against both repos together: with one of them left out, the
    # keys of its ops would look stale and --refresh-shapes would move them out.
    absent = sorted(set(REPOS) - {label for label, _ in repos})
    if absent and args.refresh_shapes:
        raise SystemExit(f"[gen] --refresh-shapes needs every repo ({', '.join(REPOS)}); "
                         f"not scanned: {', '.join(absent)}")
    # A check reads; it rewrites the inventory only together with a request to do so.
    check_only = bool(args.check_shapes) and not (args.refresh_shapes or args.verify_collect)
    if not check_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    all_op_names: Set[str] = set()
    tests: List[Dict[str, str]] = []
    names_by_repo: Dict[str, Set[str]] = {}  # op_names and Benchmark class names per repo
    for label, root in repos:
        rows, op_names = scan_repo(label, root)
        all_op_names |= op_names
        names_by_repo[label] = set(op_names)
        tests += rows
        if check_only:
            continue
        if args.verify_collect:
            verify_collect(label, root, rows)
        else:
            carry_over_collected(args.out_dir, label, rows)
        out = write_inventory(args.out_dir, label, rows)
        n_skip = sum(1 for r in rows if r["marked_skip"])
        print(f"[gen] {label}: {len(rows)} test functions -> {out} "
              f"({n_skip} with an unconditional skip mark)")

    problems = 0
    if args.check_shapes:
        # A repo that is not scanned still has its op_names in the committed inventory;
        # its class names and core_shapes keys are not known, so such keys read as stale.
        for label in absent:
            known = _inventory_rows(OPS_DIR / f"inventory_{label}.csv")
            tests += known
            all_op_names |= {n for r in known for n in r["op_names"].split(";") if n}
            print(f"[gen] check-shapes: {label} is not scanned: its op_names are taken from "
                  f"ops/inventory_{label}.csv, and keys that are a Benchmark class or a "
                  "core_shapes key of that repo alone will be reported as stale")
        problems = check_shapes(args.check_shapes, all_op_names, repos, tests)

    if args.refresh_shapes:
        for label, tree in _benchmark_modules(repos):
            names_by_repo[label] |= {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        refresh_shapes(args.refresh_shapes, all_op_names, repos, one_shape=args.one_shape,
                       names_by_repo=names_by_repo)

    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
