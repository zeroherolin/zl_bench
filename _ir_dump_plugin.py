"""pytest plugin: at session end, dump the IR of every kernel variant the
measurement actually used — ttgir grouped by benchmark shape for reading and
diffing, ttir + compile metadata keyed by cache hash for offline recompilation
(ir_gate.py) — and which kernels each result row ran (rows.tsv).

Active only when FLAGGEMS_PERF_IR_DUMP_DIR (the run dir) and TRITON_CACHE_DIR
are both set; run_pytest.sh sets both. Pieces, all keyed by the kernel's
Triton cache dir:

1. Compilation listener — writes a __constexprs.json sidecar (constexpr values
   + per-arg specializations like tt.divisibility) next to each compiled
   kernel. The ttgir alone doesn't say which autotune variant it is.
2. Shape tracking — publishes the benchmark shape currently being fed, e.g.
   "16x256x7168x2048x8". Upstream has two input paths, both wrapped (after
   _shape_override_plugin has done its patching): the legacy get_input_iter
   generator, labelled by the shape its walk over self.shapes is at
   (_shape_scope; "input<i>" for an iterator that walks no shapes), and the
   Case API (Benchmark.supports_cases() -> get_case_iter + build_inputs per
   case), whose label comes from the case's shape dict.
3. Launch tracking — kernel_load_end_hook maps a loaded function handle to its
   cache dir (must be the END hook: at load-start the handle is still None);
   launch_enter_hook counts launches per (kernel cache dir, current shape).
   Launches made inside Autotuner._bench (the tuning sweep; LibTuner inherits
   it) are counted separately, so "actually used" = launched at least once
   OUTSIDE the sweep.
4. Row tracking — every result-table row (a BenchmarkMetrics) gets the
   kernels its measurement ran: launched outside the sweep, inside the timed
   function (_timing_scope), for the row's input. The row is usually created
   before its measurement; one created after it takes what its input
   launched until then. Input construction and the timing harness's own
   launches are not counted. A benchmark that times without
   triton.testing.do_bench leaves the timed set empty; its rows get every
   launch made for their input (scope "row" instead of "timed"). Written to
   rows.tsv in table order, one line per table row, with the fingerprints of
   the kernels' binaries (_binary_id), so row i of a table in run.log maps
   to its kernels (ir_gate.py) and to what they are (ab_compare.py).
5. Dump at exit — walks TRITON_CACHE_DIR and, for each used variant, writes
       ttgir/<shape>/<kernel>/<distinguishing constexprs>_w{warps}s{stages}.ttgir
   once per shape that used it, and
       ttir/<cache hash>/<kernel>.ttir + <kernel>.json (Triton's metadata)
   once per variant. Only constexprs that VARY within a (shape, kernel) group
   are named, abbreviated per-word (BLOCK_SIZE_M -> BSM; legend in
   ttgir/naming.md). Same-name collisions are disambiguated by signature dtype
   (dtype sweeps, e.g. __fp16), then varying per-arg specializations (e.g.
   __EMdiv16), then a cache-hash prefix. ttgir/index.tsv lists every compiled
   variant per shape with launch counts and cache hash, including never-used
   sweep losers (index only, no files). atexit (not sessionfinish) so a run
   whose session ends in a CUDA error still dumps whatever was compiled.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import pytest

import _binary_id
import _shape_scope
import _timing_scope
from _fg_benchmark import benchmark_classes
from _term_style import DIM, paint, tag

_TAG = "[ir-dump-plugin]"
_MAX_LABEL = 120   # a label is a directory name; filesystems stop at 255 bytes
_DUMP_DIR_ENV = "FLAGGEMS_PERF_IR_DUMP_DIR"

_fn_to_cachedir: dict = {}   # GPU function handle -> cache-dir basename
_launches: dict = {}         # cache-dir basename -> {shape label: [real, sweep]}
_in_bench = 0                # >0 while inside Autotuner._bench (autotune sweep)
_current_shape: str | None = None
_walk_label: str | None = None  # label of the shape the input iterator is at (_on_shape)
_input_label = "input0"      # label of the input being fed when the iterator walks no shapes
_current_row: int | None = None  # id of the result row being measured
_row_timed: dict = defaultdict(set)  # row id -> cache-dir basenames launched in the timed function
_row_any: dict = defaultdict(set)    # row id -> every cache-dir basename launched while it was current
# row id -> launched for its input outside a timed function, before the row existed
_row_before: dict = {}
# launched since the last input with no row current, keyed by "in the timed function"
_unclaimed: dict = {True: set(), False: set()}
_row_count = 0
_tables: list = []           # [(op_name, dtype, [(row id, shape label)])] in print order
_dump_enabled = False


# --- compile-time sidecar -------------------------------------------------

def _plain(value):
    value = getattr(value, "value", value)  # unwrap tl.constexpr
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _compile_listener(*, src, metadata, metadata_group, times, cache_hit):
    try:
        fn = getattr(src, "fn", None)
        arg_names = getattr(fn, "arg_names", None)
        constants = getattr(src, "constants", None)
        if not arg_names or constants is None:
            return  # IRSource or unexpected layout: nothing to record
        paths = list(metadata_group.values())
        if not paths:
            return
        path = os.path.join(os.path.dirname(paths[0]), "__constexprs.json")
        if os.path.exists(path):
            return  # cache hit on a dir we already annotated

        def arg_name(key):
            # ASTSource keys constants/attrs by arg-index tuples; map back to names.
            if isinstance(key, tuple):
                return ".".join(
                    arg_names[i] if isinstance(i, int) and i < len(arg_names) else str(i)
                    for i in key
                )
            return str(key)

        out = {arg_name(k): _plain(v) for k, v in constants.items()}

        # Argument signature (dtypes): benchmarks sweep dtypes, and dtype is
        # not a constexpr — without this, dtype variants collide into hash
        # suffixes. Values look like "*fp16", "i32".
        sig = {}
        try:
            sig = {str(k): str(v) for k, v in (getattr(src, "signature", None) or {}).items()
                   if str(v) != "constexpr"}
        except Exception:
            pass

        # Per-arg specializations, e.g. ("tt.divisibility", 16) -> "div16".
        specs = {}
        for key, props in (getattr(src, "attrs", None) or {}).items():
            encoded = []
            for p in props or []:
                try:
                    pname, pval = p[0], p[1]
                except (TypeError, IndexError):
                    encoded.append(str(p))
                    continue
                encoded.append(f"div{pval}" if "divisibility" in str(pname)
                               else f"{pname}={pval}")
            if encoded:
                specs[arg_name(key)] = sorted(encoded)

        tmp = f"{path}.tmp.pid{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"name": getattr(src, "name", "unknown"),
                       "constexprs": out, "attrs": specs, "signature": sig},
                      f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except Exception as exc:  # never break compilation over a metadata dump
        print(f"{tag(_TAG)} sidecar dump failed: {exc}", file=sys.stderr)


# --- shape + launch tracking ----------------------------------------------

def _load_hook(module, function, name, metadata_group, hash):
    try:
        paths = list(metadata_group.values())
        if paths:
            _fn_to_cachedir[function] = os.path.basename(os.path.dirname(paths[0]))
    except Exception:
        pass


def _launch_hook(md):
    try:
        cachedir = _fn_to_cachedir.get(md.data.get("function"))
        if cachedir is None:
            return
        rec = _launches.setdefault(cachedir, {}).setdefault(
            _current_shape or "shape_unknown", [0, 0])
        rec[1 if _in_bench else 0] += 1
        if _in_bench:
            return
        timed = _timing_scope.in_measurement()
        if _current_row is not None:
            _row_any[_current_row].add(cachedir)
            if timed:
                _row_timed[_current_row].add(cachedir)
        else:
            # No row yet: a benchmark may time first and make its row afterwards
            # (_wrap_metrics_init hands these to it).
            _unclaimed[timed].add(cachedir)
    except Exception:
        pass


def _wrap_bench(original):
    def wrapped(self, *args, **kwargs):
        global _in_bench
        _in_bench += 1
        try:
            return original(self, *args, **kwargs)
        finally:
            _in_bench -= 1
    return wrapped


def _fit(label):
    """Keep a label usable as a directory name.

    Labels become directories under ttgir/. A shape can be long (a varlen attention
    case carries its cu_seqlens), and a name over the filesystem's 255-byte limit
    makes the whole dump fail. Truncate and append a digest, so long labels stay
    readable at the front and distinct at the back.
    """
    if len(label) <= _MAX_LABEL:
        return label
    digest = hashlib.sha1(label.encode()).hexdigest()[:12]
    return label[:_MAX_LABEL - len(digest) - 1] + "_" + digest


def _label_part(value):
    # A nested shape, e.g. [input_shape, output_size], joins its inner dims with "-":
    # [[2, 256, 14], [2, 14]] reads 2-256-14x2-14 instead of running into 225614x214,
    # which is also what [[22, 56, 14], [21, 4]] would give.
    if isinstance(value, (list, tuple)):
        return "-".join(_label_part(v) for v in value)
    return re.sub(r"\W", "", str(value))


def _label_of(shape):
    if isinstance(shape, (list, tuple)):
        return _fit("x".join(_label_part(v) for v in shape))
    return _fit(_label_part(shape))


class _WalkedList(list):
    """Benchmark.shapes while its get_input_iter runs: iterating it tells _on_shape
    which shape the iterator is at (_shape_scope.walk)."""

    def __iter__(self):
        return _shape_scope.walk(list.__iter__(self))


class _WalkedTuple(tuple):
    def __iter__(self):
        return _shape_scope.walk(tuple.__iter__(self))


_WALKED = {list: _WalkedList, tuple: _WalkedTuple}


def _new_input():
    """An input begins (or a row took what was launched for it so far)."""
    for kernels in _unclaimed.values():
        kernels.clear()


def _between_rows():
    """No row is being measured: a table was printed, or a test begins. A benchmark that
    feeds itself, without get_input_iter or build_inputs, has no other boundary between
    its rows; without one the kernels of its next row would go to the last."""
    global _current_row
    _current_row = None
    _new_input()


def pytest_runtest_setup(item):
    _between_rows()


def _on_shape(shape):
    """The input iterator reached `shape`; None: it walked all its shapes."""
    global _walk_label, _current_shape
    _walk_label = _label_of(shape) if shape is not None else None
    _current_shape = _walk_label or _input_label


def _case_label(case):
    """Label of a Case-API case, matching the legacy label of the same shape.
    The shape-family cases carry the yaml shape as shape["input"] (or
    shape["inputs"][0] for binary ops); BLAS spells it out as b/m/n/k."""
    shape = getattr(case, "shape", None)
    if isinstance(shape, dict):
        if "input" in shape:
            return _label_of(shape["input"])
        inputs = shape.get("inputs")
        if isinstance(inputs, (list, tuple)) and inputs:
            return _label_of(inputs[0])
        if shape and all(isinstance(v, (int, float)) for v in shape.values()):
            return _label_of(list(shape.values()))
    return f"case{getattr(case, 'ordinal', '')}"


def _wrap_input_iter(original):
    """get_input_iter, publishing the label of each input it yields.

    The label is the shape the iterator is at in its walk over self.shapes (or over the
    yaml shapes, for an iterator _shape_override_plugin redirected): right also when a
    shape yields several inputs, which counting the inputs would not be. An iterator
    that walks no shapes (a hardcoded list of cases), or that has walked them all before
    its first input, gets input<i> for its i-th input: it says nothing about a shape,
    which is better than naming the wrong one.
    """
    def patched(self, dtype):
        global _current_shape, _current_row, _walk_label, _input_label
        shapes = getattr(self, "shapes", None)
        walked = _WALKED.get(type(shapes))
        if walked is not None:
            self.shapes = walked(shapes)
        _walk_label = None
        it = original(self, dtype)
        idx = 0
        try:
            while True:
                # Published before the generator resumes, and updated by _on_shape as
                # it moves on, so kernels launched while the inputs are built belong
                # to the right shape too.
                _input_label = f"input{idx}"
                _current_shape = _walk_label or _input_label
                _current_row = None
                _new_input()
                try:
                    item = next(it)
                except StopIteration:
                    _current_shape = _walk_label = None
                    return
                yield item
                idx += 1
        finally:
            if walked is not None and type(getattr(self, "shapes", None)) is walked:
                self.shapes = shapes
    return patched


def _wrap_build_inputs(original):
    # The label stays set after the inputs are built, so the measurement that
    # follows attributes to this case too.
    def patched(self, case):
        global _current_shape, _current_row
        _current_shape = _case_label(case)
        _current_row = None
        _new_input()
        return original(self, case)
    return patched


def _wrap_metrics_init(original):
    # Upstream creates one BenchmarkMetrics per result row, right before
    # measuring it; from here on the row's launches are its kernels.
    def patched(self, *args, **kwargs):
        global _current_row, _row_count
        original(self, *args, **kwargs)
        self._zl_row = _row_count
        self._zl_shape = _current_shape or "shape_unknown"
        # What this input launched before the row existed. Most benchmarks make the
        # row and then measure; one that measures first would otherwise have a row
        # without kernels, which a gate takes for a row no compiler change can touch.
        measured = bool(_unclaimed[True])
        if measured:
            _row_timed[_row_count] |= _unclaimed[True]
        _row_before[_row_count] = set(_unclaimed[False])
        _new_input()
        # A row whose measurement came first is complete: what is launched from here
        # on belongs to the next one.
        _current_row = None if measured else _row_count
        _row_count += 1
    return patched


def _wrap_result_str(original):
    # A result table is logged by printing its BenchmarkResult; note its rows
    # the first time, in the order run.log shows them.
    def patched(self):
        if not getattr(self, "_zl_noted", False):
            self._zl_noted = True
            _tables.append((str(self.op_name), str(self.dtype).strip(),
                            [(getattr(m, "_zl_row", None), getattr(m, "_zl_shape", "shape_unknown"))
                             for m in (self.result or [])]))
            _between_rows()  # its rows are complete: what is launched next is not theirs
        return original(self)
    return patched


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    # trylast: run after _shape_override_plugin has re-pointed get_input_iter,
    # so we wrap the version that will actually execute.
    if not _dump_enabled:
        return
    # Wrapping in place keeps upstream's supports_cases() identity checks
    # (`type(self).build_inputs is not Base.build_inputs`) answering as before.
    wrappers = (("get_input_iter", _wrap_input_iter),
                ("build_inputs", _wrap_build_inputs))
    wrapped = 0
    for cls in benchmark_classes():
        for name, wrap in wrappers:
            own = cls.__dict__.get(name)
            if own is not None:
                setattr(cls, name, wrap(own))
                wrapped += 1
    # trylast also puts this after _pretty_report_plugin's __str__ and any
    # __str__ wrapper a benchmark module installs at import.
    from benchmark import consts
    consts.BenchmarkMetrics.__init__ = _wrap_metrics_init(consts.BenchmarkMetrics.__init__)
    consts.BenchmarkResult.__str__ = _wrap_result_str(consts.BenchmarkResult.__str__)
    print(f"{tag(_TAG)} shape tracking wrapped {wrapped} input builders "
          "(get_input_iter / Case-API build_inputs); row tracking on", file=sys.stderr, flush=True)


# --- naming ----------------------------------------------------------------

_SAN = re.compile(r"[^A-Za-z0-9.-]+")


def _fmt_value(v):
    # isinstance check first: 1 == True in Python, a plain dict lookup would
    # render GROUP_SIZE_M=1 as "T".
    if isinstance(v, bool):
        return "T" if v else "F"
    if v is None:
        return "-"
    s = str(v)
    if "." in s and not re.fullmatch(r"-?\d+(\.\d+)?", s):
        s = s.split(".")[-1]  # dotted repr like triton.language.bfloat16
    s = _SAN.sub("", s)
    return s[:24] or "x"


def _abbrev(name):
    words = [w for w in name.split("_") if w]
    if len(name) <= 4 or len(words) < 2:
        return name
    out = []
    for w in words:
        m = re.match(r"^([A-Za-z])[A-Za-z]*?(\d*)$", w)
        out.append((m.group(1) + m.group(2)) if m else w[0])
    return "".join(out)


def _disambiguate(group):
    """Suffixes for variants whose constexprs + launch config coincide: name by
    the signature dtypes that differ (dtype sweeps), else by differing per-arg
    specializations, else fall back to a hash prefix."""
    sig_keys = sorted({k for v in group for k in (v.get("signature") or {})})
    sig_varying = [
        k for k in sig_keys
        if len({(v.get("signature") or {}).get(k) for v in group}) > 1
    ]
    if sig_varying:
        # One dtype sweep usually shifts every tensor arg together; the first
        # varying arg's dtype identifies the variant. Sanitize "*fp16" -> fp16.
        sufs = []
        for v in group:
            toks = [re.sub(r"\W", "", (v.get("signature") or {}).get(k) or "none")
                    for k in sig_varying]
            uniq = sorted(set(toks))
            sufs.append("__" + (uniq[0] if len(uniq) == 1 else "_".join(
                f"{_abbrev(k)}{t}" for k, t in zip(sig_varying, toks, strict=True))))
        if len(set(sufs)) == len(group):
            return sufs
    attr_keys = sorted({k for v in group for k in (v.get("attrs") or {})})
    varying = [
        k for k in attr_keys
        if len({tuple((v.get("attrs") or {}).get(k, [])) for v in group}) > 1
    ]
    if varying:
        sufs = []
        for v in group:
            toks = [
                f"{_abbrev(k)}{'.'.join((v.get('attrs') or {}).get(k) or ['none'])}"
                for k in varying
            ]
            sufs.append("__" + "_".join(toks))
        if len(set(sufs)) == len(group):
            return sufs
    return ["__" + v["hash"][:8] for v in group]


def _name_group(items):
    """Filenames for one (shape, kernel) group: only constexprs whose value
    varies within the group, abbreviated. Returns ([(item, filename)], legend)."""
    keys = sorted({k for it in items for k in it["constexprs"]})
    varying = [
        k for k in keys
        if len({json.dumps(it["constexprs"].get(k), sort_keys=True) for it in items}) > 1
    ]
    by_ab = defaultdict(list)
    for k in varying:
        by_ab[_abbrev(k)].append(k)
    ab = {k: (a if len(ks) == 1 else k) for a, ks in by_ab.items() for k in ks}

    named = defaultdict(list)
    for it in items:
        parts = [f"{ab[k]}{_fmt_value(it['constexprs'][k])}"
                 for k in varying if k in it["constexprs"]]
        w, s = it.get("warps"), it.get("stages")
        parts.append(f"w{w}s{s}" if w not in (None, "") else "cfg-unknown")
        named["_".join(parts)].append(it)

    results = []
    for base, group in sorted(named.items()):
        sufs = _disambiguate(group) if len(group) > 1 else [""]
        for it, suf in zip(group, sufs, strict=True):
            results.append((it, f"{base}{suf}.ttgir"))
    return results, {k: a for k, a in ab.items() if a != k}


# --- dump -------------------------------------------------------------------

_NUM_WARPS_RE = re.compile(r'"ttg\.num-warps" = (\d+)')


def _compile_options(info) -> dict:
    """Triton's metadata json, made fit to recompile the ttir: warp
    specialization overwrites its num_warps with the total over all partitions
    (e.g. 12), which the num_warps option rejects. The ttgir module keeps the
    original as "ttg.num-warps"."""
    meta = dict(info["meta"])
    m = _NUM_WARPS_RE.search(info["src"].read_text())
    if m:
        meta["num_warps"] = int(m.group(1))
    return meta


def _dump_ir_guarded(cache_dir: str, run_dir: str) -> None:
    """Run the dump at exit, reporting rather than losing everything on an error.

    The dump runs from an atexit hook, after pytest has decided the run passed. An
    exception there (a path the filesystem rejects, a cache file another process
    removed) would otherwise take ttgir/, ttir/ and rows.tsv with it silently, and
    the run would still look complete -- the IR is only missed much later, when a
    gate or an A/B has nothing to attribute rows to.
    """
    import pathlib
    import shutil

    # The run dir may hold the dump of an earlier attempt (run_batch.py retries into the
    # same dir, e.g. without the dtype filter). Its files would sit next to this
    # attempt's as if this run had used those variants too.
    for name in ("ttgir", "ttir"):
        shutil.rmtree(pathlib.Path(run_dir) / name, ignore_errors=True)

    # rows.tsv first and on its own: it is what attributes result rows to kernels,
    # and it costs nothing, so a failure in the much larger IR copy below must not
    # take it down with it.
    try:
        _write_rows(pathlib.Path(run_dir) / "rows.tsv", cache_dir)
    except Exception:
        import traceback

        print(f"{tag(_TAG)} ROWS_TSV_FAILED {run_dir}: --gate cannot attribute this run's rows",
              file=sys.stderr, flush=True)
        traceback.print_exc()
    try:
        _dump_ir(cache_dir, run_dir)
    except Exception:
        import traceback

        print(f"{tag(_TAG)} IR_DUMP_FAILED {run_dir}: the run's numbers are fine, but its "
              "ttgir/ and ttir/ are missing or incomplete, so ir_gate.py cannot recompile "
              "its kernels and --gate counts its rows as unknown", file=sys.stderr, flush=True)
        traceback.print_exc()


def _dump_ir(cache_dir: str, run_dir: str) -> None:
    import pathlib
    import shutil

    cache = pathlib.Path(cache_dir)
    dest, ttir_dest = pathlib.Path(run_dir) / "ttgir", pathlib.Path(run_dir) / "ttir"
    if not cache.is_dir():
        print(f"{tag(_TAG)} no cache dir {cache}; nothing to dump", file=sys.stderr)
        return
    dest.mkdir(parents=True, exist_ok=True)

    def load(path):
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    infos = []
    for ttgir in sorted(cache.rglob("*.ttgir")):
        meta = load(ttgir.with_suffix(".json"))
        side = load(ttgir.parent / "__constexprs.json")
        infos.append({
            "src": ttgir, "hash": ttgir.parent.name, "kernel": ttgir.stem, "meta": meta,
            "warps": meta.get("num_warps"), "stages": meta.get("num_stages"),
            "constexprs": side.get("constexprs", {}), "attrs": side.get("attrs", {}),
            "signature": side.get("signature", {}),
            "per_shape": _launches.get(ttgir.parent.name, {}),
        })

    # (shape, kernel) -> variants really used there (launched outside the sweep)
    groups = defaultdict(list)
    for info in infos:
        for shape, (real, _sweep) in info["per_shape"].items():
            if real > 0:
                groups[(shape, info["kernel"])].append(info)

    rows, legends, copied = [], defaultdict(dict), 0
    for (shape, kernel), items in sorted(groups.items()):
        named, legend = _name_group(items)
        legends[kernel].update(legend)
        for it, fname in named:
            target = dest / shape / kernel / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(it["src"], target)
            copied += 1
            real, sweep = it["per_shape"][shape]
            rows.append((shape, kernel, fname, real, sweep,
                         it["warps"] or "", it["stages"] or "",
                         json.dumps(it["constexprs"], sort_keys=True), it["hash"]))

    kept_ttir = missing_ttir = 0
    for info in infos:
        if any(real > 0 for real, _ in info["per_shape"].values()):
            # Recompilation inputs: the ttir and Triton's metadata json, which
            # carries the compile options (num_warps, num_stages, target, ...).
            # Keyed by cache hash, the same key index.tsv records.
            ttir = info["src"].with_suffix(".ttir")
            if not (ttir.is_file() and info["meta"]):
                missing_ttir += 1  # e.g. TRITON_STORE_BINARY_ONLY=1
                continue
            out = ttir_dest / info["hash"]
            out.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ttir, out / ttir.name)
            (out / f"{info['kernel']}.json").write_text(json.dumps(_compile_options(info)))
            kept_ttir += 1
        else:
            # Compiled but never really used anywhere (autotune losers): index-only.
            sweep = sum(s for _, s in info["per_shape"].values())
            rows.append(("-", info["kernel"], "(sweep loser, not dumped)", 0, sweep,
                         info["warps"] or "", info["stages"] or "",
                         json.dumps(info["constexprs"], sort_keys=True), info["hash"]))

    rows.sort()
    with open(dest / "index.tsv", "w") as f:
        f.write("shape\tkernel\tfile\tlaunches\tsweep_launches\t"
                "num_warps\tnum_stages\tconstexprs\tcache_hash\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    lines = ["# ttgir naming legend", "",
             "Layout: `<shape>/<kernel>/<varying constexprs>_w{warps}s{stages}[__spec|__hash8].ttgir`",
             "Shape dirs mirror the benchmark shape yaml. Only constexprs that vary",
             "within a (shape, kernel) group appear; `index.tsv` has the full map.",
             "Values: `T`/`F` = true/false, `-` = none/null."]
    for kernel in sorted(legends):
        if not legends[kernel]:
            continue
        lines += ["", f"## {kernel}"]
        width = max(len(a) for a in legends[kernel].values())
        for full, a in sorted(legends[kernel].items(), key=lambda kv: kv[1]):
            lines.append(f"- `{a:<{width}}` = {full}")
    (dest / "naming.md").write_text("\n".join(lines) + "\n")

    shapes = sorted({r[0] for r in rows if r[0] != "-"})
    losers = sum(1 for r in rows if r[0] == "-")
    print(f">>> [Dump] ttgir -> {dest} ({copied} files across {len(shapes)} shapes; "
          f"{losers} unused variants index-only; legend: naming.md)", flush=True)
    for s in shapes:
        n = sum(1 for r in rows if r[0] == s)
        print(paint(f">>> [Dump]   {s}: {n}", DIM), flush=True)
    print(f">>> [Dump] ttir -> {ttir_dest} ({kept_ttir} used variants"
          + (f"; {missing_ttir} without ttir in the cache" if missing_ttir else "")
          + ")", flush=True)


def _fingerprint(cache_dir, name: str) -> str:
    """What a launch of the variant in <cache_dir>/<name> runs (_binary_id.fingerprint):
    kernel name, the cubin's code and the launch metadata. "?" when the cache does not
    hold them (another backend, a cache that was cleared)."""
    import pathlib

    try:
        (cubin,) = (pathlib.Path(cache_dir) / name).glob("*.cubin")
        meta = json.loads(cubin.with_suffix(".json").read_text())
        return _binary_id.fingerprint(cubin.stem, cubin.read_bytes(), meta)
    except Exception:
        return "?"


def _write_rows(path, cache_dir=None) -> None:
    """rows.tsv: one line per result-table row, numbered per (op_name, dtype) as
    the rows appear in run.log, with the cache hashes of the kernels it ran
    ("?" for a row that was not tracked), where they came from (scope) and, in the
    same order, the fingerprints of their binaries (ab_compare.py compares those
    between the two sides of an A/B; "-" for a row that launched no kernel).

    scope "timed": launched in the timed function. "row": nothing was launched in a
    timed function for this row (the benchmark times without triton.testing.do_bench,
    or _cudagraph_plugin is not loaded), so every launch made while it was current
    counts, or failing that every launch made for its input before the row existed.

    The one flow this cannot follow is a benchmark that makes several rows in one test
    without get_input_iter or build_inputs and times each with a timer of its own
    before making the row: nothing marks where one row's launches end, and they go to
    the row before. Upstream has no such benchmark."""
    seq: dict = defaultdict(int)
    prints: dict = {}

    def binaries(names) -> str:
        if cache_dir is None:
            return ""
        if names is None:
            return "?"
        if not names:
            return "-"
        for name in names:
            if name not in prints:
                prints[name] = _fingerprint(cache_dir, name)
        return ",".join(prints[name] for name in names)

    with open(path, "w") as f:
        f.write("op_name\tdtype\trow\tshape\tkernels\tscope\tbinaries\n")
        for op_name, dtype, rows in _tables:
            for row_id, shape in rows:
                i = seq[(op_name, dtype)]
                seq[(op_name, dtype)] += 1
                if row_id is None:
                    names, kernels, scope = None, "?", "-"
                elif _row_timed.get(row_id):
                    names, scope = sorted(_row_timed[row_id]), "timed"
                    kernels = ",".join(names)
                else:
                    names, scope = sorted(_row_any.get(row_id) or _row_before.get(row_id, ())), "row"
                    kernels = ",".join(names)
                f.write(f"{op_name}\t{dtype}\t{i}\t{shape}\t{kernels}\t{scope}\t{binaries(names)}\n")


# --- registration -----------------------------------------------------------

def pytest_configure(config):
    global _dump_enabled
    run_dir = os.environ.get(_DUMP_DIR_ENV, "").strip()
    if not run_dir:
        return  # e.g. a _parallel_warmup_plugin shard: IR comes from the real run
    cache_dir = os.environ.get("TRITON_CACHE_DIR", "").strip()
    if not cache_dir:
        print(f"{tag(_TAG)} warning: {_DUMP_DIR_ENV} is set but TRITON_CACHE_DIR "
              "is not; IR dump disabled", file=sys.stderr, flush=True)
        return

    import triton
    import triton.runtime.autotuner as _autotuner

    prev = triton.knobs.compilation.listener
    if prev is None:
        triton.knobs.compilation.listener = _compile_listener
    else:
        def chained(**kwargs):
            prev(**kwargs)
            _compile_listener(**kwargs)
        triton.knobs.compilation.listener = chained
    _shape_scope.listener = _on_shape
    triton.knobs.runtime.kernel_load_end_hook.add(_load_hook)
    triton.knobs.runtime.launch_enter_hook.add(_launch_hook)
    _autotuner.Autotuner._bench = _wrap_bench(_autotuner.Autotuner._bench)
    atexit.register(_dump_ir_guarded, cache_dir, run_dir)
    _dump_enabled = True
    print(f"{tag(_TAG)} shape/launch tracking on; IR dump -> {run_dir}/{{ttgir,ttir}}",
          file=sys.stderr, flush=True)
