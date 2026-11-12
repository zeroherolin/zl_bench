"""pytest plugin: make the shape yaml (--shape_file) authoritative.

Upstream reads shapes in Benchmark.set_shapes() (op_name key -> MRO class-name
key -> DEFAULT_SHAPES), but a good share of benchmarks bypass it. This plugin
closes the three gaps. Each fix reuses the class's own input construction and
only swaps where the shapes come from, so the measured kernel is unchanged; ops
without a yaml entry keep their upstream shapes.

1. Shapes hardcoded in set_shapes() (e.g. fused_marlin_moe_*): wrap
   Benchmark.init_user_config, which runs right after set_shapes(), and put
   the yaml shapes back. The same after a subclass's own init_user_config
   that assigns self.shapes and never reads it (adaptive_max_pool3d); one
   that reads it filters or validates what the yaml gave and is left alone.
   Skipped when the subclass normalizes the arity after reading the yaml
   (BLAS (B,M,N,K) <-> (M,N,K)), detected as both sides being homogeneous in
   arity but different. Case-API benchmarks (get_case_iter + build_inputs)
   enumerate self.shapes, so this covers them.
2. get_input_iter overridden with a hardcoded list that never reads
   self.shapes (conv padding / pool / nll / scaled_softmax families): yield
   one input per yaml shape through the class's standard
   input_fn(shape, dtype, device). Classes without such an input_fn are left
   alone.
3. Bespoke inputs built from TestParam objects, config tables or a sampled
   kit, handled per class name (_BESPOKE_BUILDERS, _patch_cutlass_kit). These
   couple to upstream internals; a mismatch surfaces as a loud benchmark FAIL,
   not as silent corruption.
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import sys

import yaml

import _shape_scope
from _fg_benchmark import benchmark_classes
from _term_style import tag

_TAG = "[shape-override-plugin]"
_entries: dict = {}  # top-level yaml key (op_name or class name) -> entry
_benches: list = []  # (op_name, class, shapes) of every benchmark configured in this session
_own_inputs: set = set()  # op_names whose inputs came from a get_input_iter that reads no shapes


def _yaml_entry(op_name):
    entry = _entries.get(op_name)
    return entry if isinstance(entry, dict) and entry.get("shapes") else None


def _yaml_shapes(op_name):
    entry = _yaml_entry(op_name)
    return entry["shapes"] if entry else None


# --- 1. hardcoded set_shapes() ----------------------------------------------

def _arity(shape) -> int:
    return len(shape) if isinstance(shape, (list, tuple)) else 1


def _note(bench) -> None:
    """Record the benchmark's shapes as they stand; the last call per benchmark wins, i.e.
    the one after its outermost init_user_config."""
    note = (getattr(bench, "op_name", None), type(bench),
            [_freeze(s) for s in getattr(bench, "shapes", None) or []])
    index = getattr(bench, "_zl_shape_note", None)
    if index is None:
        bench._zl_shape_note = len(_benches)
        _benches.append(note)
    else:
        _benches[index] = note


# self.shapes anywhere but on the left of an assignment
_SHAPES_READ_RE = re.compile(r"\bself\.shapes\b(?!\s*=(?!=))")


def _only_assigns_shapes(func) -> bool:
    """The method sets self.shapes and never reads it: what it leaves there is
    hardcoded, whatever the yaml gave. False when the source is unavailable."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return False
    return "self.shapes" in source and not _SHAPES_READ_RE.search(source)


def _as_upstream(shapes) -> list:
    """The shapes of a yaml entry the way Benchmark.set_shapes reads them: a tuple
    each, a scalar as a one-element tuple."""
    return [tuple(s) if isinstance(s, (list, tuple)) else (s,) for s in shapes]


def _apply_entry(self) -> None:
    """Put the shapes of the benchmark's yaml entry into self.shapes."""
    entry = _yaml_entry(getattr(self, "op_name", None))
    if entry is None:
        return
    shapes = entry["shapes"]
    have = {_arity(s) for s in self.shapes or []}
    want = {_arity(s) for s in shapes}
    if len(have) == 1 and len(want) == 1 and have != want:
        print(f"{tag(_TAG)} keep subclass shapes for {self.op_name!r}: it "
              f"normalizes arity to {next(iter(have))}, yaml has "
              f"{next(iter(want))}", file=sys.stderr)
        return
    self.shapes = _as_upstream(shapes)
    if "shape_desc" in entry:
        self.shape_desc = entry["shape_desc"]


def _configured(original, hardcodes: bool = True):
    """An init_user_config that ends with the yaml entry in place (when the original
    hardcodes the shapes) and the benchmark noted for the end-of-session report. A
    subclass's own init_user_config is wrapped too: it may never call the base class."""
    @functools.wraps(original)
    def init_user_config(self):
        original(self)
        try:
            if hardcodes:
                _apply_entry(self)
        finally:
            _note(self)
    return init_user_config


def _patch_init_user_config(benchmark_cls) -> None:
    benchmark_cls.init_user_config = _configured(benchmark_cls.init_user_config)


# --- 2. hardcoded get_input_iter ----------------------------------------------

def _reads_self_shapes(func) -> bool:
    """Already driven by self.shapes, i.e. by fix 1. True when the source is
    unavailable: leave such a method alone."""
    try:
        return "self.shapes" in inspect.getsource(func)
    except (OSError, TypeError):
        return True


def _takes_shape_dtype_device(fn) -> bool:
    """The standard input_fn(shape, dtype, device) signature, as opposed to
    e.g. BLAS-style fn(b, m, n, k, dtype, device, ...)."""
    if not callable(fn):
        return False
    try:
        inspect.signature(fn).bind(None, None, None)
    except (TypeError, ValueError):
        return False
    return True


def _yaml_input_iter(original):
    def get_input_iter(self, dtype):
        shapes = _yaml_shapes(getattr(self, "op_name", None))
        input_fn = getattr(self, "input_fn", None)
        if shapes and _takes_shape_dtype_device(input_fn):
            for shape in _shape_scope.walk(shapes):
                yield from input_fn(tuple(shape), dtype, self.device)
        else:
            _own_inputs.add(getattr(self, "op_name", None))
            yield from original(self, dtype)
    return get_input_iter


# --- 3. bespoke input construction --------------------------------------------

def _upstream(original, name, builder):
    """A name the bespoke builder needs from the benchmark module it patches.

    These builders reach into upstream internals (a TestParam dataclass, a helper).
    When upstream renames or drops one, say which builder broke and on what, instead
    of a bare KeyError from a globals lookup deep inside an input iterator.
    """
    try:
        return original.__globals__[name]
    except KeyError:
        raise RuntimeError(
            f"{_TAG} {builder}: upstream no longer defines {name!r} in "
            f"{original.__globals__.get('__name__', '?')}; update _BESPOKE_BUILDERS in "
            "_shape_override_plugin.py, or drop the yaml entry to keep upstream's shapes"
        ) from None


def _upsample_builder(self, dtype, shapes, original):
    # _cfgs rows are (N, C, Hi, Wi, Ho, Wo, align_corners, label); the yaml
    # carries the 7 numbers.
    self._cfgs = [tuple(shape) + ("yaml",) for shape in shapes]
    yield from original(self, dtype)


def _fused_deepseek_builder(self, dtype, shapes, original):
    TestParam = _upstream(original, "TestParam", "_fused_deepseek_builder")
    for num_tokens, num_heads, num_tokens_insert, block_size, max_pos, eps in _shape_scope.walk(shapes):
        yield from type(self).make_input(TestParam(
            num_tokens=num_tokens, num_heads=num_heads,
            num_tokens_insert=num_tokens_insert, block_size=block_size,
            max_pos=max_pos, eps=eps))


def _flash_mla_sparse_builder(self, dtype, shapes, original):
    TestParam = _upstream(original, "TestParam", "_flash_mla_sparse_builder")
    for s_q, s_kv, topk, h_q, d_qk in _shape_scope.walk(shapes):
        yield from type(self).make_input_flashmla(TestParam(
            s_q=s_q, s_kv=s_kv, topk=topk, h_q=h_q, d_qk=d_qk,
            have_attn_sink=True))  # every upstream case uses True


_BESPOKE_BUILDERS = {
    "UpsampleBicubic2dAaBackwardBenchmark": _upsample_builder,
    "FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark": _fused_deepseek_builder,
    "FlashmlaSparseBenchmark": _flash_mla_sparse_builder,
}


def _bespoke_input_iter(original, builder):
    def get_input_iter(self, dtype):
        shapes = _yaml_shapes(getattr(self, "op_name", None))
        if shapes:
            yield from builder(self, dtype, shapes, original)
        else:
            yield from original(self, dtype)
    return get_input_iter


def _patch_cutlass_kit(benchmark_cls) -> bool:
    """CutlassScaledMMPerfKit samples cases from mnk x quant modes; swap mnk for
    the yaml (M, N, K) and keep upstream's quant-mode product as is."""
    shapes = _yaml_shapes("cutlass_scaled_mm")
    kit = getattr(sys.modules.get(benchmark_cls.__module__), "CutlassScaledMMPerfKit", None)
    if not shapes or kit is None:
        return False
    original = kit._get_all_combinations
    # Upstream returns product(mnk, *quant_modes); dedup the tails in order.
    quant_modes = list(dict.fromkeys(combo[1:] for combo in original()))

    def _get_all_combinations():
        return ((tuple(mnk),) + modes for mnk in shapes for modes in quant_modes)

    kit._get_all_combinations = staticmethod(_get_all_combinations)
    return True


# --- registration ---------------------------------------------------------------

def _load(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"{tag(_TAG)} cannot read {path}: {exc}", file=sys.stderr)
        return {}


def pytest_collection_finish(session):
    # After collection: every Benchmark subclass is defined by now, and
    # _ir_dump_plugin (trylast) wraps the get_input_iter installed here.
    _entries.update(_load(session.config.getoption("shape_file", default="")))
    from benchmark import base
    _patch_init_user_config(base.Benchmark)

    redirected, bespoke = 0, []
    for cls in benchmark_classes():
        if cls is not base.Benchmark and "init_user_config" in cls.__dict__:
            own_init = cls.__dict__["init_user_config"]
            cls.init_user_config = _configured(own_init, _only_assigns_shapes(own_init))
        own = cls.__dict__.get("get_input_iter")
        builder = _BESPOKE_BUILDERS.get(cls.__name__)
        if cls.__name__ == "CutlassScaledMMBenchmark" and _patch_cutlass_kit(cls):
            # its get_input_iter reads the kit, which now samples the yaml shapes
            bespoke.append("CutlassScaledMMPerfKit")
        elif own is not None and builder is not None:
            cls.get_input_iter = _bespoke_input_iter(own, builder)
            bespoke.append(cls.__name__)
        elif own is not None and not _reads_self_shapes(own):
            cls.get_input_iter = _yaml_input_iter(own)
            redirected += 1

    print(f"{tag(_TAG)} yaml shapes now override set_shapes(); "
          f"{redirected} hardcoded get_input_iter redirected; bespoke: "
          f"{', '.join(bespoke) or 'none'}", file=sys.stderr, flush=True)


def _freeze(value):
    """A shape, hashable: nested lists (a shape may be [input_shape, output_size]) as
    tuples, and whatever else a benchmark keeps in self.shapes (a TestParam dataclass)
    by its repr."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _shape_source(name, cls, have, test_name: str):
    """(where the shapes a benchmark measured came from, why that is a problem or None).

    name, cls and have are the benchmark's op_name, class and final self.shapes. The
    source goes to the log for every benchmark, so a run says which key drove it. A
    problem is only what the author of the shape file cannot have meant:

    - no entry applies at all, neither the op_name nor a class along the MRO: a curated
      set whose key does not (or no longer does) match the benchmark's op_name;
    - the op has an entry, but the benchmark put its own shapes back after reading it,
      or builds its inputs without reading shapes;
    - the file has an entry named after the test, but none for the op_name the benchmark
      uses: the key is on the wrong name.

    Falling back to a class-name key is upstream's design and no problem, even when the
    benchmark hardcodes its shapes and so does not read that key either: a whole-suite
    set has no entry for most ops, and flagging each would bury the cases above
    (gen_inventory.py --check-shapes lists the hardcoded ones worth an entry).
    """
    own = _yaml_entry(name)
    key = name if own else next((c.__name__ for c in cls.__mro__ if _yaml_entry(c.__name__)), None)
    wrong_name = None
    if not own and test_name != name and _yaml_entry(test_name):
        wrong_name = (f"{name} has no entry: {test_name!r} is the name of the test, the "
                      "benchmark's op_name is the key")
    if key is None:
        return f"{name}: its own, no entry applies", wrong_name or f"{name} has no entry"
    want = [_freeze(s) for s in _as_upstream(_entries[key]["shapes"])]
    kind = "key" if own else "class key"
    if name in _own_inputs:
        source = f"its own inputs, get_input_iter reads no shapes ({kind} {key!r} not used)"
        problem = f"{name} has an entry, but its get_input_iter builds inputs without reading shapes"
    elif set(have) <= set(want):
        return f"{name}: {len(have)} of the {len(want)} of {kind} {key!r}", wrong_name
    elif own and len(set(map(_arity, have))) == 1 and len(set(map(_arity, want))) == 1 \
            and _arity(have[0]) != _arity(want[0]):
        # fix 1 left them alone: the subclass derived them from the entry
        return f"{name}: {len(have)} derived from key {key!r} (other arity)", None
    else:
        source = f"its own {len(have)} ({kind} {key!r} not used)"
        problem = f"{name} has an entry, but the benchmark replaced it with shapes of its own"
    return f"{name}: {source}", problem if own else wrong_name


def pytest_sessionfinish(session, exitstatus):
    """Log where each benchmark's shapes came from, and report the ops that did not run
    the shapes the shape file gives them.

    Upstream renames an op_name now and then (fused_marlin_moe_w8a16_fp8 became
    fused_marlin_moe_w8a16). A set still keyed on the old name matches nothing,
    upstream falls back to the benchmark's own hardcoded shapes, and the run looks
    clean -- the numbers are simply for other shapes than the ones asked for.
    """
    if not _entries:
        return
    test_name = os.environ.get("FLAGGEMS_PERF_CURRENT_OP", "").strip()
    try:
        found = [_shape_source(*note, test_name) for note in _benches if note[0]]
    except Exception as exc:  # the report is a diagnostic; the measurement stands without it
        print(f"{tag(_TAG)} warning: cannot tell where the shapes came from "
              f"({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        return
    for source in dict.fromkeys(s for s, _ in found):
        print(f"{tag(_TAG)} shapes of {source}", file=sys.stderr, flush=True)
    problems = list(dict.fromkeys(p for _, p in found if p))
    if problems:
        print(f"{tag(_TAG)} SHAPE_KEYS_UNUSED {'; '.join(problems)} -- the benchmark's own "
              "shapes were measured, not the shape file's", file=sys.stderr, flush=True)
