"""pytest plugin: make the shape yaml (--shape_file) authoritative.

Upstream reads shapes in Benchmark.set_shapes() (op_name key -> MRO class-name
key -> DEFAULT_SHAPES), but a good share of benchmarks bypass it. This plugin
closes the three gaps. Each fix reuses the class's own input construction and
only swaps where the shapes come from, so the measured kernel is unchanged; ops
without a yaml entry keep their upstream shapes.

1. set_shapes() overridden with a hardcoded list (e.g. fused_marlin_moe_*):
   wrap Benchmark.init_user_config, which runs right after set_shapes(), and
   put the yaml shapes back. Skipped when the subclass normalizes the arity
   after reading the yaml (BLAS (B,M,N,K) <-> (M,N,K)), detected as both sides
   being homogeneous in arity but different. Case-API benchmarks
   (get_case_iter + build_inputs) enumerate self.shapes, so this covers them.
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

import inspect
import os
import sys

import yaml

from _fg_benchmark import benchmark_classes
from _term_style import tag

_TAG = "[shape-override-plugin]"
_entries: dict = {}  # top-level yaml key (op_name or class name) -> entry
_looked_up: set = set()  # keys an op asked for, to spot a set that matches nothing
_ops_run: set = set()    # op_names that ran, so only those are judged


def _yaml_entry(op_name):
    entry = _entries.get(op_name)
    if entry is not None:
        _looked_up.add(op_name)
    return entry if isinstance(entry, dict) and entry.get("shapes") else None


def _yaml_shapes(op_name):
    entry = _yaml_entry(op_name)
    return entry["shapes"] if entry else None


# --- 1. hardcoded set_shapes() ----------------------------------------------

def _arity(shape) -> int:
    return len(shape) if isinstance(shape, (list, tuple)) else 1


def _patch_init_user_config(benchmark_cls) -> None:
    original = benchmark_cls.init_user_config

    def init_user_config(self):
        original(self)
        _ops_run.add(getattr(self, "op_name", None))
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
        # Same normalization as Benchmark.set_shapes, scalars included.
        self.shapes = [tuple(s) if isinstance(s, (list, tuple)) else (s,)
                       for s in shapes]
        if "shape_desc" in entry:
            self.shape_desc = entry["shape_desc"]

    benchmark_cls.init_user_config = init_user_config


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
            for shape in shapes:
                yield from input_fn(tuple(shape), dtype, self.device)
        else:
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
    for num_tokens, num_heads, num_tokens_insert, block_size, max_pos, eps in shapes:
        yield from type(self).make_input(TestParam(
            num_tokens=num_tokens, num_heads=num_heads,
            num_tokens_insert=num_tokens_insert, block_size=block_size,
            max_pos=max_pos, eps=eps))


def _flash_mla_sparse_builder(self, dtype, shapes, original):
    TestParam = _upstream(original, "TestParam", "_flash_mla_sparse_builder")
    for s_q, s_kv, topk, h_q, d_qk in shapes:
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
        own = cls.__dict__.get("get_input_iter")
        builder = _BESPOKE_BUILDERS.get(cls.__name__)
        if own is not None and builder is not None:
            cls.get_input_iter = _bespoke_input_iter(own, builder)
            bespoke.append(cls.__name__)
        elif own is not None and not _reads_self_shapes(own):
            cls.get_input_iter = _yaml_input_iter(own)
            redirected += 1
        if cls.__name__ == "CutlassScaledMMBenchmark" and _patch_cutlass_kit(cls):
            bespoke.append("CutlassScaledMMPerfKit")

    print(f"{tag(_TAG)} yaml shapes now override set_shapes(); "
          f"{redirected} hardcoded get_input_iter redirected; bespoke: "
          f"{', '.join(bespoke) or 'none'}", file=sys.stderr, flush=True)


def pytest_sessionfinish(session, exitstatus):
    """Warn when an op that ran found no entry in the shape file.

    Upstream renames an op_name now and then (fused_marlin_moe_w8a16_fp8 became
    fused_marlin_moe_w8a16). A set still keyed on the old name matches nothing,
    upstream falls back to the benchmark's own hardcoded shapes, and the run looks
    clean -- the numbers are simply for other shapes than the ones asked for. A
    whole-suite set legitimately holds keys for ops this session did not run, so
    only the ops that ran are judged.
    """
    ran = sorted(n for n in _ops_run if n)
    if not _entries or not ran:
        return
    missed = [n for n in ran if n not in _looked_up]
    if not missed:
        return
    near = sorted(k for k in _entries if any(n[:12] in k or k[:12] in n for n in missed))
    hint = f" (similar keys present: {', '.join(near[:5])})" if near else ""
    print(f"{tag(_TAG)} SHAPE_KEYS_UNUSED the shape file has no entry for {', '.join(missed)}"
          f"{hint} -- these ops ran on their own hardcoded shapes", file=sys.stderr, flush=True)
