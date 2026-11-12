"""pytest plugin: record or replay autotune configs across benchmark runs.

When comparing two runs (e.g. two compiler builds), replay the first run's
recorded configs in the second: the autotuner's choice is pinned, so the
latency delta reflects the change under test rather than the autotuner picking
a different config under different IR.

Modes (mutually exclusive, selected by env var; op identity from
FLAGGEMS_PERF_CURRENT_OP):
- FLAGGEMS_PERF_AUTOTUNE_RECORD_DIR: run full autotune, dump the chosen Config
  per (kernel, key) to <dir>/<op>.json at session end.
- FLAGGEMS_PERF_AUTOTUNE_REPLAY_DIR: pre-populate Autotuner.cache from
  <dir>/<op>.json, skipping the sweep. Missing kernel / missing key /
  non-compiling config / missing json fall back to full autotune and emit an
  AUTOTUNE_REPLAY_FALLBACK marker into the log.
"""

from __future__ import annotations

import atexit
import importlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_RECORD_DIR_ENV = "FLAGGEMS_PERF_AUTOTUNE_RECORD_DIR"
_REPLAY_DIR_ENV = "FLAGGEMS_PERF_AUTOTUNE_REPLAY_DIR"
_OP_ENV = "FLAGGEMS_PERF_CURRENT_OP"

# The two upstream repos ship different package names; either may host
# LibTuner-decorated kernels depending on which repo's benchmark is running.
_LIBTUNER_PKGS = ("flag_gems", "flaggems_vllm")

# Module-global aggregates, accumulated as run() is called and dumped at exit.
# Lock guards merges in case a kernel uses threads internally.
_record_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
_record_lock = threading.Lock()
_replay_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
_unrecorded: set = set()  # tuned kernels the replayed record has no entry for (said once each)
_dump_done = False
_wrap = None  # set in pytest_configure when record/replay is active


def _op_name() -> str:
    return os.environ.get(_OP_ENV, "").strip() or "default"


def _record_path(record_dir: str) -> Path:
    return Path(record_dir) / f"{_op_name()}.json"


def _kernel_id(tuner: Any) -> str:
    # Stable across runs: source module + qualified name. Shapes/keys split
    # by the inner key dict.
    name = (
        getattr(tuner, "__name__", None)
        or getattr(getattr(tuner, "base_fn", None), "__name__", None)
        or getattr(getattr(tuner, "fn", None), "__name__", None)
        or "unknown_kernel"
    )
    module = getattr(getattr(tuner, "fn", None), "__module__", "") or ""
    return f"{module}::{name}"


def _serialize_key(key: Tuple[Any, ...]) -> str:
    # Per-element repr, JSON-encoded into a flat string usable as a JSON key;
    # stable across runs given seeded RNG + yaml-fixed shapes.
    return json.dumps([repr(x) for x in key])


def _serialize_config(cfg: Any, allow_pre_hook: bool = False) -> Optional[Dict[str, Any]]:
    # pre_hook is an un-serializable callable. For plain triton Autotuner drop
    # such configs rather than replay without the hook and compute incorrectly.
    # For LibTuner callers pass allow_pre_hook=True: upstream's own ConfigCache
    # round-trips configs pre_hook-less and LibTuner.run re-attaches the hook by
    # matching all_kwargs against self.configs, and replay-injected configs go
    # through that same path (hopper mm's TMA configs carry a pre_hook even
    # with USE_FLAGTUNE=0 — dropping them would record nothing for those
    # kernels and silently defeat replay).
    if getattr(cfg, "pre_hook", None) is not None and not allow_pre_hook:
        return None
    try:
        entry = {
            "kwargs": dict(cfg.kwargs),
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
            "num_ctas": cfg.num_ctas,
            "maxnreg": cfg.maxnreg,
        }
    except Exception:
        return None
    # A config may replace a compile stage by a user-supplied IR file. Without it the
    # replay would compile the kernel from source: another binary under the same config.
    ir_override = getattr(cfg, "ir_override", None)
    if ir_override is not None:
        entry["ir_override"] = ir_override
    return entry


def _deserialize_config(entry: Dict[str, Any]) -> Any:
    # Late import (triton loads during collection, after plugin registration).
    from triton import Config
    extra = {"ir_override": entry["ir_override"]} if entry.get("ir_override") is not None else {}
    return Config(
        kwargs=dict(entry["kwargs"]),
        num_warps=entry.get("num_warps", 4),
        num_stages=entry.get("num_stages", 3),
        num_ctas=entry.get("num_ctas", 1),
        maxnreg=entry.get("maxnreg"),
        **extra,
    )


def _compute_key(tuner: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    # Reproduce Autotuner/LibTuner key computation; return None on any internals
    # shift (caller then just loses replay for that call).
    try:
        arg_names = tuner.arg_names
        nargs = dict(zip(arg_names, args, strict=False))  # trailing args may come as kwargs
        all_args = {**nargs, **kwargs}
        _args = {k: v for k, v in all_args.items() if k in arg_names}
        if hasattr(tuner, "get_key"):
            return tuner.get_key(_args)
        key = [_args[name] for name in tuner.keys if name in _args]
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        return tuple(key)
    except Exception:
        return None


def _emit_marker(reason: str) -> None:
    # Marker lands in run.log (grep it to audit replay coverage); flush so it
    # appears before the kernel runs. Yellow on the live terminal only —
    # run_pytest.sh strips ANSI from run.log.
    from _term_style import YELLOW, paint
    print(paint(f"AUTOTUNE_REPLAY_FALLBACK reason={reason}", YELLOW), flush=True)


def _record_run(original):
    # Read back this call's own key after run(), rather than diffing cache keys.
    # LibTuner.cache is backed by a persistent sqlite DB (flag_gems libcache), so
    # on a warm DB the key is already present and a before/after diff would come
    # up empty, recording nothing for any libtuner kernel.
    # Single-config kernels bypass the cache write, so fall back to configs[0].
    def runner(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        key = _compute_key(self, args, kwargs)
        if key is None:
            return result
        cfg = None
        try:
            cfg = self.cache[key]
        except Exception:
            cfg = None
        if cfg is None and len(getattr(self, "configs", []) or []) == 1:
            cfg = self.configs[0]
        if cfg is None:
            return result
        # get_key marks LibTuner (both flag_gems and flaggems_vllm); plain
        # triton Autotuner has no pre_hook re-attach on cache read, LibTuner does.
        entry = _serialize_config(cfg, allow_pre_hook=hasattr(self, "get_key"))
        if entry is not None:
            kid = _kernel_id(self)
            with _record_lock:
                _record_map.setdefault(kid, {})[_serialize_key(key)] = entry
        return result
    return runner


def _replay_run(original):
    # Inject recorded configs before benchmarking; on key-missing or a recorded
    # config failing to compile/launch, fall back to full autotune (+ marker).
    def runner(self, *args, **kwargs):
        injected_key = None
        if len(getattr(self, "configs", []) or []) > 1:
            kid = _kernel_id(self)
            bucket = _replay_map.get(kid)
            if bucket is None:
                # The record has nothing for this kernel: it tunes for itself, like a
                # kernel with a missing key. A record made by this plugin has every
                # tuned kernel the op ran, except those whose configs cannot be
                # stored (a pre_hook on a plain triton Autotuner). Said once per
                # kernel: it is the kernel that is missing, not one of its keys.
                if _replay_map and kid not in _unrecorded:
                    _unrecorded.add(kid)
                    _emit_marker("kernel_missing")
            else:
                key = _compute_key(self, args, kwargs)
                # Overwrite unconditionally: LibTuner.cache is backed by a
                # persistent sqlite DB, so gating on `key not in self.cache`
                # would skip injection whenever that DB is warm -- leaving the
                # run silently self-tuned instead of replaying.
                if key is not None:
                    rec = bucket.get(_serialize_key(key))
                    if rec is None:
                        _emit_marker("key_missing")
                    else:
                        try:
                            cfg = _deserialize_config(rec)
                            self.cache[key] = cfg
                            injected_key = key
                        except Exception as exc:
                            _emit_marker(f"deserialize_{type(exc).__name__}")
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:
            if injected_key is not None:
                # Recorded config failed under the current build: drop it and
                # let run() autotune. Only Autotuner.cache is a plain dict;
                # LibTuner.cache is a sqlite-backed ConfigCache with no
                # __delitem__, so eviction is impossible there -- say so in the
                # marker instead of retrying with the same bad config and
                # reporting a clean fallback.
                evicted = True
                try:
                    del self.cache[injected_key]
                except Exception:
                    evicted = False
                # _no_evict means the retry below re-reads the same recorded
                # config, so it is not a clean "fell back to live autotune":
                # treat those measurement points as unverified.
                suffix = "" if evicted else "_no_evict"
                _emit_marker(f"compile_{type(exc).__name__}{suffix}")
                return original(self, *args, **kwargs)
            raise
    return runner


def _load_replay_dir(replay_dir: str) -> int:
    path = _record_path(replay_dir)
    if not path.is_file():
        _emit_marker("record_missing")
        return 0
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        _emit_marker(f"load_{type(exc).__name__}")
        return 0
    if not isinstance(data, dict):
        return 0
    count = 0
    for kid, bucket in data.items():
        if not isinstance(bucket, dict):
            continue
        _replay_map[kid] = dict(bucket)
        count += len(bucket)
    return count


def _dump_record(record_dir: str) -> None:
    global _dump_done
    if _dump_done:
        return
    _dump_done = True
    path = _record_path(record_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _record_lock:
            payload = {kid: dict(bucket) for kid, bucket in _record_map.items()}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except Exception as exc:
        # Don't break the run if dump fails; the perf row is still produced.
        from _term_style import tag
        print(f"{tag('[autotune-record-plugin]')} dump failed: {exc}",
              file=sys.stderr, flush=True)


def pytest_configure(config):
    global _wrap
    from _term_style import tag
    record_dir = os.environ.get(_RECORD_DIR_ENV, "").strip()
    replay_dir = os.environ.get(_REPLAY_DIR_ENV, "").strip()
    if record_dir and replay_dir:
        # Mutually exclusive: recording while replaying is pointless. Surface loudly.
        print(f"{tag('[autotune-record-plugin]')} error: both RECORD_DIR and "
              "REPLAY_DIR set; ignoring both (no-op)", file=sys.stderr, flush=True)
        return
    if not record_dir and not replay_dir:
        return

    import triton.runtime.autotuner as _autotuner
    _wrap = _record_run if record_dir else _replay_run
    _autotuner.Autotuner.run = _wrap(_autotuner.Autotuner.run)

    if record_dir:
        # atexit (not sessionfinish): persist whatever was recorded even if an op
        # crash kills the session; a later replay run falls back for missing keys.
        atexit.register(_dump_record, record_dir)
        print(f"{tag('[autotune-record-plugin]')} recording autotune configs to "
              f"{_record_path(record_dir)} (Autotuner)",
              file=sys.stderr, flush=True)
    else:
        loaded = _load_replay_dir(replay_dir)
        print(f"{tag('[autotune-record-plugin]')} replaying {loaded} recorded entries from "
              f"{_record_path(replay_dir)} (Autotuner)",
              file=sys.stderr, flush=True)


def pytest_collection_finish(session):
    # LibTuner is patched here rather than in pytest_configure: only after
    # collection (which imports the benchmark module) do we know which FlagGems
    # package is actually in use, and importing the unused one just to patch it
    # would initialize a second runtime in this process for nothing.
    if _wrap is None:
        return
    from _term_style import tag
    patched = []
    for pkg in _LIBTUNER_PKGS:
        if pkg not in sys.modules:
            continue
        try:
            libentry = importlib.import_module(f"{pkg}.utils.libentry")
            tuner_cls = libentry.LibTuner
            if "run" in tuner_cls.__dict__:
                tuner_cls.run = _wrap(tuner_cls.__dict__["run"])
                patched.append(f"LibTuner[{pkg}]")
        except Exception as exc:
            print(f"{tag('[autotune-record-plugin]')} warning: LibTuner patch "
                  f"failed for {pkg}: {exc}", file=sys.stderr, flush=True)
    print(f"{tag('[autotune-record-plugin]')} libtuner coverage: "
          f"{', '.join(patched) or 'none (no FlagGems package imported?)'}",
          file=sys.stderr, flush=True)
