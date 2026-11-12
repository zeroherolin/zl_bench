"""pytest plugin: cut wall time by doing the autotune sweep on N GPUs in parallel.

Why: for shape-heavy ops the sweep dominates, not the measurement. Measured on
fused_marlin_moe_w4a16_mxfp4 / 53 shapes: 3755s total, of which the summed kernel
time is 49ms; ttgir/index.tsv holds 2858 compiled variants, 2645 of them `sweep
loser`. The same 53 shapes replayed (configs pinned, no sweep) take 196s. The
sweep is ~19x the measurement, so that is the part worth parallelizing.

What this does NOT parallelize: the measurement. N processes hammering N GPUs of
one box couple through the power/thermal budget, so per-card latency gets dragged
by whatever the neighbours are doing, by an amount that does not reproduce. This
plugin only parallelizes "decide which config wins", then hands the configs to
the normal single-process single-GPU path, which times things exactly as it would
without the plugin.

Enable with PARALLEL_WARMUP_GPUS=<N>; unset (or <2) makes the plugin a no-op, so
`-p _parallel_warmup_plugin` can stay in the plugin list permanently. Every entry of
the shape set is split between the GPUs, so the benchmark is covered whichever entry
it reads (its op_name, which need not be the test's name, or a class-name key).

No new output structure: the merged configs land in the run's own
autotune_records/<op>.json — the same path record mode writes and REPLAY_FROM
reads — and the sweep's scratch dirs are deleted. The run just finishes sooner,
plus a few log lines. (It is the config actually used for the measurement, so the
artifact stays faithful to what was run.)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

_GPUS_ENV = "PARALLEL_WARMUP_GPUS"
_WORKER_ENV = "FLAGGEMS_PERF_WARMUP_WORKER"
_RECORD_DIR_ENV = "FLAGGEMS_PERF_AUTOTUNE_RECORD_DIR"
_REPLAY_DIR_ENV = "FLAGGEMS_PERF_AUTOTUNE_REPLAY_DIR"
_OP_ENV = "FLAGGEMS_PERF_CURRENT_OP"


def _op_name() -> str:
    return os.environ.get(_OP_ENV, "").strip() or "default"


def _requested_gpus() -> int:
    """N from the env, clamped to what the box has. 0 disables."""
    raw = os.environ.get(_GPUS_ENV, "").strip()
    if not raw or os.environ.get(_WORKER_ENV):
        return 0
    try:
        want = int(raw)
    except ValueError:
        want = -1
    if want in (0, 1):
        return 0  # the documented way to turn it off
    if want < 2:
        # asked for and not given: say so, or the run just takes as long as before
        from _term_style import tag
        print(f"{tag('[parallel-warmup-plugin]')} {_GPUS_ENV}={raw!r} is not a GPU count of 2 or "
              "more; sweeping serially", flush=True)
        return 0
    try:
        import torch
        have = torch.cuda.device_count()
    except Exception:
        return 0
    return max(0, min(want, have))


def _visible_devices() -> List[str]:
    """The device ids a shard may be pinned to, in parent-visible order.

    Must respect an inherited CUDA_VISIBLE_DEVICES: writing a bare shard index
    into the child would re-interpret it as an absolute id, so a caller who
    picked idle cards (CUDA_VISIBLE_DEVICES=5,6) would silently get cards 0,1 —
    exactly the busy-GPU case the caller was avoiding.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        return [d.strip() for d in raw.split(",") if d.strip()]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _read_shape_yaml(path: str) -> Dict[str, Any]:
    """The entries of the shape set that have shapes."""
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    return {key: entry for key, entry in doc.items()
                if isinstance(entry, dict) and entry.get("shapes")}


def _shard(shapes: List[Any], n: int) -> List[List[Any]]:
    """Round-robin, not contiguous blocks.

    Shape lists are usually sorted ascending, so a block split hands one worker
    every big (slowest-to-compile) shape and the wall time collapses to that
    worker. Round-robin spreads compile cost evenly.
    """
    return [s for s in (shapes[i::n] for i in range(n)) if s]


def _shard_entries(entries: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """The shape set cut into n sets, each entry's shapes dealt round-robin.

    Every entry is cut, not just the one named after the test: which entry a benchmark
    reads is its op_name, which need not be the test's name (FlagGems-vllm's
    test_fused_marlin_moe_w4a16_mxfp4 reads fused_marlin_moe_w4a16), or a class-name
    key. Cut the same way, whichever entry it reads is covered by the shards between
    them.

    An entry with fewer shapes than shards goes to the first shards only. A shard is
    never left without an entry the others have: upstream would then fall back to the
    next key (a class name) or to the benchmark's own shapes, and sweep shapes nobody
    asked for. Such a shard repeats the entry's first shape instead: a config tuned
    twice, which the merge counts as a conflict only if the winners differ.
    """
    shards: List[Dict[str, Any]] = [{} for _ in range(n)]
    for key, entry in entries.items():
        parts = _shard(list(entry["shapes"]), n)
        for i in range(n):
            shards[i][key] = dict(entry, shapes=parts[i] if i < len(parts) else parts[0][:1])
    return shards


def _spawn(shards: List[Dict[str, Any]], scratch: Path, argv: List[str], devices: List[str]
           ) -> Tuple[List[Path], List[int]]:
    """One subprocess per shard, each pinned to its own GPU.

    Subprocesses rather than pytest-xdist: importing FlagGems initializes a
    CUDA context (its vendor probe queries torch.cuda device properties), which
    does not survive fork-based parallelism.
    """
    import yaml
    procs: List[Tuple[int, subprocess.Popen, Any]] = []
    recs: List[Path] = []
    for i, shard in enumerate(shards):
        w = scratch / f"w{i}"
        w.mkdir(parents=True, exist_ok=True)
        (w / "shapes.yaml").write_text(
            yaml.safe_dump(shard, sort_keys=False, default_flow_style=None, allow_unicode=True))
        rec = w / "rec"
        rec.mkdir(exist_ok=True)

        child_argv: List[str] = []
        skip_next = False
        for a in argv:
            if skip_next:
                skip_next = False
                continue
            if a == "--shape_file":
                skip_next = True
                continue
            if a.startswith("--shape_file="):
                continue
            child_argv.append(a)
        child_argv += ["--shape_file", str(w / "shapes.yaml")]

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = devices[i]
        env[_WORKER_ENV] = "1"                    # stops recursion
        env[_RECORD_DIR_ENV] = str(rec)           # shard records its own picks
        env.pop(_REPLAY_DIR_ENV, None)            # a shard must sweep
        env.pop(_GPUS_ENV, None)
        env["TRITON_CACHE_DIR"] = str(w / ".triton_cache")
        env.pop("FLAGGEMS_PERF_IR_DUMP_DIR", None)  # IR comes from the real run
        env["FLAGGEMS_PERF_COLOR"] = "never"

        log = open(w / "worker.log", "w")
        procs.append((i, subprocess.Popen(
            [sys.executable, "-u", "-m", "pytest", *child_argv],
            stdout=log, stderr=subprocess.STDOUT, env=env), log))
        recs.append(rec)

    try:
        for _, p, _log in procs:
            p.wait()
    except BaseException:
        # Ctrl-C (or anything else) must not leave N children holding GPUs:
        # terminate, then reap, then let the exception continue.
        for _, p, _log in procs:
            if p.poll() is None:
                p.terminate()
        for _, p, _log in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        raise
    finally:
        for _, _p, log in procs:
            log.close()
    return recs, [i for i, p, _ in procs if p.returncode != 0]


def _merge(recs: List[Path], op: str) -> Tuple[Dict[str, Any], int, int]:
    """Union the shards' {kernel: {key: config}} maps.

    A key present in several shards with *different* values means that key's
    winner moved under parallel interference; count those — the count is how much
    to trust this warmup. First value wins.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    conflicts = 0
    for rec in recs:
        p = rec / f"{op}.json"
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for kernel, bucket in data.items():
            if not isinstance(bucket, dict):
                continue
            tgt = merged.setdefault(kernel, {})
            for key, cfg in bucket.items():
                if key in tgt:
                    conflicts += tgt[key] != cfg
                    continue
                tgt[key] = cfg
    return merged, sum(len(b) for b in merged.values()), conflicts


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Do the parallel sweep here, before _autotune_record_plugin configures.

    That plugin decides record-vs-replay and patches Autotuner.run inside its own
    pytest_configure, so the swap to replay has to be in place before it runs —
    hence tryfirst. Ordering within the -p list is not relied upon.
    """
    from _term_style import tag
    gpus = _requested_gpus()
    if not gpus:
        return

    op = _op_name()
    shape_file = getattr(config.option, "shape_file", "") or ""
    if not shape_file or not Path(shape_file).is_file():
        print(f"{tag('[parallel-warmup-plugin]')} no --shape_file; "
              "parallel warmup needs an explicit shape set, skipping", flush=True)
        return
    if os.environ.get(_REPLAY_DIR_ENV, "").strip():
        print(f"{tag('[parallel-warmup-plugin]')} REPLAY_FROM is set; configs are "
              "already pinned, nothing to sweep, skipping", flush=True)
        return

    try:
        entries = _read_shape_yaml(shape_file)
    except Exception as exc:
        print(f"{tag('[parallel-warmup-plugin]')} cannot read shapes "
              f"({type(exc).__name__}: {exc}); skipping", flush=True)
        return
    most = max((len(e["shapes"]) for e in entries.values()), default=0)
    if most < 2:
        # e.g. the single-shape set: every entry would go to the first shard whole
        print(f"{tag('[parallel-warmup-plugin]')} no entry of the shape set has two shapes "
              "to split between GPUs; sweeping serially", flush=True)
        return

    devices = _visible_devices()
    if len(devices) < 2:
        print(f"{tag('[parallel-warmup-plugin]')} only {len(devices)} visible "
              "GPU(s); nothing to parallelize, skipping", flush=True)
        return

    shards = _shard_entries(entries, min(gpus, most, len(devices)))
    scratch = Path(tempfile.mkdtemp(prefix=f"warmup_{op}_"))
    print(f"{tag('[parallel-warmup-plugin]')} sweeping on {len(shards)} GPUs: every entry of "
          f"the shape set is split between them (up to {most} shapes); this phase's "
          "latency is discarded, only configs are kept", flush=True)

    keep_scratch = False
    t0 = time.time()
    try:
        recs, bad = _spawn(shards, scratch, list(config.invocation_params.args), devices)
        merged, total, conflicts = _merge(recs, op)
        elapsed = time.time() - t0

        if not total:
            keep_scratch = True
            print(f"{tag('[parallel-warmup-plugin]')} no configs recovered in "
                  f"{elapsed:.0f}s; falling back to normal serial autotune "
                  f"(worker logs kept in {scratch})", flush=True)
            return

        # Publish the merged configs where record mode would have written them,
        # so the artifact layout is unchanged and REPLAY_FROM still works. Then
        # point the record plugin at that file in replay mode: the serial
        # measurement below reuses these configs instead of sweeping again.
        rec_dir = os.environ.get(_RECORD_DIR_ENV, "").strip()
        out = Path(rec_dir) if rec_dir else (scratch / "merged")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{op}.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
        if not rec_dir:
            keep_scratch = True  # nowhere else to keep the configs
        os.environ.pop(_RECORD_DIR_ENV, None)   # record+replay is rejected
        os.environ[_REPLAY_DIR_ENV] = str(out)

        msg = (f"{tag('[parallel-warmup-plugin]')} sweep done in {elapsed:.0f}s: "
               f"{len(merged)} kernels / {total} configs -> {out / f'{op}.json'}; "
               "measurement continues serially on one GPU")
        if bad:
            msg += f" (WARNING shard(s) {bad} exited non-zero; missing keys will "
            msg += "fall back to live autotune)"
        print(msg, flush=True)
        if conflicts:
            print(f"{tag('[parallel-warmup-plugin]')} WARNING {conflicts} config "
                  "key(s) got different winners across shards — parallel "
                  "interference reached the sweep. Treat this run as a rough "
                  f"pass; unset {_GPUS_ENV} for a clean baseline", flush=True)
    except Exception as exc:
        keep_scratch = True
        print(f"{tag('[parallel-warmup-plugin]')} warmup failed "
              f"({type(exc).__name__}: {exc}); falling back to normal serial "
              f"autotune (scratch kept in {scratch})", flush=True)
    finally:
        if not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
