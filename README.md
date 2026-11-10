<div align="center">

# ZL_BENCH — Operator-level A/B benchmarking of Triton compiler changes on FlagGems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Triton 3.6 (FlagTree)](https://img.shields.io/badge/Triton-3.6%20%28FlagTree%29-6E40C9)](https://github.com/flagos-ai/FlagTree)
[![GPU: NVIDIA Hopper](https://img.shields.io/badge/GPU-NVIDIA%20Hopper-76B900?logo=nvidia&logoColor=white)](#requirements)
[![Built on FlagGems](https://img.shields.io/badge/built%20on-FlagGems-0969DA)](https://github.com/flagos-ai/FlagGems)
[![Built on FlagGems-vllm](https://img.shields.io/badge/built%20on-FlagGems--vllm-0969DA)](https://github.com/flagos-ai/FlagGems-vllm)
[![ShellCheck](https://img.shields.io/badge/shell-ShellCheck-4EAA25?logo=gnubash&logoColor=white)](https://www.shellcheck.net)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](tests)

**English** | [简体中文](README.zh-CN.md)

zl_bench is a set of pytest plugins and driver scripts for the benchmark suites of [FlagGems](https://github.com/flagos-ai/FlagGems) and [FlagGems-vllm](https://github.com/flagos-ai/FlagGems-vllm). It measures how a change to the [FlagTree](https://github.com/flagos-ai/FlagTree) Triton compiler affects operator performance, with the noise sources that usually swamp such a comparison under control.

</div>

## Contents

- [Features](#features)
- [Tested versions](#tested-versions)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Keeping up with upstream](#keeping-up-with-upstream)
- [FAQ](#faq)
- [Development](#development)
- [License](#license)

## Features

- **Same configs on both sides.** Record/replay pins the choices of Triton autotune and FlagGems libtuner, so an A/B measures the compiler change and not a different config. Without it, two runs of identical code can differ by around 10%.
- **Low-noise timing.** Kernels are timed with CUDA graphs after an explicit warmup, keeping launch overhead and first-call compilation out of the numbers. Calls that cannot be captured, such as ones that synchronize with the host, fall back to `do_bench` and are marked in the log.
- **IR for every measurement.** Each run keeps the ttgir it actually used, per shape and with readable names, the ttir and compile options for offline recompilation, and which kernels each result row ran.
- **Compile-diff gate.** Recompile the recorded ttir with two compiler builds down to cubin and A/B only the result rows whose kernels changed: one to two orders of magnitude faster than a full run.
- **Statistical A/B.** Both sides run interleaved on one GPU over repeated rounds; each row gets a rank-test verdict, undecided rows get more rounds, and rows whose binary did not change serve as controls that measure the false-positive rate.
- **Batch screening.** Run the 1600+ benchmarks of both repos across GPUs, with timeouts, retries, resume and a per-row summary.
- **Authoritative shapes.** One yaml decides the shapes of every op, including benchmarks that hardcode theirs.

How it works: [docs/design.md](docs/design.md).

## Tested versions

zl_bench patches upstream internals, so it is tied to the upstream versions it was tested with. Everything in this README, including the [reference full run](#batch-runs) and `ops/dismiss.txt`, refers to these commits:

| Component | Commit | Date | Source |
|---|---|---|---|
| FlagGems | [`a4e2556d5`](https://github.com/flagos-ai/FlagGems/commit/a4e2556d5) | 2026-11-04 | `flagos-ai/FlagGems` master |
| FlagGems-vllm | [`a579b7c`](https://github.com/flagos-ai/FlagGems-vllm/commit/a579b7c) | 2026-11-10 | `flagos-ai/FlagGems-vllm` main |
| FlagTree | [`6d7597f66`](https://github.com/flagos-ai/FlagTree/commit/6d7597f66) | 2026-11-10 | `flagos-ai/FlagTree` main |

Environment: NVIDIA H20 (Hopper, driver 595.45), CUDA 13.0, PyTorch 2.11.0, Python 3.12, vLLM 0.20.2. With other upstream versions, rebuild the inventory and refresh the shape sets first ([Keeping up with upstream](#keeping-up-with-upstream)); each `run.log` records the Triton path and git revision it ran with.

## Requirements

- An NVIDIA GPU (developed and tested on H20, Hopper)
- Python 3.10+ and PyTorch
- A FlagTree build of Triton 3.6 (the plugins use `knobs.compilation.listener` and the kernel launch hooks)
- Checkouts of FlagGems and/or FlagGems-vllm

## Installation

Put zl_bench next to the FlagGems checkouts; if they live elsewhere, point `FLAGGEMS_MAIN_DIR` / `FLAGGEMS_VLLM_DIR` at them.

```
<workspace>/
├── zl_bench/
├── FlagGems/          # package flag_gems
└── FlagGems-vllm/     # package flaggems_vllm
```

```bash
cd <workspace>
pip install -r zl_bench/requirements.txt
pip install -e FlagGems --no-deps         # --no-deps leaves FlagTree's triton alone
pip install -e FlagGems-vllm --no-deps
```

The compiler under test is whatever `import triton` resolves to. To test a FlagTree build without installing it, prepend its `python/` dir: `PYTHONPATH=<FlagTree>/python`.

## Quick start

Check that the GPU is idle (`nvidia-smi`), then benchmark one op:

```bash
cd zl_bench
OP=mm OP_FILE=mm DTYPES=bfloat16 SHAPE_FILE=shape/all_ops_single.yaml bash run_pytest.sh
```

```
>>> op=mm  mode=record  USE_FLAGTUNE=0
...
Operator: mm  (dtype=torch.bfloat16, mode=kernel, level=core)
Status     Torch Latency (ms)   Gems Latency (ms)   Gems Speedup         TFLOPS  Size Detail
--------------------------------------------------------------------------------------------
SUCCESS              0.003182            0.003061          1.040         36.998  [torch.Size([384, 384]), torch.Size([384, 384])]
>>> done. outputs in <zl_bench>/runs/mm_<timestamp>
```

The run dir holds the log, the autotune configs that were chosen and the IR ([Outputs](#outputs)). This was a **record** run: it tuned live, and two record runs are not comparable with each other. To compare compilers, follow [A/B comparison](#ab-comparison).

## Usage

### Selecting an op

A benchmark is `$FLAGGEMS_DIR/benchmark/test_<OP_FILE>.py::test_<OP>`: `OP` is the test function name without `test_`, `OP_FILE` the file name without `test_` and `.py` (subdirs such as `FLA/test_xxx` included). `ops/inventory_main.csv` and `ops/inventory_vllm.csv` list every op with its `op_file`.

```bash
OP=softmax OP_FILE=softmax bash run_pytest.sh                            # FlagGems
FLAGGEMS_DIR=../FlagGems-vllm OP=fused_marlin_moe_w4a16_mxfp4 \
  OP_FILE=fused_marlin_moe_w4a16_mxfp4 bash run_pytest.sh                # FlagGems-vllm
SHAPE_FILE=shape/DeepSeek-V4-Flash-p32768d1024.yaml bash run_pytest.sh   # another shape set
```

Both repos have ops with the same name but different implementations; `FLAGGEMS_DIR` decides which one runs. Measurement is always one process on one GPU, which is what makes timings comparable; pick the GPU with `CUDA_VISIBLE_DEVICES`. The top of `run.log` records the triton path and git revision and the `FLAGTREE_*`/`TRITON_*`/`MLIR_*` variables, so the two sides of an A/B can be reconciled.

### A/B comparison

Record on the baseline compiler, then replay the recorded configs on the changed one:

```bash
# A: baseline compiler, records its autotune choices
PYTHONPATH=/path/FlagTree-base/python bash run_pytest.sh             # -> runs/<op>_<ts_A>/
# B: changed compiler, replays A's choices
PYTHONPATH=/path/FlagTree-dev/python REPLAY_FROM=$PWD/runs/<op>_<ts_A> bash run_pytest.sh
# Gems latency per row, ratio A/B (>1: B is faster)
python ab_compare.py runs/<op>_<ts_A> runs/<op>_<ts_B>
```

This single comparison is a quick look. GPUs of one box differ systematically by a few percent (3–4% on an 8×H20 box) and single rows are noisy, so draw conclusions from [repeated rounds](#repeated-measures-ab).

- B must use `REPLAY_FROM`: two record runs tune independently and can pick different configs, a difference large enough to hide the change under test ([why](docs/design.md#why-replay)).
- B reuses A's archived `shapes.yaml` unless `SHAPE_FILE` is set, so refreshing the shape sets does not break old baselines.
- `USE_FLAGTUNE` must be the same on both sides.
- Point `REPLAY_FROM` at the original record: replay runs write no `autotune_records/`.
- For IR differences, diff `ttgir/<shape>/<kernel>/` of the two runs.

**Read the markers in `run.log`.** `AUTOTUNE_REPLAY_FALLBACK` means a recorded config was missing or failed to compile and that point was tuned live, so it no longer uses A's config (`ab_compare.py` flags such ops `[replay_fallback]`); with the `_no_evict` suffix the retry reused the bad config, so treat the point as unverified. `BENCHMARK_DIRECT_NO_CUDAGRAPH` marks points timed with plain `do_bench` because the call cannot be graph captured (`phase=host_sync`: it synchronizes with the host); both sides fall back alike, but those numbers are noisier, especially for sub-millisecond kernels ([details](docs/design.md#timing)).

### Repeated-measures A/B

`ab_run.py` replays one record (a run or batch dir made in record mode, usually with the A side's compiler) on two compiler setups for several rounds; `ab_compare.py` turns the rounds into a verdict per result row.

```bash
# A side and B side: a PYTHONPATH each, or one build with a compile switch (--a-env / --b-env)
python ab_run.py --record runs/<record> --ops softmax mm \
  --a-pythonpath /path/FlagTree-base/python --b-pythonpath /path/FlagTree-dev/python \
  --rounds 6 --until-decided --out runs/ab_dev
python ab_compare.py runs/ab_dev -v --csv runs/ab_dev/rows.csv
```

- Each (op, round) runs A and B back to back on one GPU, A first in odd rounds and B first in even ones, so per-GPU bias and slow drift cancel. Items share one queue over all GPUs, longest first; every run is a fresh process, so rounds are independent samples. Each run gets the same handling as in a [batch](#batch-runs) (timeout, retries, idle-GPU check). `<out>/r<k>_a` and `r<k>_b` are laid out like batch dirs; rerunning with the same `--out` resumes.
- Verdict per row ([details](docs/design.md#statistics)): `faster` / `slower` when the medians differ by more than `--threshold` (0.5%) and a two-sided Mann-Whitney test gives p < `--alpha` (0.01) over at least `--min-rounds` (6) rounds; `same` within the threshold or at p ≥ `--clear` (0.1); otherwise `undecided`. `--until-decided` adds `--step` rounds to ops with undecided rows up to `--max-rounds` (14). A row left `undecided` is decided on the pairing when every round points the same way and the gap clears `--threshold`: the rounds run both sides back to back on one GPU, so a pair cancels the GPU and the drift the unpaired test must absorb (six agreeing pairs are a two-sided sign test at p = 0.031, `--alpha-paired`; the unpaired test over 6 vs 6 samples needs them to separate completely). The CSV carries `p_paired`.
- `gap = median(A) / median(B) − 1`, positive when B is faster. It is the effect of the compiler change under the record's configs, not of each side at its own best ([why](docs/design.md#each-side-at-its-best)).
- With `--gate <ir_gate out>` (next section), rows whose kernels compiled to the same binary on both sides are controls: they cannot differ, so `ab_compare.py` reports how many of them the rule calls faster or slower, the measured false-positive rate.
- **Recheck weak results on fresh rounds.** A row with a small or noisy gap was selected and judged on the same samples, which overstates it. `ab_compare.py --recheck-list` writes the ops with such rows (|gap| < 2%, |gap| < 2× the samples' coefficient of variation, or do_bench timing); `ab_run.py --add-rounds 6` gives them new rounds numbered after the existing ones; `ab_compare.py --confirm <first>-<last>` (the range `ab_run.py` prints) judges every row on the other rounds and keeps a faster/slower row only if the new rounds alone reproduce it (one-sided, p < 0.05):

  ```bash
  python ab_compare.py runs/ab_dev --gate runs/gate_dev --recheck-list runs/ab_dev/recheck.txt
  python ab_run.py <same sides and record> --out runs/ab_dev --ops-file runs/ab_dev/recheck.txt --add-rounds 6
  python ab_compare.py runs/ab_dev --gate runs/gate_dev --confirm 15-20 --csv runs/ab_dev/rows.csv
  ```

  Run `--until-decided` before the recheck: once fresh rounds exist, analyze with `--confirm` so they stay separate.

### Compiler changes: gate first

After a FlagTree pass change most kernels still compile to the same binary, so a full A/B is slow and mostly measures noise. Gate first:

```bash
# 1. Record a baseline batch once, on the multi-shape set; every run dir keeps ttir/ and rows.tsv
PYTHONPATH=/path/FlagTree-base/python python run_batch.py --all --shape-file shape/all_ops_multi.yaml \
  --dtypes "bfloat16 float16" --op-timeout 7200 --batch-dir runs/base_multi

# 2. Recompile the recorded ttir with both builds down to cubin (no GPU, minutes)
python ir_gate.py runs/base_multi --base /path/FlagTree-base/python --new /path/FlagTree-dev/python \
  --out runs/gate_dev
#    One build, two compile switches: same path for both, plus --base-env FLAGTREE_XXX=0 --new-env FLAGTREE_XXX=1

# 3. Repeated-measures A/B of the ops with a changed row, replaying the baseline configs
python ab_run.py --record runs/base_multi --ops-file runs/gate_dev/changed_ops.txt \
  --a-pythonpath /path/FlagTree-base/python --b-pythonpath /path/FlagTree-dev/python \
  --dtypes "bfloat16 float16" --gate runs/gate_dev --until-decided --out runs/ab_dev
python ab_compare.py runs/ab_dev --gate runs/gate_dev -v --csv runs/ab_dev/rows.csv
```

- Record on the multi-shape set. Kernel variants depend on the shape; a variant only a second shape uses is invisible in a single-shape record, so the gate can only clear what was recorded.
- The gate recompiles the variants the measurement used and compares the cubin and the launch metadata (shared memory, warps). Identical means the measured kernel is the same binary, so there is nothing to measure. TTGIR is not enough: instruction order and later stages change the binary without changing the TTGIR ops. `variants.tsv` shows, per variant, the first stage that differs, whether the TTGIR changed, was only reordered or stayed the same, and the TTGIR ops whose count changed. `--stage ttgir` stops at TTGIR for a quicker first look; without `--base` the recorded ttgir is the baseline, which also limits the gate to TTGIR.
- `rows.tsv` of the gate marks each result row changed if a kernel it ran changed (from the run's own `rows.tsv`), so an op that only changed in one shape's kernel does not turn its other rows into results; those rows become controls. Runs recorded without row tracking are judged per op.
- A changed binary does not imply a performance change; step 3 decides. Check SASS before trusting a big swing: ptxas scheduling alone can make the same TTGIR twice as slow on another build (e.g. through different load batching).

### Batch runs

`run_batch.py` runs `run_pytest.sh` for each selected op, sharded across GPUs and serial within each GPU.

```bash
# Every collectable op of both repos, minus unconditional skips and ops/dismiss.txt
python run_batch.py --all --gpus 0,1,2,3,4,5,6,7 --dtypes bfloat16
# A subset: main repo by default, vllm: prefix for the other; @op_file for same-named tests
python run_batch.py --ops softmax vllm:fused_marlin_moe_w4a16_mxfp4 nextafter_@nextafter_
# Several shapes per op instead of one
python run_batch.py --ops-file my_ops.txt --shape-file shape/all_ops_multi.yaml
# Resume: reuse PASS/SKIP, rerun FAIL/TIMEOUT
python run_batch.py --all --batch-dir runs/batch_xxx --resume
# Replay an earlier batch per op (shapes also default to that batch's)
python run_batch.py --ops-file my_ops.txt --replay-root runs/batch_xxx
# Extra environment for every op, e.g. a compiler switch
python run_batch.py --ops-file my_ops.txt --env FLAGTREE_XXX=1
```

- Each op runs in its own process, so a CUDA crash only loses that op, and `<batch>/<repo>/<op>/` has the same layout as a single-op run.
- `--op-timeout` (default 1800 s) kills a hung op and records `TIMEOUT` without retry. Test failures and signal kills are retried up to twice, stopping as soon as the same exit code repeats.
- With `--dtypes`, an op that rejects the dtype is rerun without the filter (`dtype_fallback`); any other failure without a SUCCESS row gets one last unfiltered run (`dtype_rescue`).
- Runtime skipif and xfail count as SKIP. Only PASS and SKIP are marked complete for `--resume`.
- `ops_status.csv` has one row per op (status, time, rows, notes) and is updated as ops finish; `summary.csv` has one row per result row, and `batch_summary.py` rebuilds it for older batches. The notes column counts `no_cudagraph_rows` and `replay_fallback`; read it first.
- Before an op starts, its GPU must be idle: no process in `nvidia-smi`, and under 1 GiB in use (processes of other containers are not listed). A neighbour on the same GPU skews both timing and tuning, so the batch waits (`--busy-gpu wait`, the default; `warn` measures anyway and notes `gpu_busy`). GPU numbers follow `nvidia-smi` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`).
- Batch numbers are a screen: GPUs running side by side couple through power and thermals. Confirm suspects with a [repeated-measures A/B](#repeated-measures-ab).

**Reference full run** ([tested versions](#tested-versions), 7×H20, `--dtypes bfloat16`, single-shape set): `--all` selects 1608 ops and takes about 2.5 hours, 12 s per op at the median (mostly process start and imports). PASS 1529, SKIP 43; the other 36 fail for upstream or environment reasons and are listed in `ops/dismiss.txt`. Every SKIP is an upstream skipif (a missing TransformerEngine or vLLM reference, another vendor or architecture). On the multi-shape set (275 ops, `--dtypes "bfloat16 float16"`) an op takes about a minute at the median and up to two hours in record mode (`chunk_gdn2`), so give it `--op-timeout 7200`.

### Faster sweeps on multiple GPUs

With many shapes, most of the wall time goes to the autotune sweep rather than the measurement. `PARALLEL_WARMUP_GPUS=N` runs the sweep on N GPUs and then measures serially on one GPU as usual:

```bash
PARALLEL_WARMUP_GPUS=8 SHAPE_FILE=shape/DeepSeek-V4-Flash-p32768d1024.yaml bash run_pytest.sh
```

Outputs, replay and the result table are unchanged. Shards running side by side disturb each other, which can change which config wins; the log then warns:

```
WARNING 1 config key(s) got different winners across shards — parallel interference reached the sweep
```

With no conflicts the configs can be trusted; with many, rerun without `PARALLEL_WARMUP_GPUS` before drawing conclusions. The plugin does nothing under `REPLAY_FROM`, with fewer than two visible GPUs, or when the shape set has no entry under the `OP` name ([details](docs/design.md#parallel-sweep)).

## Configuration

`run_pytest.sh` is configured through environment variables; `run_batch.py` sets them per op from its own flags (see `--help`). The upstream level and mode are fixed to `--level core --mode kernel`; edit the pytest command in `run_pytest.sh` to change them.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OP` | `fused_marlin_moe_w4a16_mxfp4` | test function name without `test_` |
| `OP_FILE` | `fused_marlin_moe_w4a16_mxfp4` | benchmark file name without `test_` and `.py` |
| `FLAGGEMS_DIR` | `$FLAGGEMS_MAIN_DIR`, else `../FlagGems` | repo of the op under test |
| `FLAGGEMS_MAIN_DIR` / `FLAGGEMS_VLLM_DIR` | `../FlagGems` / `../FlagGems-vllm` | the two checkouts, as used by `run_batch.py` and `gen_inventory.py` |
| `SHAPE_FILE` | record: `shape/all_ops_multi.yaml`; replay: `$REPLAY_FROM/shapes.yaml` | shape set; ops without an entry use upstream's default shapes |
| `DTYPES` | empty: upstream's dtype sweep | space-separated dtype whitelist, e.g. `"bfloat16 float16"` |
| `REPLAY_FROM` | empty: record mode | recorded run dir whose autotune choices to replay |
| `USE_FLAGTUNE` | `0` | `1` uses the FlagTune expanded tuning space (slow full search on first run); same value on both A/B sides |
| `PARALLEL_WARMUP_GPUS` | empty: serial | shard the autotune sweep across N (≥2) GPUs |
| `GRAPH_MAX_CALLS` | `1000` | most calls unrolled into one CUDA graph; `0` = no cap ([details](docs/design.md#graph-size)) |
| `OUT_DIR` | `runs/<op>_<timestamp>` | output dir |
| `FLAGGEMS_PERF_COLOR` | empty: auto by tty | `always`/`never` for terminal color; `run.log` is always plain |

### Shape sets

A shape set is a yaml whose top-level keys are op_names (the `op_name` given to upstream's `Benchmark`, usually equal to `OP`) or Benchmark class names:

```yaml
softmax:
  shapes:
    - [1024, 1024]
    - [64, 512, 512]
```

| File | Content |
|---|---|
| `shape/all_ops_single.yaml` | one curated shape per op; default of `run_batch.py` |
| `shape/all_ops_multi.yaml` | several shapes per op; default of `run_pytest.sh` |
| `shape/DeepSeek-V4-Flash-p*d1024.yaml` | marlin MoE shapes from DeepSeek-V4-Flash traces (raw trace in the `.txt` of the same name), shared by the four quantization schemes |
| `shape/DeepSeek-V4-Flash-sweep.yaml` | the same MoE config with num_tokens over powers of two |

- Both `all_ops_*` sets include the keys of upstream's `core_shapes.yaml` (the single set keeps one shape of each). `--shape_file` replaces core_shapes rather than extending it, and without class-name keys such as `BlasBenchmark` the BLAS family would fall back to 1-D shapes and fail.
- Ops without an entry run upstream's default shapes. For ops that hardcode their shapes in `set_shapes()` that means all of them (16 large MoE shapes for marlin), so give new or renamed ops of that kind an entry in both sets.
- Benchmarks parameterized by cases that never read shapes (e.g. `scaled_mm_benchmark`) always run their built-in shapes ([details](docs/design.md#shape-override)).

### Plugins

`run_pytest.sh` loads its plugins from the `PLUGINS` array; comment out a line to disable one. What each plugin does and when to turn it off: [docs/design.md](docs/design.md#plugins).

## Outputs

```
runs/softmax_20260716-031752/
├── run.log              # full log with the result tables, ANSI stripped
├── shapes.yaml          # the shape set used
├── autotune_records/    # autotune configs chosen in record mode, read by REPLAY_FROM
│   └── softmax.json
├── rows.tsv             # per result row: shape and the cache hashes of the kernels it ran
├── ttgir/               # IR of the variants actually used, per shape
│   ├── 1024x1024/softmax_kernel_inner/w4s3__bf16.ttgir
│   ├── index.tsv        # every compiled variant: shape, file, launches, constexprs, cache hash
│   └── naming.md        # legend of the abbreviated file names
└── ttir/<cache hash>/   # ttir and compile options of the variants used, input of ir_gate.py
```

- Only variants used by the measurement are written; sweep losers appear in `index.tsv` only.
- File names list the constexprs that differ within a (shape, kernel) group, abbreviated (`BLOCK_SIZE_M` → `BSM`), then `w{warps}s{stages}`; ties are broken by dtype, argument specialization or hash ([details](docs/design.md#ir-dump)).
- Kernels launched outside input construction and measurement, such as module-level probes, go to `shape_unknown/`.
- `rows.tsv` follows the order of the result tables in `run.log`: row `i` of a table maps to the kernels its measurement launched (scope `timed`), which is how `ir_gate.py` attributes changes to rows. A benchmark that times without `triton.testing.do_bench` gets every kernel launched while the row was being measured (scope `row`).
- The IR is written at process exit, also after a CUDA crash.

## Keeping up with upstream

1. Update both checkouts (for forks: `git fetch upstream && git merge --ff-only upstream/<branch>`).
2. Rebuild the inventory and refresh the shape sets:

   ```bash
   python gen_inventory.py --verify-collect
   python gen_inventory.py --refresh-shapes shape/all_ops_single.yaml --one-shape
   python gen_inventory.py --refresh-shapes shape/all_ops_multi.yaml
   ```

   Curated shape sets (anything other than `all_ops_*`) are not rewritten by `--refresh-shapes`; check them instead and fix what it reports:

   ```bash
   python gen_inventory.py --check-shapes shape/DeepSeek-V4-Flash-p4096d1024.yaml --check-shapes my_shapes.yaml
   ```

   A key matching no current op_name is **silently ignored at run time**: upstream falls back to the benchmark's own hardcoded shapes, the run looks clean, and the numbers are for shapes you did not ask for. A run also prints `SHAPE_KEYS_UNUSED` when an op it ran found no entry in the shape file, and `IR_DUMP_FAILED` / `ROWS_TSV_FAILED` when the IR dump or the row-to-kernel map could not be written — grep `run.log` for both.

   `--refresh-shapes` keeps the op_name keys that still exist upstream and syncs the core_shapes keys. Keys of renamed ops move to `shape/<name>.unmatched.yaml`: map them to the new name (`grep -rn 'op_name=' $FLAGGEMS_DIR/benchmark/test_<op>.py`), move them back and delete the file.
3. Run a full screen. Add confirmed upstream or environment failures to `ops/dismiss.txt` with the reason and date, and now and then rerun the dismissed ops with `--ops-file` to remove those that pass again.
4. Update [Tested versions](#tested-versions) and the reference full run to the commits you screened.

`ops/inventory_*.csv` has one row per test function: `op`, `op_file`, `marked_skip` (unconditional `pytest.mark.skip`), `collected` (pytest collect check) and `op_names`. Ops with `marked_skip=yes` or `collected=no` (e.g. FlagGems-vllm ops importable only on Ascend) are not part of `--all`.

## FAQ

- **The same op shows a different gap in another A/B.**

  Check whether the two used the same record. Each record tunes its own configs, and the effect of a compiler change can depend strongly on the config (several-fold for some ops). A gap is the effect under one record's configs; compare A/Bs only on the same record.

- **`ab_compare.py` leaves rows undecided.**

  Their gap is above the threshold but the samples overlap too much for p < 0.01. Run `ab_run.py --until-decided` with a higher `--max-rounds`; rows that stay undecided are small or unstable effects.

- **My latency differs a lot from someone else's.**

  Check that the GPU is not shared (`nvidia-smi`), then whether they used CUDA graph timing: for small shapes such as M=1, launch overhead dominates and plain `do_bench` can read 2× or more; large shapes mostly agree.

- **Same shapes, nothing changed, yet two runs give different speedups.**

  Expected for two record runs: each sweeps again, and timing noise makes them pick different configs. Replay one from the other.

- **The first run of an op is very slow.**

  Either `USE_FLAGTUNE=1`, whose first run searches the whole FlagTune space (over ten minutes for marlin, with `run.log` sitting on the test name meanwhile), or a libtuner op sweeping for the first time (about 50 minutes for mm). Winners persist in `~/.flaggems/config_cache/`, after which the same shapes take seconds. With the default `USE_FLAGTUNE=0` one shape usually takes around ten seconds; for many shapes use `PARALLEL_WARMUP_GPUS`.

- **Should I pass `--warmup` or `--iter`?**

  No. The warmup budget drives the explicit warmup before capture, and the 100 ms iteration budget is converted to a count from the kernel time.

- **A shape dir under `ttgir/` is missing files.**

  Check the `launches` column of `index.tsv`: variants that never ran outside the sweep for that shape are not written. Fallback markers in `run.log` explain unusual cases.

- **An op in a batch is SKIP.**

  The `SKIPPED [...]` lines at the end of its `run.log` give the reason, usually an upstream skipif (missing reference implementation, unsupported architecture).

## Development

```
run_pytest.sh       single-op entry
run_batch.py        batch entry
ab_run.py           repeated-measures A/B (rounds of interleaved replays)
ab_compare.py       per-row verdicts of an A/B, or a single comparison of two runs
ab_stats.py         verdict rule and Mann-Whitney test
ir_gate.py          compile-diff gate
gen_inventory.py    op inventory and shape-set refresh
batch_summary.py    run.log parsing and summary.csv
_*_plugin.py        pytest plugins; _fg_benchmark.py, _term_style.py and _timing_scope.py are shared helpers
ops/                op inventories and the batch exclusion list
shape/              shape sets
tests/              unit tests of the GPU-free logic
docs/               design notes
```

```bash
pip install -r requirements.txt ruff
ruff check .
shellcheck run_pytest.sh
python -m pytest
```

CI (`.github/workflows/ci.yml`) runs these checks on Python 3.10 and 3.12. Measurements need a GPU and the upstream repos and are not part of CI; after changing a plugin, run a small batch that covers both repos, the special input paths and the do_bench fallback, and check that each run's `rows.tsv` has one line per result row:

```bash
python run_batch.py --ops abs mm softmax conv1d_padding perf_unique2 vllm:fused_marlin_moe_w4a16_mxfp4 \
  vllm:flash_mla_sparse_fwd --dtypes bfloat16
```

## License

MIT — see [LICENSE](LICENSE).
