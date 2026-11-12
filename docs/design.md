# Design notes

**English** | [简体中文](design.zh-CN.md)

How zl_bench gets comparable numbers out of the FlagGems benchmarks, and what to keep in mind when reading them. For usage, see the [README](../README.md).

- [Plugins](#plugins)
- [Why replay](#why-replay)
- [Timing](#timing)
- [IR dump](#ir-dump)
- [Statistics](#statistics)
- [Shape override](#shape-override)
- [Parallel sweep](#parallel-sweep)
- [Upstream assumptions](#upstream-assumptions)

## Plugins

`run_pytest.sh` loads these with `-p` from its `PLUGINS` array:

| Plugin | Role | Disable when |
|---|---|---|
| `_parallel_warmup_plugin` | with `PARALLEL_WARMUP_GPUS=N`, runs the autotune sweep on N GPUs ([Parallel sweep](#parallel-sweep)) | never needed: without the variable it does nothing |
| `_seed_plugin` | seeds random, numpy and torch so data-dependent ops (sort, topk, ...) get byte-identical inputs | never |
| `_shape_override_plugin` | makes the shape yaml win over upstream code that bypasses it, logs where each benchmark's shapes came from and flags a run that did not measure the file's shapes ([Shape override](#shape-override)) | never |
| `_autotune_record_plugin` | records or replays the choices of Triton autotune and libtuner ([Why replay](#why-replay)) | never |
| `_cudagraph_plugin` | times with warmed-up CUDA graphs, falling back to `do_bench` with a marker ([Timing](#timing)) | to match someone else's plain `do_bench` numbers |
| `_pretty_report_plugin` | folds inputs that are identical across all rows of the result table and adds color; column titles and the `SUCCESS`/`FAILED` words stay as upstream prints them, so log parsing is unaffected | to match upstream's raw table verbatim |
| `_ir_dump_plugin` | records each compiled variant, its real use per shape and the kernels each result row ran, and writes `ttgir/`, `ttir/` and `rows.tsv` at exit ([IR dump](#ir-dump)) | never |

## Why replay

### Two tuning layers

| Mechanism | Stored in | After a compiler change | A/B risk |
|---|---|---|---|
| Triton `@triton.autotune` | process memory | every process sweeps again | timing noise can make A and B pick **different configs** |
| FlagGems `@libtuner` (incl. the FlagTune expanded space) | `<cache dir>/config_cache/*.db` (sqlite, persists across processes); the cache dir is `~/.flaggems` for FlagGems and `~/.flagdnn` for FlagGems-vllm, or `FLAGGEMS_CACHE_DIR` for both | **still valid**: table names hash only the kernel source and the config space | B reuses A's winner and measures "the compiler change under the old config" rather than "each side at its best" |

libtuner also stores a BenchmarkCache in the same db: the measured latency of every config of a sweep. With a warm db even a record run does not re-measure and picks the winner from stored latencies; "record sweeps live" only holds for a cold db.

### Magnitude

Config differences can outweigh the change under test. Two record runs of the same shapes and settings, a dozen minutes apart, can differ by around 10% in mean speedup, shifted in the same direction across shapes rather than randomly. Diffing their `autotune_records/*.json` usually shows whole steps, not tweaks: `num_stages`, `BLOCK_SIZE_*` and `num_warps` all jump. Record-mode numbers therefore only provide the configs for replay; a conclusion backed by record data alone is unverified.

### Each side at its best

The libtuner cache is invalidated by changes to the kernel source, `tune_configs.yaml`, the expand yaml or `USE_FLAGTUNE`; a new Triton major or minor version, or another vendor backend, starts a db of its own (`TunedConfig_<vendor>_triton_<major>_<minor>.db`). zl_bench only clears its own `TRITON_CACHE_DIR`, never that db. To compare each side at its own best config, give each side an empty cache: point `FLAGGEMS_CACHE_DIR` at a fresh directory or `FLAGGEMS_DB_URL` at a throwaway file, or delete `TunedConfig_*.db` under `~/.flaggems/config_cache/` (FlagGems) or `~/.flagdnn/config_cache/` (FlagGems-vllm). Leaving out `REPLAY_FROM` is not enough: with a warm cache both sides hit the same winner anyway. Both views are valid; state which one a conclusion uses.

### Record and replay details

- **Record** reads back the config of each call from `cache[key]` after `run()` instead of looking for new cache keys, since a warm libtuner cache never gets new keys. Single-config kernels, which skip the cache, record their only config.
- **Replay** writes the recorded config into the cache before every call, unconditionally: skipping keys that are already cached would leave a warm libtuner cache tuning on its own.
- Configs with a `pre_hook` (e.g. the TMA configs of Hopper mm): libtuner reattaches the hook to a replayed config by matching its kwargs, so these record and replay normally. Plain `@triton.autotune` has no such mechanism; those configs are not recorded. On replay such a kernel tunes for itself and is reported: `key_missing` for a key the record lacks, `kernel_missing` (once) for a kernel the record has no entry for at all.
- **Fallback**: a kernel or a key missing from the record, or a recorded config that fails to compile with the new compiler, falls back to live tuning and logs `AUTOTUNE_REPLAY_FALLBACK reason=...`. A failed config can be evicted from Triton's in-memory cache but not from libtuner's sqlite cache; the marker then ends in `_no_evict`, because the retry reads the same config again. A kernel with a single config is not tuned, so there is nothing to replay or report.
- To check that a record covers a kernel, look for its entry in `autotune_records/<op>.json`.

## Timing

### Warmup before capture

Upstream `--mode kernel` times with `triton.testing.do_bench`; `_cudagraph_plugin` replaces it with CUDA-graph timing that follows `triton.testing.do_bench_cudagraph`: a graph of back-to-back calls covering the 100 ms repetition budget, replayed 10 times, median per call. It measures kernel time without launch overhead. (Upstream's own `--mode cudagraph` neither warms up nor falls back, so it is not used.)

`do_bench_cudagraph` warms up for only 5 iterations, too few for ops whose first calls compile autotune configs, run libtuner selection or JIT-initialize. If that one-off work leaks into the captured graph or the first timed iteration, latency jitters between runs; on the M=1 multi-kernel path the first round can read about a third of steady state. The plugin therefore:

1. synchronizes, so inputs still being produced on the default stream are complete before the side-stream capture;
2. runs the op once and discards the result, absorbing compilation;
3. runs it for the caller's warmup budget (`Config.warm_up`), with the count derived from one steady-state run and clamped to 5–200;
4. runs it once more with torch's sync debug mode set to error, to find host synchronization ([Fallback](#fallback-to-do_bench));
5. captures and times.

If a new op still shows first-round jitter, raise the cap in `_warmup_before_capture` first.

### Graph size

Capture time grows linearly with the number of calls in the graph, and for some ops each call is expensive to capture: about 12 ms for `conv1d_padding`, so filling 100 ms with thousands of calls costs half a minute or more per measurement; recording its 13 fp16 shapes took 67 minutes uncapped and 30 minutes capped. The plugin therefore puts at most `GRAPH_MAX_CALLS` (default 1000) calls into one graph and reaches the budget by replaying the graph back to back within each of the 10 timed samples, so each sample still spans the same number of calls.

Back-to-back replays add nothing measurable from 250 calls per graph up: `abs` on 64 elements reads 1.03 µs per call at 250, 500 and 1000 calls per graph and uncapped, and 23.0 µs on 4096×4096 at every size. Below that, small graphs read tiny kernels too fast (0.86 µs at 100 calls), so keep the cap at 250 or more. The graph size itself shifts some ops slightly: `conv1d_padding` reads 45.9 µs per call at 500 calls per graph and 46.8 µs at 2000, as each call's outputs grow the graph's private memory pool. Both sides of an A/B use the same plan, so comparisons are unaffected; absolute numbers of such ops can differ from an uncapped `do_bench_cudagraph` by that much. `GRAPH_MAX_CALLS=0` restores the uncapped graph.

### Fallback to do_bench

Some ops cannot be captured: host syncs, dynamic allocations, autograd backward, illegal stream operations. They are timed with plain `do_bench` instead, and `run.log` gets `BENCHMARK_DIRECT_NO_CUDAGRAPH op=... phase=... reason=...`, where `phase` is the step that rejected the graph (`host_sync` or `capture`). A call that fails by itself, during the warmup and before any capture, is not such a case and raises as it would under `do_bench`: the autotuner times every config of a sweep through `do_bench` and drops those that raise (more shared memory than the GPU has, a failed compile), and a fallback there would run each rejected config once more and report it as a measurement that could not be captured. These safeguards keep the fallback working:

- **Host syncs are found before capture.** A call that synchronizes with the host (`tensor.item()`, `nonzero`, a device-to-host copy) can never be captured. Running it once under `torch.cuda.set_sync_debug_mode("error")` detects that without starting a capture, and the measurement goes straight to `do_bench` (`phase=host_sync`).
- **Memory of a failed capture is released.** An aborted capture keeps its graph and private memory pool alive in reference cycles until the garbage collector runs; with inputs of tens of GiB a few of them leave too little memory for the next measurement. The plugin collects and empties the cache before the fallback. A capture that runs out of memory is retried once with one call per graph (`CUDAGRAPH_OOM_RETRY`), still timed by graph replay.
- **Fatal errors are matched on the error text itself.** Illegal addresses, OOM, device-side asserts and the like are re-raised rather than timed. torch appends an "enable device-side assertions" hint to every CUDA error, so matching loose substrings would make every capture failure fatal.
- **Capture runs under a throwaway RNG state.** An aborted capture can leave the default CUDA generator in capturing mode, after which every `torch.randn` in the process raises "Offset increment outside graph capture". The generator's state is swapped for a scratch one during capture and restored afterwards, so the real generator never takes part in a capture.
- **An aborted capture is followed by one untimed call.** cusolver keeps the aborted capture in its handle state and reports `CUSOLVER_STATUS_EXECUTION_FAILED` once on the next call before recovering (e.g. `linalg_eigvals`, `linalg_matrix_rank`).

### Accuracy of fallback rows

Fallback does not bias an A/B, since both sides fall back the same way for the same op, but `do_bench` includes the Python launch path in every iteration. On sub-millisecond kernels the two methods can differ by 2× or more and rows fluctuate more, so do not over-read absolute values or differences under about 10% on small shapes. Fallback happens per measurement, and one table can mix both methods; `grep BENCHMARK_DIRECT_NO_CUDAGRAPH run.log` shows which rows are affected.

## IR dump

`_ir_dump_plugin` works on the run's private Triton cache, keyed by the cache dir of each compiled kernel:

- **Compile listener**: writes a `__constexprs.json` sidecar with the constexpr values, signature dtypes and per-argument specializations (e.g. `tt.divisibility`) next to each kernel, since the ttgir alone does not say which autotune variant it is.
- **Shape tracking**: publishes the shape being fed. Upstream has two input paths, the legacy `get_input_iter` generator and the Case API (`get_case_iter` + `build_inputs` when `supports_cases()` is true, e.g. the `UnaryPointwiseBenchmark` and `BlasBenchmark` families); both are wrapped, in place so upstream's `supports_cases()` identity checks keep working, and yield the same shape labels. A label is the shape with its dims joined by `x`; a nested shape such as `[input_shape, output_size]` joins its inner dims with `-` (`2-256-14-14x7-7`), and a label over 120 characters is cut and ends in a digest. On the legacy path the label is the shape the iterator is at in its walk over `self.shapes`: while `get_input_iter` runs, `self.shapes` is a list that reports each shape as the iteration reaches it (`_shape_scope`), so a shape that yields several inputs labels all of them, which counting the inputs would not. An iterator `_shape_override_plugin` feeds from the yaml reports those shapes the same way. One that walks no shapes (a hardcoded list of cases), or all of them before its first input, gets `input<i>` for its i-th input, which names no shape rather than the wrong one. The label names directories under `ttgir/` and fills the `shape` column; which kernels a row ran does not depend on it.
- **Launch tracking**: counts launches per (kernel, shape), separately inside and outside `Autotuner._bench`. A variant is "used" if it ran at least once outside the sweep.
- **Row tracking**: upstream creates one `BenchmarkMetrics` per result row, in almost every benchmark right before measuring it. From then until the next input is built, every launch outside the sweep and inside the timed function belongs to that row; `_timing_scope` marks the timed function, so launches the timing harness itself makes (e.g. fills torch issues when a capture begins) stay out. A benchmark that measures first and makes the row afterwards is followed too: what was launched in a timed function for the input, with no row current, goes to the row made next. When a result table is printed, its rows are noted in print order. `rows.tsv` lists, per (op_name, dtype) table and row index as in `run.log`, the shape label and the cache hashes of the kernels the row ran. A row for which nothing was launched in a timed function (a benchmark that times without `triton.testing.do_bench`, a few call `do_bench_cudagraph` themselves; or a run without `_cudagraph_plugin`) gets every launch made while it was current, or failing that for its input before it existed, marked `scope=row`. That set can include kernels that only build inputs, which errs on the side of calling a row changed. A row left without any kernel launched none (small inputs may take a path without Triton); the gate clears such a row only when the run's rows account for every variant it used.
- **Binaries**: `rows.tsv` also lists, per row and in the order of its kernels, the fingerprint of each kernel's binary (see [Controls and false positives](#controls-and-false-positives)). The cache hash cannot serve: it covers the compiler's own sources and differs between any two builds.
- **Dump at exit**: for each used variant, the ttgir goes to `ttgir/<shape>/<kernel>/<name>.ttgir` and the ttir plus Triton's metadata json to `ttir/<cache hash>/`. Sweep losers are listed in `ttgir/index.tsv` only. Dumping at process exit rather than session end keeps the IR of a run whose session ended in an error; a process that is killed (a segfault, an abort, a timeout) leaves none. `rows.tsv` is written first and on its own, so a failure of the much larger IR copy (`IR_DUMP_FAILED`) does not take the row-to-kernel map with it, and `ttgir/` and `ttir/` of an earlier attempt in the same run dir are removed first: a batch retries into the same dir, and the first attempt's variants are not this run's.

File names keep only what tells the variants of a (shape, kernel) group apart: constexprs whose value differs within the group, abbreviated per word (`BLOCK_SIZE_M` → `BSM`; legend in `naming.md`), then `w{warps}s{stages}`. Remaining collisions are broken by signature dtype (`__fp16`), then argument specialization (`__EMdiv16`), then a hash prefix.

The json in `ttir/` is the compile options for recompilation: Triton's metadata with `num_warps` restored from the ttgir module, because warp specialization rewrites the metadata value to the sum over partitions (e.g. 12), which the compile option rejects.

## Statistics

### One sample per round

A round runs each side once per op, in a fresh process with its own Triton cache, and contributes one latency per result row and side. Within a measurement the plugin already takes the median of 10 graph replays; what remains between rounds is run-to-run noise: GPU clocks and thermals, memory placement, the GPU a round lands on. That noise is what a verdict has to beat, so the unit of replication is the round, not the replay.

Pairing A and B on one GPU back to back cancels the per-GPU bias (a few percent between the GPUs of one box) and slow drift; alternating which side goes first cancels order effects such as a warm L2 or clock ramp. A round is used only as such a pair: when a side fails, `ab_run.py` reruns both sides of the round and first drops the rows the earlier attempt left in the rounds' `ops_status.csv`, so that a rerun cut short after one side does not pass for a finished round. `ab_compare.py` leaves out a round whose sides ran on different GPUs (the `gpu` column of those files) or that has a row for one side only. A side that failed after measuring some rows did end: those rows count, are flagged `failed_side`, and are weak evidence for the recheck.

### Verdict

`ab_stats.compare` judges a row from its A and B samples:

| Verdict | Condition |
|---|---|
| `undecided` | fewer than `--min-rounds` (6) rounds |
| `same` | \|gap\| ≤ `--threshold` (0.5%) |
| `faster` / `slower` | the rank test shows the gap: Mann-Whitney p < `--alpha` (0.01) |
| `faster` / `slower` | the pairing shows it: every round points the same way, and the sign test over the rounds is below `--alpha-paired` (0.05) |
| `same` | neither does, and Mann-Whitney p ≥ `--clear` (0.1) |
| `undecided` | otherwise: a gap, but not yet significant |

The conditions apply in this order. `gap = median(A) / median(B) − 1`. The Mann-Whitney U test compares ranks, so it needs no assumption about the latency distribution, which is skewed and sometimes bimodal (some kernels alternate between two speeds from run to run). It uses the exact null distribution for small samples without ties and the normal approximation otherwise, as `scipy.stats.mannwhitneyu` does. With 6 rounds per side the smallest attainable two-sided p is 0.0022, so `--alpha 0.01` is reachable only when the two samples barely overlap. The threshold keeps a statistically clear but immaterial shift (a fraction of a percent) out of the results.

### Two tests

The rank test pools the rounds: it asks whether the A samples as a group lie apart from the B samples. Whatever the two runs of a round share, above all the GPU the round landed on, is noise to it. Over GPUs a few percent apart, the samples of a row whose B side is 2% faster in every single round still overlap almost completely; the rank test puts such a row at p ≈ 0.2, which by itself reads `same`.

The sign test uses what the pooling throws away: each round is a pair measured back to back on one GPU, so the pair cancels the GPU, and only the direction of each round is counted. Under no difference a round goes either way with probability one half; six rounds all pointing the same way have a two-sided p of 2/2⁶ = 0.031. The rule asks for every round, not just for enough of them, so a row that flips between two speeds is not decided by a lucky majority, and it asks for the same threshold on the median gap as the rank test. A verdict is therefore possible at 6 agreeing rounds and not at 5 (p = 0.0625), whatever `--min-rounds` says. `--alpha-paired 0` turns the pairing off.

The price is a looser level. In a simulation of rows without any difference (per-run noise of 0.5–1%, with or without a per-round GPU offset of 3% shared by the two sides), 6 rounds give 2–3% faster/slower verdicts, against under 1% for the rank test alone; from 10 rounds on both stay under 1%. What it buys: with 0.5% per-run noise, a 2% effect over GPUs 3% apart is found in 99% of rows instead of 1% (61% at 1% noise). The CSV names the test that showed each faster/slower row (`basis`), the report counts the rows that rest on the pairing alone, and those rows count as weak evidence for the recheck below. On a real A/B, read the rate off the controls.

### Controls and false positives

With `--gate`, rows whose kernels compiled to identical binaries on both sides cannot differ; they go through the same rule as the changed rows, and every `faster` or `slower` among them is a false positive. Their rate is the measured false-positive rate of the rule on this machine and this record, which a fixed α alone does not give: run-to-run effects that are not random noise (a bimodal kernel, a slow GPU on some rounds) show up here. Controls ride along on the ops that are measured anyway, so they cost nothing extra.

The gate's verdict is a prediction: it compiled the recorded variants with the recorded options on both builds. A control is only a control if the two sides launched what the gate compiled, and that can fail without any marker in the log. An operator library may set a compile option only where the compiler has it: FlagGems-vllm launches `_qsa_sparse_paged_gqa_kernel` with `num_stages=2, pipeline_merge_loads="any"` on a FlagTree that knows the option and with `num_stages=3` on one that does not, from the same TTIR. A record made with the older build holds only the second variant; the gate recompiles it on both builds, finds one binary and clears all 18 rows of that benchmark, while in 12 of them the newer side measures a different kernel. Such rows would be left out of the results, and their difference could surface at most as "false positives".

So the runs carry the evidence themselves. `rows.tsv` records, for each result row, a fingerprint of every kernel it launched: the kernel's name, the code of its cubin and the launch metadata the launcher reads (`_binary_id`). Line info is left out, because a cubin names source files by path, files of the Triton install among them, so two builds in different directories emit cubins that differ in `.debug_line` only. `ab_compare.py --gate` compares the fingerprints of the two sides round by round: a row the gate cleared whose sides ran different binaries in any round is **diverged**, and counts as a result like a changed row. A cleared row is a control when the binaries agree in every round, or when the runs recorded no fingerprints and had no replay fallback; it is unknown when a run could not tell what the row launched (`?`), or had a replay fallback and no fingerprints. `ir_gate.py` points at the situation up front: it warns when a build has compile options the record, or the other build, does not have, which is when an operator library can behave this way.

### More rounds, and fresh rounds

`--until-decided` adds rounds to ops with undecided rows. Testing again after each addition inflates the false-positive rate a little (sequential testing); the controls, which get the same extra rounds, show by how much.

A row near the threshold, or decided on few rounds, has been both selected and judged on the same samples, which overstates small effects (the winner's curse). The fix is an independent check: `ab_run.py --add-rounds N` runs N new rounds, and `ab_compare.py --confirm` judges each faster/slower row on those alone, one-sided in the original direction at α = 0.05, by either test: the rank test, or every new round agreeing. Either may confirm, whichever gave the verdict: a row the pairing decided is one whose samples overlap, on fresh rounds as before, so the rank test alone could never confirm it. One-sided is right because the direction was fixed before the new data existed; a row that passes both checks is a false positive only if both are, which the independence of the new rounds makes rare. A confirmed row reports the gap over all rounds; a row that is not reproduced becomes undecided rather than same, since a small real effect can also miss on a few rounds. New rounds too few to reach 0.05 by any outcome prove nothing either way and are reported as `too_few_rounds`, the row keeping its verdict: one-sided, n rounds reach at best 1/C(2n, n) by the rank test (0.05 at three rounds, 0.014 at four) and 1/2ⁿ by the pairing (0.0625 at four, 0.031 at five). So three rounds never confirm, whatever they show (with tied samples the rank test's normal approximation can read lower than that bound; the bound is checked first), four confirm only when the two samples are fully separated, and four that all agree while overlapping are `too_few_rounds` as well.

## Shape override

Upstream reads shapes in `Benchmark.set_shapes()`, falling back from the op_name key to an MRO class-name key to the base-class `DEFAULT_SHAPES`. A good share of benchmarks bypass that. `_shape_override_plugin` closes three gaps, each by reusing the class's own input construction and only swapping the source of shapes, so the measured kernel stays the same; ops without a yaml entry keep their upstream shapes.

A shape set serves both repos, and a name is not always the same thing in both: `fused_recurrent_gated_delta_rule` is an op_name in each, taking a sequence length in FlagGems and a case name in FlagGems-vllm, and each benchmark fails to unpack the other's entry. A key can therefore be scoped, `vllm:<name>` or `main:<name>`: it applies to that repo only and beats the plain key there. Upstream reads the file itself and knows nothing of scopes, so `run_pytest.sh` resolves the set for the repo under test (`shape_set.py`) and hands upstream the result, archived as the run's `shapes.yaml`. `gen_inventory.py --refresh-shapes` scopes a core_shapes key when it merges it for the first time and the other repo uses the name without defining the key, or defines it with other shapes; keys the set already has keep their form.

1. **Shapes hardcoded in `set_shapes()`** (e.g. `fused_marlin_moe_*`): `Benchmark.init_user_config`, which runs right after `set_shapes()`, is wrapped to put the yaml shapes back. The same is done after a subclass's own `init_user_config` that assigns `self.shapes` and never reads it (`adaptive_max_pool3d`, `beam_search_score`): what it leaves there is hardcoded whatever the yaml gave. One that reads `self.shapes` filters or validates what the yaml gave and is left alone. Case-API benchmarks enumerate `self.shapes`, so this covers them too. It steps aside when the subclass normalizes the arity after reading the yaml (BLAS (B, M, N, K) ↔ (M, N, K)), detected as both sides having a single but different arity.
2. **`get_input_iter` with a hardcoded list that never reads `self.shapes`** (conv padding, pool, nll, scaled_softmax families): yields one input per yaml shape through the class's standard `input_fn(shape, dtype, device)`. Classes without such an `input_fn` are left alone.
3. **Bespoke input construction**, registered by class name: `UpsampleBicubic2dAaBackwardBenchmark`, `FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark` and `FlashmlaSparseBenchmark` build their inputs from config tables or `TestParam` objects; `CutlassScaledMMPerfKit` samples cases from an (M, N, K) list, which is replaced while the quantization-mode product stays as upstream defines it. These couple to upstream internals, and a mismatch shows up as a benchmark failure, not as silently different inputs.

Benchmarks that never read shapes at all, such as `scaled_mm_benchmark` (parameterized by cases), always run their built-in shapes.

### Which shapes a run measured

A yaml that drives nothing looks exactly like one that does: upstream falls back silently, and the run passes. The key is the benchmark's op_name, which need not be the name of the test and differs between the two repos for some benchmarks (`test_fused_marlin_moe_w8a16_int8` is `fused_marlin_moe_w8a16_int8` in FlagGems and `fused_marlin_moe_w8a16` in FlagGems-vllm), and upstream renames op_names now and then. So the plugin records each benchmark's shapes once its `init_user_config` has returned, subclass overrides included, and at session end prints one line per benchmark saying where they came from: `shapes of mm: 1 of the 1 of key 'mm'`, `... of class key 'Benchmark'`, or `its own 6 (class key 'Benchmark' not used)`.

`SHAPE_KEYS_UNUSED` is printed, and carried into the notes of a batch, only when that source cannot be what the author of the file meant:

- no key applies to the op at all, neither its op_name nor a class along the MRO (a curated set keyed on another name);
- the file has an entry under the name of the test, but none under the op_name the benchmark uses;
- the op has an entry, but the benchmark rejected it and put its defaults back (the `chunk_*` benchmarks of FlagGems-vllm do when a shape has another length than they unpack), or builds its inputs without reading shapes and has no standard `input_fn` to redirect through.

Shapes that are a subset of the entry are fine: several benchmarks drop the shapes that would not fit in memory. Falling back to a class-name key is upstream's design and is not flagged, even when the benchmark hardcodes its shapes and so does not read that key either: a whole-suite set has no entry of their own for most ops, and flagging each would bury the cases above. `gen_inventory.py --check-shapes` finds the same drift without running anything.

## Parallel sweep

With many shapes the sweep dominates: for `fused_marlin_moe_w4a16_mxfp4` with 53 shapes, a record run took 3755 s while the measured kernels summed to 49 ms, and 2645 of the 2858 compiled variants were sweep losers; replaying the same configs took 196 s.

`_parallel_warmup_plugin` runs before the record/replay plugin configures. It splits the shapes of every entry of the shape set round-robin (shape lists are usually sorted, and contiguous blocks would give one shard all the large shapes), runs one pytest subprocess per shard pinned to its own GPU, merges the shards' records into the run's `autotune_records/<op>.json`, and switches the run to replay those configs. Every entry is split because the one a benchmark reads is keyed by its op_name, which need not be the test's name, or by a class name; split alike, whichever it reads is covered by the shards between them. An entry with fewer shapes than shards is repeated rather than left out of a shard, where upstream would fall back to shapes nobody asked for. The measurement that follows is the normal single-GPU path. Subprocesses are used instead of pytest-xdist because importing FlagGems creates a CUDA context, which does not survive fork-based workers.

The measurement itself is never parallel: processes saturating several GPUs of one box couple through power limits and thermals, and per-GPU latency then depends on the neighbours. Within the sweep the same coupling can change which config wins. A key with different winners in different shards counts as a conflict, the first value is kept, and the log reports the count. It is the trust level of the speed-up: zero is fine, many means rerunning without the variable before drawing conclusions. Shard pinning respects an inherited `CUDA_VISIBLE_DEVICES`; if no shard returns configs, the run falls back to serial tuning and keeps the shard logs.

## Upstream assumptions

- The upstream benchmark framework: the `Benchmark` class in `benchmark/base.py` (`set_shapes`, `init_user_config`, `get_input_iter`, and the Case API in the main repo), the conftest options `--shape_file`, `--level`, `--mode` and `--dtypes`, `consts.BenchmarkResult` (a table is logged by printing it) and `consts.BenchmarkMetrics` (one per result row). Row tracking follows a row made before its measurement, as the base class does, and one made after it; shape labels rely on input iterators walking `self.shapes`. Each repo ships its own copy; the plugins use whichever `benchmark` package is running.
- LibTuner lives in `<package>.utils.libentry`; `_autotune_record_plugin` patches whichever package (`flag_gems` or `flaggems_vllm`) was imported.
- Triton provides `knobs.compilation.listener`, `kernel_load_end_hook` and `launch_enter_hook` (FlagTree's Triton 3.6 does).
- The plugins monkeypatch upstream internals. If behavior changes after a major upstream update, first check that each plugin still prints its registration line at startup.
