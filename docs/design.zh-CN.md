# 设计说明

[English](design.md) | **简体中文**

zl_bench 如何从 FlagGems benchmark 得到可比的数字，以及解读时要注意什么。用法见 [README](../README.zh-CN.md)。

- [插件](#插件)
- [为什么要 replay](#为什么要-replay)
- [计时](#计时)
- [IR 落盘](#ir-落盘)
- [统计判定](#统计判定)
- [shape 覆盖](#shape-覆盖)
- [多卡并行 sweep](#多卡并行-sweep)
- [对上游的依赖](#对上游的依赖)

## 插件

`run_pytest.sh` 从 `PLUGINS` 数组用 `-p` 加载以下插件：

| 插件 | 作用 | 何时关闭 |
|---|---|---|
| `_parallel_warmup_plugin` | 设 `PARALLEL_WARMUP_GPUS=N` 时把 autotune sweep 放到 N 张卡上（见[多卡并行 sweep](#多卡并行-sweep)） | 无需关闭：不设该变量即不起作用 |
| `_seed_plugin` | 固定 random、numpy、torch 的种子，数据相关算子（sort、topk 等）的输入逐字节一致 | 不关 |
| `_shape_override_plugin` | 让 shape yaml 优先于绕开它的上游代码，在日志里写明每个 benchmark 的 shape 来源，并标记没有测到文件中 shape 的运行（见 [shape 覆盖](#shape-覆盖)） | 不关 |
| `_autotune_record_plugin` | record 或 replay Triton autotune 与 libtuner 的选择（见[为什么要 replay](#为什么要-replay)） | 不关 |
| `_cudagraph_plugin` | 用预热后的 CUDA graph 计时，失败时回退到 `do_bench` 并打标记（见[计时](#计时)） | 需要与别人的普通 `do_bench` 数字对齐时 |
| `_pretty_report_plugin` | 把结果表中所有行都相同的输入折叠起来并着色；列名和 `SUCCESS`/`FAILED` 字样保持上游原样，日志解析不受影响 | 需要与上游原始表格逐字一致时 |
| `_ir_dump_plugin` | 记录每个编译变体、它在各 shape 下的实际使用和每个结果行运行过的 kernel，退出时写出 `ttgir/`、`ttir/` 与 `rows.tsv`（见 [IR 落盘](#ir-落盘)） | 不关 |

## 为什么要 replay

### 两层调优

| 机制 | 存储位置 | 编译器改动后 | A/B 风险 |
|---|---|---|---|
| Triton `@triton.autotune` | 进程内存 | 每个进程重新 sweep | 计时噪声可能让 A、B 选出**不同 config** |
| FlagGems `@libtuner`（含 FlagTune 扩展空间） | `<缓存目录>/config_cache/*.db`（sqlite，跨进程持久）；缓存目录在 FlagGems 是 `~/.flaggems`，在 FlagGems-vllm 是 `~/.flagdnn`，设了 `FLAGGEMS_CACHE_DIR` 时两者都用它 | **仍然有效**：表名只对 kernel 源码和 config 空间取 hash | B 沿用 A 的 winner，测到的是"旧 config 下的编译器差异"，而不是"两侧各自最优" |

libtuner 在同一个 db 里还存有 BenchmarkCache，即 sweep 中每个 config 的实测 latency。db 已热时，连 record 运行也不重测，直接按存下的 latency 选 winner；"record 现场 sweep"只在冷 db 下成立。

### 影响有多大

config 差异可能超过被测改动本身。同一组 shape、同样设置、相隔十几分钟的两次 record，平均 speedup 可差 10% 左右，而且各 shape 朝同一方向偏移，不是随机噪声。diff 两次的 `autotune_records/*.json`，通常看到的是整档跳变而不是微调：`num_stages`、`BLOCK_SIZE_*`、`num_warps` 都会变。因此 record 模式的数字只用来给 replay 提供 config；只有 record 数据支撑的结论算未验证。

### 两侧各自最优

kernel 源码、`tune_configs.yaml`、expand yaml 或 `USE_FLAGTUNE` 改变会让 libtuner 缓存失效；换了 Triton 主版本或次版本、或换了厂商后端，则另起一个 db（`TunedConfig_<vendor>_triton_<major>_<minor>.db`）。zl_bench 只清理自己的 `TRITON_CACHE_DIR`，不碰这个 db。要比较两侧各自最优的 config，给每一侧一个空缓存：把 `FLAGGEMS_CACHE_DIR` 指向一个新目录，或把 `FLAGGEMS_DB_URL` 指向一个一次性文件，或删掉 `~/.flaggems/config_cache/`（FlagGems）或 `~/.flagdnn/config_cache/`（FlagGems-vllm）下的 `TunedConfig_*.db`。只是不设 `REPLAY_FROM` 还不够：缓存已热时两侧仍会命中同一个 winner。两种口径都成立，报告结论时注明用的是哪一种。

### record 与 replay 的细节

- **record** 在 `run()` 之后直接读回本次调用的 `cache[key]`，而不是找新增的缓存键，因为热的 libtuner 缓存不会产生新键。只有一个 config 的 kernel 不经过缓存，直接记录这个 config。
- **replay** 在每次调用前无条件把记录的 config 写入缓存：如果跳过已缓存的键，热的 libtuner 缓存就会自行调优。
- 带 `pre_hook` 的 config（如 Hopper mm 的 TMA config）：libtuner 会按 kwargs 匹配，把 hook 接回 replay 的 config，所以照常 record 和 replay。普通 `@triton.autotune` 没有这个机制，这类 config 不记录。replay 时这样的 kernel 自己现场调优并被报告：记录里缺某个键是 `key_missing`，记录里完全没有这个 kernel 是 `kernel_missing`（只报一次）。
- **回退**：记录中缺少某个 kernel 或某个键，或记录的 config 在新编译器下编译失败时，回退到现场调优，并记录 `AUTOTUNE_REPLAY_FALLBACK reason=...`。失败的 config 可以从 Triton 的内存缓存中移除，但无法从 libtuner 的 sqlite 缓存中移除，这时标记以 `_no_evict` 结尾，因为重试会再次读到同一个 config。只有一个 config 的 kernel 不调优，无所谓 replay，也不会报告。
- 要确认记录覆盖了某个 kernel，查看 `autotune_records/<op>.json` 里有没有它的条目。

## 计时

### capture 前预热

上游 `--mode kernel` 用 `triton.testing.do_bench` 计时；`_cudagraph_plugin` 把它换成照 `triton.testing.do_bench_cudagraph` 实现的 CUDA graph 计时：一张 graph 里连续展开覆盖 100 ms 重复预算的调用，replay 10 次，取单次调用的中位数。测的是去掉 launch 开销的 kernel 时间。（上游自带的 `--mode cudagraph` 既不预热也不回退，所以不用。）

`do_bench_cudagraph` 只预热 5 次，对首次调用要编译 autotune config、做 libtuner 选择或 JIT 初始化的算子远远不够。这些一次性开销漏进捕获的 graph 或第一个计时迭代时，latency 在不同运行之间抖动；M=1 的多 kernel 路径上，首轮可能只有稳态的三分之一左右。因此插件依次：

1. 同步一次，确保默认流上还在生成的输入在侧流 capture 前已完成；
2. 运行一次并丢弃结果，吸收编译；
3. 按调用方的预热预算（`Config.warm_up`）循环运行，次数由一次稳态运行的耗时换算，并限制在 5–200 次；
4. 在 torch 的同步调试模式（设为 error）下再运行一次，检查是否与 host 同步（见[回退](#回退到-do_bench)）；
5. capture 并计时。

新算子若仍有首轮抖动，先调大 `_warmup_before_capture` 里的次数上限。

### graph 大小

capture 耗时与 graph 里的调用次数成正比，而有些算子每次调用的 capture 代价很高：`conv1d_padding` 约 12 ms，凑满 100 ms 要几千次调用，每次测量就要半分钟以上；记录它的 13 个 fp16 shape，不设上限用了 67 分钟，设上限后 30 分钟。因此插件每张 graph 最多放 `GRAPH_MAX_CALLS`（默认 1000）次调用，在 10 个计时样本的每一个里连续 replay 这张 graph 若干次来凑满预算，每个样本覆盖的调用次数不变。

每张 graph 有 250 次及以上调用时，连续 replay 不会带来可测的额外耗时：64 个元素的 `abs` 在每张 graph 250、500、1000 次调用和不设上限时都是每次 1.03 µs，4096×4096 时各种大小都是 23.0 µs。再小的 graph 会把极小的 kernel 测得偏快（100 次调用时 0.86 µs），所以上限不要低于 250。graph 大小本身会让少数算子略有偏移：`conv1d_padding` 每张 graph 500 次调用时每次 45.9 µs，2000 次时 46.8 µs，因为每次调用的输出都会让 graph 的私有显存池变大。A/B 两侧用同样的方案，对比不受影响；这类算子的绝对值与不设上限的 `do_bench_cudagraph` 可能相差这么多。`GRAPH_MAX_CALLS=0` 恢复不设上限的 graph。

### 回退到 do_bench

有些算子无法 capture：host 同步、动态显存分配、autograd backward、不合法的流操作等。它们改用普通 `do_bench` 计时，`run.log` 里记一条 `BENCHMARK_DIRECT_NO_CUDAGRAPH op=... phase=... reason=...`，`phase` 是拒绝该 graph 的步骤（`host_sync` 或 `capture`）。调用本身就失败（发生在 warmup 阶段、还没开始 capture）不属于这种情况，它像在 `do_bench` 下一样直接抛出：autotuner 对 sweep 里的每个 config 都通过 `do_bench` 计时，抛异常的直接丢弃（shared memory 超出 GPU 上限、编译失败），在那里回退只会把每个被拒的 config 再跑一次，还把它报告成一次无法 capture 的测量。以下措施保证回退可用：

- **host 同步在 capture 前查出。** 与 host 同步的调用（`tensor.item()`、`nonzero`、device 到 host 的拷贝）永远无法 capture。在 `torch.cuda.set_sync_debug_mode("error")` 下运行一次即可查出，不必启动 capture，测量直接走 `do_bench`（`phase=host_sync`）。
- **释放失败 capture 的显存。** 中途失败的 capture 会让它的 graph 和私有显存池留在循环引用里，直到垃圾回收运行；输入有几十 GiB 时，几个这样的显存池就会让下一次测量显存不足。插件在回退前先回收并清空缓存。capture 显存不足时，改为每张 graph 一次调用重试一次（`CUDAGRAPH_OOM_RETRY`），仍用 graph replay 计时。
- **致命错误按错误原文匹配。** 非法地址、OOM、device-side assert 等直接抛出，不计时。torch 给每个 CUDA 错误都附加 "enable device-side assertions" 提示，宽泛的子串匹配会把所有 capture 失败都当成致命错误。
- **capture 在一次性 RNG 状态下进行。** 中途失败的 capture 可能让默认 CUDA 生成器停在 capturing 状态，此后进程里任何 `torch.randn` 都会报 "Offset increment outside graph capture"。capture 期间把生成器状态换成临时状态，结束后换回，真正的生成器始终不参与 capture。
- **capture 失败后先空跑一次。** cusolver 会把作废的 capture 留在句柄状态里，下一次调用报一次 `CUSOLVER_STATUS_EXECUTION_FAILED` 后才恢复（如 `linalg_eigvals`、`linalg_matrix_rank`）。

### 回退行的精度

回退不会让 A/B 产生偏差，因为同一个算子两侧回退方式相同；但 `do_bench` 的每次迭代都包含 Python launch 路径。亚毫秒级 kernel 上两种方法可差 2 倍以上，行间波动也更大，所以小 shape 行的绝对值和 10% 以内的差异不要过度解读。回退按测量点发生，同一张表可能混有两种方法；用 `grep BENCHMARK_DIRECT_NO_CUDAGRAPH run.log` 查看受影响的行。

## IR 落盘

`_ir_dump_plugin` 作用于本次运行独立的 Triton 缓存，以每个编译 kernel 的缓存目录为键：

- **编译监听**：在每个 kernel 旁写一个 `__constexprs.json`，记录 constexpr 取值、签名 dtype 和逐参数特化（如 `tt.divisibility`），因为单看 ttgir 分不出是哪个 autotune 变体。
- **shape 跟踪**：发布当前正在喂入的 shape。上游有两条输入路径：传统的 `get_input_iter` 生成器，以及 Case API（`supports_cases()` 为真时走 `get_case_iter` + `build_inputs`，如 `UnaryPointwiseBenchmark`、`BlasBenchmark` 族）。两条都被包装，且是原地包装，上游 `supports_cases()` 的身份判断照常成立，生成的 shape 标签一致。标签是 shape 各维用 `x` 连接；嵌套的 shape（如 `[input_shape, output_size]`）内层各维用 `-` 连接（`2-256-14-14x7-7`），超过 120 个字符的标签会被截断并以摘要结尾。传统路径上，标签是迭代器遍历 `self.shapes` 时当前走到的 shape：`get_input_iter` 运行期间，`self.shapes` 是一个会在遍历到每个 shape 时上报的列表（`_shape_scope`），所以一个 shape 产出多个输入时它们都得到这个标签，按输入个数去数就做不到。`_shape_override_plugin` 用 yaml 喂入的迭代器以同样的方式上报那些 shape。不遍历 shape 的迭代器（写死的用例列表），或在第一个输入之前就把 shape 全部走完的，第 i 个输入的标签是 `input<i>`：不指向任何 shape，好过指错。标签用来命名 `ttgir/` 下的目录并填 `shape` 列，某一行运行了哪些 kernel 与它无关。
- **启动跟踪**：按 (kernel, shape) 统计启动次数，`Autotuner._bench` 内外分开计数。在 sweep 之外运行过至少一次的变体才算"用到"。
- **行追踪**：上游为每个结果行创建一个 `BenchmarkMetrics`，几乎所有 benchmark 都是在测量该行之前创建。从这时起到下一个输入构造之前，在 sweep 之外、且在被计时函数内部的每次启动都归这一行；`_timing_scope` 标记被计时函数，计时框架自己的启动（如 capture 开始时 torch 发出的 fill）不算在内。先测量、后建行的 benchmark 也能跟上：为这个输入在计时函数里启动、而当时还没有当前行的 kernel，归给接下来建的那一行。打印结果表时按打印顺序记下它的各行。`rows.tsv` 按 (op_name, dtype) 表与 `run.log` 里的行号，列出 shape 标签和该行运行过的 kernel 的 cache hash。没有任何 kernel 是在计时函数里启动的行（不通过 `triton.testing.do_bench` 计时的 benchmark，少数自己调用 `do_bench_cudagraph`；或不加载 `_cudagraph_plugin` 的运行）取它作为当前行期间的所有启动，没有的话取建行之前为它的输入所做的启动，标为 `scope=row`。其中可能包含只用于构造输入的 kernel，偏向于把一行判为有变化。最终没有任何 kernel 的行就是没有启动 kernel（小输入可能走不经过 Triton 的路径）；门控只有在该运行用到的每个变体都能在某一行里找到时才放行这样的行。
- **二进制**：`rows.tsv` 还按 kernel 的顺序为每一行列出各 kernel 二进制的指纹（见[对照组与误报](#对照组与误报)）。cache hash 不能用于此：它覆盖编译器自身的源码，任意两个构建之间都不同。
- **退出时落盘**：每个用到的变体，ttgir 写到 `ttgir/<shape>/<kernel>/<名称>.ttgir`，ttir 和 Triton 的 metadata json 写到 `ttir/<cache hash>/`。sweep 落选者只记在 `ttgir/index.tsv` 里。在进程退出而不是 session 结束时落盘，会话因报错结束的运行也能留下 IR；进程被杀（段错误、abort、超时）则不会留下。`rows.tsv` 最先、单独写出，体积大得多的 IR 拷贝失败（`IR_DUMP_FAILED`）时不会连带丢掉行到 kernel 的映射；同一 run 目录里上一次尝试留下的 `ttgir/` 与 `ttir/` 会先删掉：批量运行在同一个目录里重试，第一次尝试的变体不属于这次运行。

文件名只保留能区分同一 (shape, kernel) 组内各变体的信息：组内取值不同的 constexpr，按词缩写（`BLOCK_SIZE_M` → `BSM`，图例见 `naming.md`），后接 `w{warps}s{stages}`。仍然重名时，依次用签名 dtype（`__fp16`）、参数特化（`__EMdiv16`）、hash 前缀区分。

`ttir/` 里的 json 是用于重编译的编译参数：即 Triton 的 metadata，但 `num_warps` 从 ttgir 模块恢复，因为 warp specialization 会把 metadata 里的值改写成各分区之和（如 12），而编译参数不接受这个值。

## 统计判定

### 每轮一个样本

每一轮里每个算子的两侧各运行一次，都是带独立 Triton 缓存的新进程，每个结果行每侧得到一个 latency。单次测量内部插件已经取了 10 次 graph replay 的中位数；轮与轮之间剩下的是运行间噪声：GPU 频率和温度、显存布局、这一轮分到哪张卡。判定要胜过的正是这部分噪声，所以重复的单位是轮，而不是 replay。

A、B 在同一张卡上紧挨着运行，抵消卡间偏差（同一台机器各卡之间几个百分点）和缓慢漂移；轮流决定哪一侧先跑，抵消 L2 预热、频率爬升这类先后顺序带来的影响。一轮只作为这样的一对来使用：某一侧失败时，`ab_run.py` 把这一轮的两侧一起重跑，并先删掉上一次尝试留在该轮 `ops_status.csv` 里的行，这样重跑在跑完一侧后被打断时，这一轮不会被当成已完成。两侧落在不同卡上的轮次（各轮 `ops_status.csv` 的 `gpu` 列），以及只有一侧有行的轮次，`ab_compare.py` 不计入。测完一部分行之后才失败的一侧算是跑完了：这些行计入，标为 `failed_side`，在复测时算弱证据。

### 判定规则

`ab_stats.compare` 根据一行的 A、B 两组样本判定：

| 判定 | 条件 |
|---|---|
| `undecided` | 不足 `--min-rounds`（6）轮 |
| `same` | \|gap\| ≤ `--threshold`（0.5%） |
| `faster` / `slower` | 秩检验显示出差距：Mann-Whitney p < `--alpha`（0.01） |
| `faster` / `slower` | 配对显示出差距：每一轮方向一致，且各轮的符号检验低于 `--alpha-paired`（0.05） |
| `same` | 两者都没有显示，且 Mann-Whitney p ≥ `--clear`（0.1） |
| `undecided` | 其余情况：有差距，但还不显著 |

各条件按表中顺序判断。`gap = median(A) / median(B) − 1`。Mann-Whitney U 检验比较的是秩，不需要对 latency 分布做任何假设；latency 分布是偏态的，有时还是双峰的（有的 kernel 在不同运行之间在两种速度间跳动）。小样本且无并列值时用精确零分布，其余用正态近似，与 `scipy.stats.mannwhitneyu` 一致。每侧 6 轮时双侧 p 最小只能到 0.0022，所以只有两组样本几乎不重叠时才达得到 `--alpha 0.01`。阈值把统计上明确、但幅度可以忽略（零点几个百分点）的偏移排除在结果之外。

### 两种检验

秩检验把各轮混在一起：它问的是 A 的样本作为一个整体是否与 B 的样本分开。同一轮两次运行共有的东西，首先是这一轮落在哪张卡上，对它来说都是噪声。各卡相差几个百分点时，即使 B 侧在每一轮都快 2%，两组样本仍然几乎完全重叠；秩检验给这样的行 p ≈ 0.2，单看它就是 `same`。

符号检验用的正是被混掉的信息：每一轮是同一张卡上背靠背测出的一对，配对抵消了卡的差异，只统计每一轮的方向。没有差异时，一轮朝哪个方向的概率各为一半；6 轮全部同向的双侧 p 为 2/2⁶ = 0.031。规则要求每一轮都同向，而不只是足够多的轮次同向，这样在两种速度间跳动的行不会因为碰巧的多数而被判定；对中位数差距的阈值要求与秩检验相同。因此 6 轮同向可以判定，5 轮不行（p = 0.0625），与 `--min-rounds` 的取值无关。`--alpha-paired 0` 关闭配对判定。

代价是水平更宽松。对没有任何差异的行做模拟（单次运行噪声 0.5–1%，两侧共有的每轮卡间偏移为 0 或 3%），6 轮时有 2–3% 的行被判为 faster/slower，只用秩检验时不到 1%；10 轮起两者都在 1% 以下。换来的是：单次运行噪声 0.5%、卡间相差 3% 时，2% 的效应有 99% 的行被检出，而不是 1%（噪声 1% 时为 61%）。CSV 的 `basis` 列写明每个 faster/slower 行由哪种检验显示，报告会统计只靠配对判定的行数，这些行在下面的复测中算作证据弱。实际 A/B 的误报率以对照组为准。

### 对照组与误报

加 `--gate` 后，kernel 在两侧编译成相同二进制的行不可能有差别；它们与有变化的行走同一条规则，其中被判为 `faster` 或 `slower` 的都是误报。这个比例就是该规则在这台机器、这份记录上的实测误报率，单靠固定的 α 得不到：并非随机噪声的运行间效应（双峰 kernel、某几轮分到偏慢的卡）会在这里体现。对照组跟着本来就要测的算子一起跑，不额外花时间。

门控的结论是一个预测：它用记录下的编译选项，在两个构建上编译了记录下的变体。对照组成立的前提是两侧实际启动的就是门控编译的东西，而这个前提可能在日志里没有任何标记的情况下不成立。算子库可能只在编译器支持时才设置某个编译选项：FlagGems-vllm 在认识该选项的 FlagTree 上用 `num_stages=2, pipeline_merge_loads="any"` 启动 `_qsa_sparse_paged_gqa_kernel`，在不认识的 FlagTree 上用 `num_stages=3`，TTIR 相同。用旧构建做的记录里只有后一种变体；门控在两个构建上重编译它，得到同一个二进制，于是放行该 benchmark 的全部 18 行，而其中 12 行新构建那一侧测的是另一个 kernel。这些行会被排除在结果之外，它们的差异最多以“误报”的形式露面。

所以证据由运行自己携带。`rows.tsv` 为每个结果行记录它启动的每个 kernel 的指纹：kernel 名、cubin 的代码和启动器读取的 launch 元数据（`_binary_id`）。行号信息不计入，因为 cubin 按路径记录源文件，其中包括 Triton 安装目录里的文件，两个位于不同目录的构建产出的 cubin 只在 `.debug_line` 上不同。`ab_compare.py --gate` 逐轮比较两侧的指纹：门控放行、但任一轮两侧运行了不同二进制的行为 **diverged**，与有变化的行一样算作结果。门控放行的行，在每一轮两侧二进制都一致时，或运行没有记录指纹且没有 replay 回退时，才是对照组；某次运行无法确定该行启动了什么（`?`），或有 replay 回退又没有指纹时，为 unknown。`ir_gate.py` 会提前指出这种情况：某个构建有记录或另一个构建没有的编译选项时给出警告，算子库正是在这种时候可能有上述行为。

### 补轮与全新轮次

`--until-decided` 给有 undecided 行的算子补轮。每补一次就重新检验，会让误报率略有上升（序贯检验）；对照组得到同样的补轮，能看出上升了多少。

接近阈值或只靠少数几轮判定的行，是用同一批样本既挑出来又判定的，小效应会被高估（赢家诅咒）。办法是独立复测：`ab_run.py --add-rounds N` 跑 N 个新轮次，`ab_compare.py --confirm` 只用这些新轮次判定每个 faster/slower 行，按原方向做单侧检验，α = 0.05，两种检验任一通过即可：秩检验，或新轮次全部同向。不论原判定来自哪种检验，两者都可以用来确认：由配对判定的行，其样本本来就重叠，新轮次里也一样，只靠秩检验永远确认不了。用单侧检验是因为方向在新数据产生之前就已确定；两次检验都通过的行，只有两次都误报才是假阳性，新轮次相互独立，这种情况很少。确认成立的行报告全部轮次上的 gap；没能复现的行改为 undecided 而不是 same，因为真实的小效应也可能在少数几轮里没测出来。新轮次太少、无论结果如何都到不了 0.05 时，既不能确认也不能否定，报告为 `too_few_rounds`，该行保持原判定：单侧检验下，n 轮用秩检验最小只能到 1/C(2n, n)（3 轮 0.05，4 轮 0.014），用配对最小到 1/2ⁿ（4 轮 0.0625，5 轮 0.031）。所以 3 轮无论结果如何都不能确认（样本有并列值时，秩检验的正态近似可能给出比这个下限更小的 p；先检查下限），4 轮只有在两侧样本完全分开时才能确认，4 轮全部同向但有重叠的同样报告为 `too_few_rounds`。

## shape 覆盖

上游在 `Benchmark.set_shapes()` 里读取 shape，回退顺序是 op_name 键 → MRO 类名键 → 基类 `DEFAULT_SHAPES`，但相当一部分 benchmark 绕开了这一步。`_shape_override_plugin` 补上三处缺口，每处都复用该类自己的输入构造，只换 shape 来源，被测 kernel 不变；yaml 里没有条目的算子保持上游 shape。

一份 shape 集同时服务两个仓库，而同一个名字在两边不一定是同一个东西：`fused_recurrent_gated_delta_rule` 在两个仓库里都是 op_name，在 FlagGems 里取序列长度，在 FlagGems-vllm 里取 case 名，哪一方拿到另一方的条目都会解包失败。因此键可以加限定，`vllm:<名字>` 或 `main:<名字>`：只对该仓库生效，并在那里优先于普通键。上游自己读取文件，不认识限定键，所以 `run_pytest.sh` 把 shape 集按被测仓库解析（`shape_set.py`）后交给上游，并存档为本次运行的 `shapes.yaml`。`gen_inventory.py --refresh-shapes` 首次并入某个 core_shapes 键时，如果另一个仓库也用这个名字却没有定义这个键，或者定义了不同的 shape，就给它加限定；集合里已有的键保持原来的形式。

1. **在 `set_shapes()` 里写死 shape**（如 `fused_marlin_moe_*`）：包装在 `set_shapes()` 之后运行的 `Benchmark.init_user_config`，把 yaml 的 shape 放回去。子类自己的 `init_user_config` 只给 `self.shapes` 赋值而从不读取它时（`adaptive_max_pool3d`、`beam_search_score`）也同样处理：不管 yaml 给了什么，它留下的都是写死的。读取 `self.shapes` 的则是在对 yaml 给的 shape 做筛选或校验，不去动它。Case API 的 benchmark 遍历 `self.shapes`，同样被覆盖到。子类在读取 yaml 后统一了 shape 维数时（BLAS 的 (B, M, N, K) ↔ (M, N, K)）不做覆盖，判断依据是两边各自维数一致但互不相同。
2. **`get_input_iter` 里写死了列表、完全不读 `self.shapes`**（conv padding、pool、nll、scaled_softmax 等族）：按 yaml 的每个 shape，经该类标准的 `input_fn(shape, dtype, device)` 生成输入。没有这种 `input_fn` 的类保持不变。
3. **特殊的输入构造**，按类名注册：`UpsampleBicubic2dAaBackwardBenchmark`、`FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark` 和 `FlashmlaSparseBenchmark` 用配置表或 `TestParam` 对象构造输入；`CutlassScaledMMPerfKit` 从一个 (M, N, K) 列表中抽样 case，替换的是这个列表，量化模式的组合保持上游定义。这些依赖上游内部实现，不匹配时表现为 benchmark 失败，而不是悄悄换了输入。

完全不读 shape 的 benchmark，如按 case 参数化的 `scaled_mm_benchmark`，始终跑内置 shape。

### 一次运行实际测了哪些 shape

一份什么都没驱动的 yaml，看上去和起作用的 yaml 完全一样：上游静默回退，运行照样通过。键是 benchmark 的 op_name，它不一定等于测试函数名，有些 benchmark 在两个仓库里还不一样（`test_fused_marlin_moe_w8a16_int8` 在 FlagGems 里是 `fused_marlin_moe_w8a16_int8`，在 FlagGems-vllm 里是 `fused_marlin_moe_w8a16`），上游也时不时改 op_name。所以插件在每个 benchmark 的 `init_user_config` 返回之后（包括子类重写的版本）记下它的 shape，并在 session 结束时为每个 benchmark 打印一行来源：`shapes of mm: 1 of the 1 of key 'mm'`、`... of class key 'Benchmark'`，或 `its own 6 (class key 'Benchmark' not used)`。

只有当这个来源不可能是 shape 文件作者的本意时，才打印 `SHAPE_KEYS_UNUSED`（批量运行会把它带进备注列）：

- 没有任何键适用于该算子，op_name 和 MRO 上的类名都没有（自定义集的键挂在别的名字上）；
- 文件里有以测试名为键的条目，但 benchmark 实际使用的 op_name 没有条目；
- 算子有条目，但 benchmark 没接受它、换回了自己的默认 shape（FlagGems-vllm 的 `chunk_*` benchmark 在 shape 的长度与它解包的不符时就这样做），或者它构造输入时不读 shape，又没有标准的 `input_fn` 可以改道。

shape 是条目的子集不算问题：有几个 benchmark 会去掉显存放不下的 shape。回退到类名键是上游的设计，不标记，即使 benchmark 写死了 shape、因而也没有读这个键：全量 shape 集里大多数算子本来就没有自己的条目，逐个标记会把上面这些情况淹没。`gen_inventory.py --check-shapes` 不用运行就能查出同样的漂移。

## 多卡并行 sweep

shape 多时 sweep 占主导：`fused_marlin_moe_w4a16_mxfp4` 的 53 个 shape，record 一次用了 3755 秒，而被测 kernel 的耗时合计只有 49 毫秒，2858 个编译变体中 2645 个是 sweep 落选者；用同样的 config replay 只需 196 秒。

`_parallel_warmup_plugin` 在 record/replay 插件配置之前运行。它把 shape 集里每个条目的 shape 轮转分片（shape 列表通常有序，按连续块切会让一个分片拿到所有大 shape），每个分片起一个 pytest 子进程并绑定一张卡，把各分片的记录合并进本次运行的 `autotune_records/<op>.json`，再把本次运行切换为 replay 这些 config。之所以每个条目都分，是因为 benchmark 读的条目以它的 op_name 为键（不一定等于测试名），也可能是类名键；都按同样方式分，无论它读哪个，各分片合起来都能覆盖。shape 数少于分片数的条目会在多出的分片里重复，而不是缺席：缺席时上游会回退到没人指定的 shape。随后的测量走普通的单卡路径。用子进程而不用 pytest-xdist，是因为 import FlagGems 会创建 CUDA context，fork 出的 worker 无法沿用。

测量本身从不并行：多个进程压满同一台机器的多张卡时，会通过功耗墙和散热互相影响，单卡 latency 取决于邻居。sweep 中同样的影响可能改变哪个 config 胜出。同一个键在不同分片选出不同 winner 即为一次冲突，保留先出现的值，日志报告冲突数。冲突数就是这次加速的可信度：为 0 没问题，多了就要在下结论前去掉该变量重跑。分片绑卡遵守继承来的 `CUDA_VISIBLE_DEVICES`；如果没有分片返回 config，本次运行回退为串行调优，并保留分片日志。

## 对上游的依赖

- 上游 benchmark 体系：`benchmark/base.py` 的 `Benchmark` 类（`set_shapes`、`init_user_config`、`get_input_iter`，主仓另有 Case API），conftest 的 `--shape_file`、`--level`、`--mode`、`--dtypes` 选项，`consts.BenchmarkResult`（打印它即输出一张结果表）和 `consts.BenchmarkMetrics`（每个结果行一个）。行追踪既能跟上测量之前创建的行（基类的做法），也能跟上测量之后创建的行；shape 标签依赖输入迭代器遍历 `self.shapes`。两个仓库各带一份，插件使用正在运行的那个 `benchmark` 包。
- LibTuner 位于 `<包>.utils.libentry`；`_autotune_record_plugin` 给实际被 import 的包（`flag_gems` 或 `flaggems_vllm`）打补丁。
- Triton 提供 `knobs.compilation.listener`、`kernel_load_end_hook` 和 `launch_enter_hook`（FlagTree 的 Triton 3.6 满足）。
- 插件通过 monkeypatch 挂钩上游内部结构。上游大版本更新后行为异常时，先检查各插件启动时是否仍打印注册信息。
