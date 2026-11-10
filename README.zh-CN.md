<div align="center">

# ZL_BENCH — 在 FlagGems 上对 Triton 编译器改动做算子级 A/B 性能测试

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Triton 3.6 (FlagTree)](https://img.shields.io/badge/Triton-3.6%20%28FlagTree%29-6E40C9)](https://github.com/flagos-ai/FlagTree)
[![GPU: NVIDIA Hopper](https://img.shields.io/badge/GPU-NVIDIA%20Hopper-76B900?logo=nvidia&logoColor=white)](#环境要求)
[![Built on FlagGems](https://img.shields.io/badge/built%20on-FlagGems-0969DA)](https://github.com/flagos-ai/FlagGems)
[![Built on FlagGems-vllm](https://img.shields.io/badge/built%20on-FlagGems--vllm-0969DA)](https://github.com/flagos-ai/FlagGems-vllm)
[![ShellCheck](https://img.shields.io/badge/shell-ShellCheck-4EAA25?logo=gnubash&logoColor=white)](https://www.shellcheck.net)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](tests)

[English](README.md) | **简体中文**

zl_bench 是基于 [FlagGems](https://github.com/flagos-ai/FlagGems) 与 [FlagGems-vllm](https://github.com/flagos-ai/FlagGems-vllm) benchmark 的一组 pytest 插件和驱动脚本，用来测量 [FlagTree](https://github.com/flagos-ai/FlagTree)（Triton 编译器）的改动对算子性能的影响，并把通常会淹没这类对比的噪声来源控制住。

</div>

## 目录

- [功能特点](#功能特点)
- [测试版本](#测试版本)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [使用](#使用)
- [配置](#配置)
- [产物](#产物)
- [跟进上游更新](#跟进上游更新)
- [常见问题](#常见问题)
- [开发](#开发)
- [许可证](#许可证)

## 功能特点

- **两侧同 config。** record/replay 锁定 Triton autotune 与 FlagGems libtuner 的选择，A/B 测到的是编译器改动，而不是换了一套 config。不锁定时，同一份代码跑两次也可能差 10% 左右。
- **低噪声计时。** 用 CUDA graph 计时，capture 前显式预热，launch 开销和首次调用的编译都不进数字。无法 capture 的调用（如与 host 同步的）回退到 `do_bench`，并在日志里打标记。
- **每次测量都留 IR。** 每次运行保留实际用到的 ttgir（按 shape 分组、命名可读）、可离线重编译的 ttir 和编译参数，以及每个结果行实际运行了哪些 kernel。
- **编译差异门控。** 用两个编译器构建把记录下的 ttir 一直编译到 cubin，只对 kernel 有变化的结果行做 A/B，比跑全量快一到两个数量级。
- **统计化 A/B。** 两侧在同一张卡上交替运行多轮；每个结果行用秩检验给出判定，判不了的自动补轮；二进制没变的行作为对照组，实测误报率。
- **批量筛查。** 在多卡上跑两个仓库的 1600 多个 benchmark，支持超时、重试、断点续跑和逐行汇总。
- **shape 由 yaml 说了算。** 一份 yaml 决定所有算子的 shape，包括在代码里写死 shape 的 benchmark。

实现原理见 [docs/design.zh-CN.md](docs/design.zh-CN.md)。

## 测试版本

zl_bench 会 monkeypatch 上游内部实现，可用性与测试时的上游版本绑定。本 README 的所有内容，包括[全量参考](#批量测试)和 `ops/dismiss.txt`，都基于以下 commit：

| 组件 | Commit | 日期 | 来源 |
|---|---|---|---|
| FlagGems | [`a4e2556d5`](https://github.com/flagos-ai/FlagGems/commit/a4e2556d5) | 2026-11-04 | `flagos-ai/FlagGems` master |
| FlagGems-vllm | [`a579b7c`](https://github.com/flagos-ai/FlagGems-vllm/commit/a579b7c) | 2026-11-10 | `flagos-ai/FlagGems-vllm` main |
| FlagTree | [`6d7597f66`](https://github.com/flagos-ai/FlagTree/commit/6d7597f66) | 2026-11-10 | `flagos-ai/FlagTree` main |

环境：NVIDIA H20（Hopper，驱动 595.45）、CUDA 13.0、PyTorch 2.11.0、Python 3.12、vLLM 0.20.2。换用其他上游版本时，先重建算子清单、刷新 shape 集（见[跟进上游更新](#跟进上游更新)）；每个 `run.log` 都会记录实际使用的 Triton 路径与 git 版本。

## 环境要求

- NVIDIA GPU（在 H20 / Hopper 上开发和测试）
- Python 3.10+ 与 PyTorch
- FlagTree 构建的 Triton 3.6（插件用到 `knobs.compilation.listener` 和 kernel launch 钩子）
- FlagGems 和/或 FlagGems-vllm 的源码

## 安装

把 zl_bench 放在两个 FlagGems 仓库旁边；放在别处时，用 `FLAGGEMS_MAIN_DIR` / `FLAGGEMS_VLLM_DIR` 指定它们的位置。

```
<workspace>/
├── zl_bench/
├── FlagGems/          # 包名 flag_gems
└── FlagGems-vllm/     # 包名 flaggems_vllm
```

```bash
cd <workspace>
pip install -r zl_bench/requirements.txt
pip install -e FlagGems --no-deps         # --no-deps 避免动 FlagTree 的 triton
pip install -e FlagGems-vllm --no-deps
```

被测编译器就是 `import triton` 解析到的那个构建。不安装就测某个 FlagTree 构建时，把它的 `python/` 目录放到最前：`PYTHONPATH=<FlagTree>/python`。

## 快速开始

先用 `nvidia-smi` 确认 GPU 空闲，再测一个算子：

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

run 目录里有日志、本次选中的 autotune config 和 IR（见[产物](#产物)）。这是一次 **record** 运行：现场调优，两次 record 的数字不能互相比较。对比编译器请按 [A/B 对比](#ab-对比) 来做。

## 使用

### 选择算子

一个 benchmark 对应 `$FLAGGEMS_DIR/benchmark/test_<OP_FILE>.py::test_<OP>`：`OP` 是测试函数名去掉 `test_`，`OP_FILE` 是文件名去掉 `test_` 和 `.py`（可带子目录，如 `FLA/test_xxx`）。`ops/inventory_main.csv` 与 `ops/inventory_vllm.csv` 列出了所有算子及其 `op_file`。

```bash
OP=softmax OP_FILE=softmax bash run_pytest.sh                            # FlagGems
FLAGGEMS_DIR=../FlagGems-vllm OP=fused_marlin_moe_w4a16_mxfp4 \
  OP_FILE=fused_marlin_moe_w4a16_mxfp4 bash run_pytest.sh                # FlagGems-vllm
SHAPE_FILE=shape/DeepSeek-V4-Flash-p32768d1024.yaml bash run_pytest.sh   # 换一个 shape 集
```

两个仓库有同名但实现不同的算子，测哪个由 `FLAGGEMS_DIR` 决定。测量始终是单进程单卡，这是计时可比的前提；用 `CUDA_VISIBLE_DEVICES` 选卡。`run.log` 开头记录 triton 路径与 git 版本，以及 `FLAGTREE_*`/`TRITON_*`/`MLIR_*` 环境变量，A/B 两侧对账时先核对这几行。

### A/B 对比

在基线编译器上 record，再在改动后的编译器上 replay 记录下的 config：

```bash
# A：基线编译器，记录 autotune 选择
PYTHONPATH=/path/FlagTree-base/python bash run_pytest.sh             # -> runs/<op>_<ts_A>/
# B：改动后的编译器，replay A 的选择
PYTHONPATH=/path/FlagTree-dev/python REPLAY_FROM=$PWD/runs/<op>_<ts_A> bash run_pytest.sh
# 逐行对比 Gems latency，比值 A/B（>1 表示 B 更快）
python ab_compare.py runs/<op>_<ts_A> runs/<op>_<ts_B>
```

这种单次对比只适合快速看一眼。同一台机器的各卡之间有几个百分点的系统偏差（8×H20 上实测 3–4%），单行也有噪声，结论要用[多轮重复测量](#重复测量-ab)得出。

- B 侧必须用 `REPLAY_FROM`：两次 record 各自调优，可能选出不同 config，这个差异足以掩盖被测改动（[原因](docs/design.zh-CN.md#为什么要-replay)）。
- 不设 `SHAPE_FILE` 时，B 侧沿用 A 侧存档的 `shapes.yaml`，以后刷新 shape 集也不影响老基线。
- 两侧的 `USE_FLAGTUNE` 必须相同。
- `REPLAY_FROM` 要指向最初 record 的那次运行：replay 不产生 `autotune_records/`。
- 看 IR 差异就 diff 两次运行的 `ttgir/<shape>/<kernel>/`。

**留意 `run.log` 里的标记。** `AUTOTUNE_REPLAY_FALLBACK` 表示记录里缺这个 config，或它在新编译器下编译失败，该测量点改为现场调优，不再用 A 侧的 config（`ab_compare.py` 会给这类算子标 `[replay_fallback]`）；带 `_no_evict` 后缀时重试仍读到了那个坏 config，该点按未验证处理。`BENCHMARK_DIRECT_NO_CUDAGRAPH` 表示该调用无法 graph capture（`phase=host_sync`：它与 host 同步），该点用普通 `do_bench` 计时；两侧回退一致，但这些数字噪声更大，亚毫秒级 kernel 尤其明显（[详情](docs/design.zh-CN.md#计时)）。

### 重复测量 A/B

`ab_run.py` 在两套编译器配置上多轮 replay 同一份记录（record 模式下得到的 run 或批次目录，通常用 A 侧的编译器记录）；`ab_compare.py` 把多轮数据变成每个结果行的判定。

```bash
# A 侧与 B 侧：各给一个 PYTHONPATH，或同一构建加编译开关（--a-env / --b-env）
python ab_run.py --record runs/<record> --ops softmax mm \
  --a-pythonpath /path/FlagTree-base/python --b-pythonpath /path/FlagTree-dev/python \
  --rounds 6 --until-decided --out runs/ab_dev
python ab_compare.py runs/ab_dev -v --csv runs/ab_dev/rows.csv
```

- 每个（算子, 轮次）在同一张卡上先后跑 A 和 B，奇数轮先 A、偶数轮先 B，抵消卡间偏差和缓慢漂移。所有任务共用一个队列、分到所有卡上，耗时长的先跑；每次运行都是新进程，各轮是独立样本。每次运行的处理与[批量测试](#批量测试)相同（超时、重试、空闲检查）。`<out>/r<k>_a` 与 `r<k>_b` 的结构与批次目录相同；用同一个 `--out` 重跑即断点续跑。
- 每行的判定（[详情](docs/design.zh-CN.md#统计判定)）：至少 `--min-rounds`（6）轮，中位数之差超过 `--threshold`（0.5%）且双侧 Mann-Whitney 检验 p < `--alpha`（0.01）时为 `faster` / `slower`；差距在阈值内或 p ≥ `--clear`（0.1）时为 `same`；其余为 `undecided`。`--until-decided` 给有 undecided 行的算子每次补 `--step` 轮，直到 `--max-rounds`（14）。（判为 `undecided` 的行，若各轮方向一致且差距超过 `--threshold`，改用配对判定：每轮两侧在同一张卡上背靠背运行，配对可抵消卡间差异与慢漂移；6 轮同向的双边符号检验 p = 0.031，即 `--alpha-paired`。CSV 中为 `p_paired` 列。）
- `gap = median(A) / median(B) − 1`，B 更快时为正。它是在该记录的 config 下编译器改动的效果，而不是两侧各自最优之间的差（[原因](docs/design.zh-CN.md#两侧各自最优)）。
- 加 `--gate <ir_gate 输出目录>`（见下一节）后，kernel 在两侧编译成同一二进制的行作为对照组：它们不可能有差别，`ab_compare.py` 会报告其中被判为 faster 或 slower 的数量，即实测误报率。
- **证据弱的结果用全新的轮次复测。** 差距小或噪声大的行，是用同一批样本既挑出来又判定的，会被高估。`ab_compare.py --recheck-list` 写出含这类行（|gap| < 2%、|gap| 小于样本变异系数的 2 倍，或用 do_bench 计时）的算子；`ab_run.py --add-rounds 6` 给它们补新的轮次，编号接在已有轮次之后；`ab_compare.py --confirm <起>-<止>`（`ab_run.py` 会打印这个范围）先用其余轮次判定每一行，faster/slower 的行只有在新轮次单独也能复现（单侧，p < 0.05）时才保留：

  ```bash
  python ab_compare.py runs/ab_dev --gate runs/gate_dev --recheck-list runs/ab_dev/recheck.txt
  python ab_run.py <same sides and record> --out runs/ab_dev --ops-file runs/ab_dev/recheck.txt --add-rounds 6
  python ab_compare.py runs/ab_dev --gate runs/gate_dev --confirm 15-20 --csv runs/ab_dev/rows.csv
  ```

  `--until-decided` 要在复测之前跑；有了复测轮次之后，分析时用 `--confirm` 把它们分开。

### 编译器改动：先过门控

改了 FlagTree pass 后，大多数 kernel 编译出的二进制不变，全量 A/B 既慢，测到的又主要是噪声。先过门控：

```bash
# 1. 用基线构建在多 shape 集上记录一次批次；每个 run 目录自带 ttir/ 与 rows.tsv
PYTHONPATH=/path/FlagTree-base/python python run_batch.py --all --shape-file shape/all_ops_multi.yaml \
  --dtypes "bfloat16 float16" --op-timeout 7200 --batch-dir runs/base_multi

# 2. 两个构建分别把记录下的 ttir 编译到 cubin（不用 GPU，几分钟）
python ir_gate.py runs/base_multi --base /path/FlagTree-base/python --new /path/FlagTree-dev/python \
  --out runs/gate_dev
#    同一构建比较编译开关：两侧给同一路径，再加 --base-env FLAGTREE_XXX=0 --new-env FLAGTREE_XXX=1

# 3. 对有变化行的算子做重复测量 A/B，replay 基线的 config
python ab_run.py --record runs/base_multi --ops-file runs/gate_dev/changed_ops.txt \
  --a-pythonpath /path/FlagTree-base/python --b-pythonpath /path/FlagTree-dev/python \
  --dtypes "bfloat16 float16" --gate runs/gate_dev --until-decided --out runs/ab_dev
python ab_compare.py runs/ab_dev --gate runs/gate_dev -v --csv runs/ab_dev/rows.csv
```

- 在多 shape 集上记录。kernel 变体取决于 shape，只有其他 shape 才用到的变体在单 shape 记录里看不到，门控只能排除记录到的东西。
- 门控重编译测量实际用到的变体，比较 cubin 和 launch 元数据（shared memory、warp 数）。两侧相同就说明测到的 kernel 是同一个二进制，无需再测。只看 TTGIR 不够：指令顺序和后续阶段的变化会改变二进制，而 TTGIR 的 op 不变。`variants.tsv` 逐个变体给出最先出现差异的阶段，TTGIR 是变了、只是重排还是没变，以及数量有变化的 TTGIR op。`--stage ttgir` 只比到 TTGIR，用于快速初看；不给 `--base` 时以记录时落盘的 ttgir 为基线，同样只能比到 TTGIR。
- 门控的 `rows.tsv` 按结果行标记：该行运行过的 kernel 有一个变了，这一行就算有变化（依据 run 目录自己的 `rows.tsv`）。这样某算子只有一个 shape 的 kernel 变了时，它的其他行不会被当成结果，而是作为对照组。没有行追踪的旧记录按算子判断。
- 二进制变了不代表性能一定变，由第 3 步判断。大幅波动先查 SASS：仅 ptxas 调度不同，就可能让同一份 TTGIR 在另一个构建上慢一倍（例如 load 批量方式不同）。

### 批量测试

`run_batch.py` 对选中的每个算子调一次 `run_pytest.sh`，按卡分片，每张卡上串行。

```bash
# 两个仓库所有可收集的算子，排除无条件 skip 与 ops/dismiss.txt 里的
python run_batch.py --all --gpus 0,1,2,3,4,5,6,7 --dtypes bfloat16
# 子集：默认 main 仓，另一个仓库加 vllm: 前缀；同名测试用 @op_file 区分
python run_batch.py --ops softmax vllm:fused_marlin_moe_w4a16_mxfp4 nextafter_@nextafter_
# 每个算子测多个 shape
python run_batch.py --ops-file my_ops.txt --shape-file shape/all_ops_multi.yaml
# 断点续跑：复用 PASS/SKIP，重跑 FAIL/TIMEOUT
python run_batch.py --all --batch-dir runs/batch_xxx --resume
# 逐算子 replay 之前的批次（shape 默认也沿用那次的）
python run_batch.py --ops-file my_ops.txt --replay-root runs/batch_xxx
# 给每个算子加环境变量，如编译开关
python run_batch.py --ops-file my_ops.txt --env FLAGTREE_XXX=1
```

- 每个算子一个独立进程，CUDA crash 只影响该算子；`<batch>/<repo>/<op>/` 与单算子运行的目录结构相同。
- `--op-timeout`（默认 1800 秒）杀掉卡住的算子，记为 `TIMEOUT`，不重试。测试失败和被信号杀掉最多重试两次，连续两次退出码相同就停止。
- 指定 `--dtypes` 时，算子不支持该 dtype 就去掉限制重跑（`dtype_fallback`）；其他失败且没有任何 SUCCESS 行时，再不加限制跑最后一次（`dtype_rescue`）。
- 运行时 skipif 和 xfail 都算 SKIP。只有 PASS 和 SKIP 会标记为完成，供 `--resume` 复用。
- `ops_status.csv` 每个算子一行（状态、耗时、行数、备注），边跑边更新；`summary.csv` 每个结果行一行，`batch_summary.py` 可为旧批次重新生成。备注列统计 `no_cudagraph_rows` 与 `replay_fallback`，解读结果时先看这一列。
- 算子开始前，它的卡必须空闲：`nvidia-smi` 里没有进程，且已用显存不到 1 GiB（其他容器的进程不会列出）。同卡上的其他进程会同时干扰计时和调优，所以批次会等待（`--busy-gpu wait`，默认；`warn` 照常测量并在备注里记 `gpu_busy`）。卡号与 `nvidia-smi` 一致（`CUDA_DEVICE_ORDER=PCI_BUS_ID`）。
- 批量数字只用于筛查：多张卡同时跑会通过功耗和散热互相影响。可疑算子用[重复测量 A/B](#重复测量-ab) 确认。

**全量参考**（[测试版本](#测试版本)，7×H20，`--dtypes bfloat16`，单 shape 集）：`--all` 选出 1608 个算子，约 2.5 小时，每算子中位 12 秒（主要是进程启动与 import）。PASS 1529、SKIP 43，其余 36 个因上游或环境问题失败，已列入 `ops/dismiss.txt`。SKIP 都是上游 skipif（缺 TransformerEngine 或 vLLM 参考实现、只支持其他厂商或架构）。多 shape 集上（275 个算子，`--dtypes "bfloat16 float16"`）每个算子中位约一分钟，record 模式下最长要两小时（`chunk_gdn2`），所以要设 `--op-timeout 7200`。

### 多卡加速 sweep

shape 多时，墙上时间主要花在 autotune sweep 而不是测量上。`PARALLEL_WARMUP_GPUS=N` 让 sweep 在 N 张卡上并行，之后照常在一张卡上串行测量：

```bash
PARALLEL_WARMUP_GPUS=8 SHAPE_FILE=shape/DeepSeek-V4-Flash-p32768d1024.yaml bash run_pytest.sh
```

产物、replay 和结果表都不变。并行的分片会互相干扰，可能改变哪个 config 胜出，这时日志会警告：

```
WARNING 1 config key(s) got different winners across shards — parallel interference reached the sweep
```

没有冲突时 config 可信；冲突多时，下结论前去掉 `PARALLEL_WARMUP_GPUS` 重跑。设了 `REPLAY_FROM`、可见卡少于两张，或 shape 集里没有以 `OP` 为名的条目时，插件不起作用（[详情](docs/design.zh-CN.md#多卡并行-sweep)）。

## 配置

`run_pytest.sh` 通过环境变量配置；`run_batch.py` 根据自己的参数为每个算子设置这些变量（见 `--help`）。上游的 level 与 mode 固定为 `--level core --mode kernel`，要改就编辑 `run_pytest.sh` 里的 pytest 命令。

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `OP` | `fused_marlin_moe_w4a16_mxfp4` | 测试函数名去掉 `test_` |
| `OP_FILE` | `fused_marlin_moe_w4a16_mxfp4` | benchmark 文件名去掉 `test_` 和 `.py` |
| `FLAGGEMS_DIR` | `$FLAGGEMS_MAIN_DIR`，未设时 `../FlagGems` | 被测算子所在的仓库 |
| `FLAGGEMS_MAIN_DIR` / `FLAGGEMS_VLLM_DIR` | `../FlagGems` / `../FlagGems-vllm` | 两个仓库的位置，供 `run_batch.py` 与 `gen_inventory.py` 使用 |
| `SHAPE_FILE` | record：`shape/all_ops_multi.yaml`；replay：`$REPLAY_FROM/shapes.yaml` | shape 集；没有条目的算子用上游默认 shape |
| `DTYPES` | 空：上游的 dtype 扫描 | 空格分隔的 dtype 白名单，如 `"bfloat16 float16"` |
| `REPLAY_FROM` | 空：record 模式 | 要 replay 其 autotune 选择的 record 目录 |
| `USE_FLAGTUNE` | `0` | `1` 使用 FlagTune 扩展调优空间（首跑全量搜索，很慢）；A/B 两侧须相同 |
| `PARALLEL_WARMUP_GPUS` | 空：串行 | 把 autotune sweep 分到 N（≥2）张卡上 |
| `GRAPH_MAX_CALLS` | `1000` | 一张 CUDA graph 里最多展开的调用次数；`0` 表示不设上限（[详情](docs/design.zh-CN.md#graph-大小)） |
| `OUT_DIR` | `runs/<op>_<时间戳>` | 产物目录 |
| `FLAGGEMS_PERF_COLOR` | 空：按 tty 判断 | `always`/`never` 控制终端颜色；`run.log` 始终是纯文本 |

### shape 集

shape 集是一个 yaml，顶层键是 op_name（传给上游 `Benchmark` 的 `op_name`，多数与 `OP` 相同）或 Benchmark 类名：

```yaml
softmax:
  shapes:
    - [1024, 1024]
    - [64, 512, 512]
```

| 文件 | 内容 |
|---|---|
| `shape/all_ops_single.yaml` | 每个算子一个精选 shape；`run_batch.py` 的默认 |
| `shape/all_ops_multi.yaml` | 每个算子多个 shape；`run_pytest.sh` 的默认 |
| `shape/DeepSeek-V4-Flash-p*d1024.yaml` | 来自 DeepSeek-V4-Flash trace 的 marlin MoE shape（同名 `.txt` 为原始 trace），四种量化方案共用 |
| `shape/DeepSeek-V4-Flash-sweep.yaml` | 同一 MoE 配置，num_tokens 取 2 的幂 |

- 两个 `all_ops_*` 集都包含上游 `core_shapes.yaml` 的键（单 shape 集每个键只留一个 shape）。`--shape_file` 会整体替换 core_shapes 而不是叠加，缺少 `BlasBenchmark` 这类类名键时，BLAS 族会退到 1 维 shape 而出错。
- 没有条目的算子跑上游默认 shape。对在 `set_shapes()` 里写死 shape 的算子，这意味着跑完全部写死的 shape（marlin 是 16 组大 MoE shape），所以新增或改名的这类算子要在两个集里都补上条目。
- 按 case 参数化、完全不读 shape 的 benchmark（如 `scaled_mm_benchmark`）始终跑内置 shape（[详情](docs/design.zh-CN.md#shape-覆盖)）。

### 插件

`run_pytest.sh` 从 `PLUGINS` 数组加载插件，注释掉一行即停用对应插件。各插件的作用和何时关闭见 [docs/design.zh-CN.md](docs/design.zh-CN.md#插件)。

## 产物

```
runs/softmax_20260716-031752/
├── run.log              # 完整日志，含结果表，已去 ANSI 颜色
├── shapes.yaml          # 本次使用的 shape 集
├── autotune_records/    # record 模式选中的 autotune config，供 REPLAY_FROM 读取
│   └── softmax.json
├── rows.tsv             # 每个结果行：shape 与它运行过的 kernel 的 cache hash
├── ttgir/               # 实际用到的变体的 IR，按 shape 分组
│   ├── 1024x1024/softmax_kernel_inner/w4s3__bf16.ttgir
│   ├── index.tsv        # 所有编译变体：shape、文件、启动次数、constexpr、cache hash
│   └── naming.md        # 文件名缩写图例
└── ttir/<cache hash>/   # 实际用到的变体的 ttir 与编译参数，ir_gate.py 的输入
```

- 只写测量实际用到的变体；sweep 落选者只出现在 `index.tsv` 里。
- 文件名列出同一 (shape, kernel) 组内取值不同的 constexpr，并缩写（`BLOCK_SIZE_M` → `BSM`），后接 `w{warps}s{stages}`；仍重名时依次用 dtype、参数特化或 hash 区分（[详情](docs/design.zh-CN.md#ir-落盘)）。
- 在输入构造和测量之外启动的 kernel（如模块级探测）归到 `shape_unknown/`。
- `rows.tsv` 的顺序与 `run.log` 里的结果表一致：某张表的第 `i` 行对应它测量时启动的 kernel（scope 为 `timed`），`ir_gate.py` 据此把变化归到结果行。不通过 `triton.testing.do_bench` 计时的 benchmark，取该行测量期间启动的所有 kernel（scope 为 `row`）。
- IR 在进程退出时写出，CUDA crash 后也会写。

## 跟进上游更新

1. 更新两个仓库（fork 用 `git fetch upstream && git merge --ff-only upstream/<分支>`）。
2. 重建算子清单、刷新 shape 集：

   ```bash
   python gen_inventory.py --verify-collect
   python gen_inventory.py --refresh-shapes shape/all_ops_single.yaml --one-shape
   python gen_inventory.py --refresh-shapes shape/all_ops_multi.yaml
   ```

   自定义 shape 集（`all_ops_*` 以外的）不会被 `--refresh-shapes` 重写，改用 `--check-shapes` 检查并按提示修正：

   ```bash
   python gen_inventory.py --check-shapes shape/DeepSeek-V4-Flash-p4096d1024.yaml --check-shapes my_shapes.yaml
   ```

   匹配不到当前 op_name 的键**在运行时被静默忽略**：上游回退到基准自带的硬编码 shape，运行看起来完全正常，但测的不是指定的 shape。运行时还会在本次运行的算子找不到条目时打印 `SHAPE_KEYS_UNUSED`，在 IR 落盘或行到内核的映射写入失败时打印 `IR_DUMP_FAILED` / `ROWS_TSV_FAILED`，两者都可在 `run.log` 中 grep。

   `--refresh-shapes` 只保留上游仍存在的 op_name 键，并同步 core_shapes 的键。改名算子的键会移到 `shape/<name>.unmatched.yaml`：找到新名（`grep -rn 'op_name=' $FLAGGEMS_DIR/benchmark/test_<op>.py`）后挪回，再删掉这个文件。
3. 跑一次全量筛查。确认为上游或环境问题的失败写进 `ops/dismiss.txt` 并注明原因与日期；也定期用 `--ops-file` 重跑清单里的算子，移出已恢复的。
4. 把[测试版本](#测试版本)和全量参考更新为本次筛查所用的 commit。

`ops/inventory_*.csv` 每个测试函数一行：`op`、`op_file`、`marked_skip`（无条件 `pytest.mark.skip`）、`collected`（pytest collect 核实结果）和 `op_names`。`marked_skip=yes` 或 `collected=no`（如 FlagGems-vllm 中只能在昇腾上导入的算子）的不进 `--all`。

## 常见问题

- **同一个算子在另一次 A/B 里 gap 不同。**

  先看两次用的是不是同一份记录。每份记录各自调优出 config，编译器改动的效果可能强烈依赖 config（有的算子差几倍）。gap 是某份记录的 config 下的效果，只在同一份记录上比较 A/B。

- **`ab_compare.py` 留下了 undecided 的行。**

  差距超过阈值，但两侧样本重叠太多，达不到 p < 0.01。用 `ab_run.py --until-decided` 并调高 `--max-rounds`；一直判不了的是小而不稳定的效果。

- **latency 和别人测的差很多。**

  先确认 GPU 没有被别的任务占用（`nvidia-smi`），再确认对方是否用了 CUDA graph 计时：M=1 这类小 shape 下 launch 开销占主导，普通 `do_bench` 可能高出 2 倍以上；大 shape 基本一致。

- **shape 相同、什么都没改，两次的 speedup 却不同。**

  两次 record 就会这样：每次都重新 sweep，计时噪声让它们选出不同 config。用其中一次 replay 另一次。

- **某个算子第一次跑特别慢。**

  要么是 `USE_FLAGTUNE=1`，首次运行要搜索整个 FlagTune 空间（marlin 超过十分钟，期间 `run.log` 停在测试名不动属正常）；要么是 libtuner 算子首次 sweep（mm 约 50 分钟）。winner 会存进 `~/.flaggems/config_cache/`，之后同样的 shape 几秒即可。默认 `USE_FLAGTUNE=0` 时单个 shape 通常十秒左右；shape 多时用 `PARALLEL_WARMUP_GPUS`。

- **要设 `--warmup` 或 `--iter` 吗？**

  不用。warmup 预算用于 capture 前的显式预热，100 ms 的 iter 预算会按 kernel 耗时换算成次数。

- **`ttgir/` 下某个 shape 目录少了文件。**

  看 `index.tsv` 的 `launches` 列：在该 shape 下没有在 sweep 之外运行过的变体不会写出。`run.log` 里的回退标记可以解释异常情况。

- **批量里某个算子是 SKIP。**

  它的 `run.log` 末尾的 `SKIPPED [...]` 行写明了原因，通常是上游 skipif（缺参考实现、架构不支持）。

## 开发

```
run_pytest.sh       单算子入口
run_batch.py        批量入口
ab_run.py           重复测量 A/B（多轮交替 replay）
ab_compare.py       A/B 的逐行判定，或两个 run 的单次对比
ab_stats.py         判定规则与 Mann-Whitney 检验
ir_gate.py          编译差异门控
gen_inventory.py    算子清单与 shape 集刷新
batch_summary.py    run.log 解析与 summary.csv
_*_plugin.py        pytest 插件；_fg_benchmark.py、_term_style.py、_timing_scope.py 为共用模块
ops/                算子清单与批量排除清单
shape/              shape 集
tests/              不依赖 GPU 部分的单元测试
docs/               设计说明
```

```bash
pip install -r requirements.txt ruff
ruff check .
shellcheck run_pytest.sh
python -m pytest
```

CI（`.github/workflows/ci.yml`）在 Python 3.10 与 3.12 上跑这些检查。实际测量需要 GPU 和上游仓库，不在 CI 里；改插件后，跑一个覆盖两个仓库、特殊输入路径和 do_bench 回退的小批次，并检查每个 run 的 `rows.tsv` 与结果行一一对应：

```bash
python run_batch.py --ops abs mm softmax conv1d_padding perf_unique2 vllm:fused_marlin_moe_w4a16_mxfp4 \
  vllm:flash_mla_sparse_fwd --dtypes bfloat16
```

## 许可证

MIT — 见 [LICENSE](LICENSE)。
