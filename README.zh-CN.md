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
- **统计化 A/B。** 两侧在同一张卡上交替运行多轮；每个结果行由秩检验和逐轮配对的符号检验给出判定，判不了的自动补轮；二进制没变的行作为对照组，实测误报率。
- **批量筛查。** 在多卡上跑两个仓库的 1800 多个 benchmark，支持超时、重试、断点续跑和逐行汇总。
- **shape 由 yaml 说了算。** 一份 yaml 决定所有算子的 shape，包括在代码里写死 shape 的 benchmark。

实现原理见 [docs/design.zh-CN.md](docs/design.zh-CN.md)。

## 测试版本

zl_bench 会 monkeypatch 上游内部实现，可用性与测试时的上游版本绑定。本 README 的所有内容，包括[全量参考](#批量测试)和 `ops/dismiss.txt`，都基于以下 commit：

| 组件 | Commit | 日期 | 来源 |
|---|---|---|---|
| FlagGems | [`a4e2556d5`](https://github.com/flagos-ai/FlagGems/commit/a4e2556d5) | 2026-11-04 | `flagos-ai/FlagGems` master |
| FlagGems-vllm | [`a579b7c`](https://github.com/flagos-ai/FlagGems-vllm/commit/a579b7c) | 2026-11-10 | `flagos-ai/FlagGems-vllm` main |
| FlagTree | [`6d7597f66`](https://github.com/flagos-ai/FlagTree/commit/6d7597f66) | 2026-11-10 | `flagos-ai/FlagTree` main |

环境：NVIDIA H20（Hopper，驱动 570.158.01）、CUDA 13.0、PyTorch 2.11.0、Python 3.12、vLLM 0.20.2。换用其他上游版本时，先重建算子清单、刷新 shape 集（见[跟进上游更新](#跟进上游更新)）；每个 `run.log` 都会记录实际使用的 Triton 路径，以及编译器和算子仓库的 git 版本。

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

两个仓库有同名但实现不同的算子，测哪个由 `FLAGGEMS_DIR` 决定。测量始终是单进程单卡，这是计时可比的前提；用 `CUDA_VISIBLE_DEVICES` 选卡。`run.log` 开头记录 triton 路径、编译器与算子仓库的 git 版本（有未提交改动时带 `-dirty`），以及 `FLAGTREE_*`/`TRITON_*`/`MLIR_*` 环境变量，A/B 两侧对账时先核对这几行。

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

**留意 `run.log` 里的标记。** `AUTOTUNE_REPLAY_FALLBACK` 表示记录里缺这个 config，或它在新编译器下编译失败，该测量点改为现场调优，不再用 A 侧的 config（`ab_compare.py` 会给这类算子标 `[replay_fallback]`）。reason 说明缺的是什么：整份记录（`record_missing`）、某个 kernel 的全部 config（`kernel_missing`，每个 kernel 只打印一次）或某一个 key（`key_missing`）；带 `_no_evict` 后缀时重试仍读到了那个坏 config，该点按未验证处理。`BENCHMARK_DIRECT_NO_CUDAGRAPH` 表示该调用无法 graph capture（`phase=host_sync`：它与 host 同步），该点用普通 `do_bench` 计时；两侧回退一致，但这些数字噪声更大，亚毫秒级 kernel 尤其明显（[详情](docs/design.zh-CN.md#计时)）。

### 重复测量 A/B

`ab_run.py` 在两套编译器配置上多轮 replay 同一份记录（record 模式下得到的 run 或批次目录，通常用 A 侧的编译器记录）；`ab_compare.py` 把多轮数据变成每个结果行的判定。

```bash
# A 侧与 B 侧：各给一个 PYTHONPATH，或同一构建加编译开关（--a-env / --b-env）
python ab_run.py --record runs/<record> --ops softmax mm \
  --a-pythonpath /path/FlagTree-base/python --b-pythonpath /path/FlagTree-dev/python \
  --rounds 6 --until-decided --out runs/ab_dev
python ab_compare.py runs/ab_dev -v --csv runs/ab_dev/rows.csv
```

- 每个（算子, 轮次）在同一张卡上先后跑 A 和 B，奇数轮先 A、偶数轮先 B，抵消卡间偏差和缓慢漂移。所有任务共用一个队列、分到所有卡上，耗时长的先跑；每次运行都是新进程，各轮是独立样本。每次运行的处理与[批量测试](#批量测试)相同（超时、重试、空闲检查）。`<out>/r<k>_a` 与 `r<k>_b` 的结构与批次目录相同。用同一个 `--out`、相同的两侧设置、记录、`--dtypes` 和 `--shape-file` 重跑即断点续跑：两侧都通过的轮次保留；有一侧失败或缺失的轮次两侧一起重跑，保证同一轮的两侧仍是同一张卡上的一对。Ctrl-C（或 SIGTERM、关闭终端）会停掉正在跑的运行并以 130 退出，目录仍可续跑；被打断的那一轮在重跑之前不计入任何统计。
- 每行的判定（[详情](docs/design.zh-CN.md#统计判定)），前提是至少 `--min-rounds`（6）轮：中位数之差超过 `--threshold`（0.5%），且两种检验之一显示出差异时为 `faster` / `slower`——双侧 Mann-Whitney 检验 p < `--alpha`（0.01），或者配对：每一轮方向都一致，且各轮的符号检验低于 `--alpha-paired`（0.05；6 轮同向时 p = 0.031）。差距在阈值内，或 p ≥ `--clear`（0.1）且配对也没有显示差异时为 `same`；其余为 `undecided`。两种检验面对的噪声不同：秩检验把各轮混在一起，卡间几个百分点的差异都算作噪声，每一轮都存在的 2% 效应也可能只有 p = 0.2；同一轮的两侧在同一张卡上，配对把这部分抵消了。CSV 中有 `p`、`p_paired`，以及 faster/slower 行由哪种检验显示（`basis` 列：`rank` 或 `pairing`）。`--until-decided` 给有 undecided 行的算子每次补 `--step` 轮，直到 `--max-rounds`（14）。
- `gap = median(A) / median(B) − 1`，B 更快时为正。它是在该记录的 config 下编译器改动的效果，而不是两侧各自最优之间的差（[原因](docs/design.zh-CN.md#两侧各自最优)）。
- 加 `--gate <ir_gate 输出目录>`（见下一节）后，kernel 在两侧编译成同一二进制的行作为对照组：它们不可能有差别，`ab_compare.py` 会报告其中被判为 faster 或 slower 的数量，即实测误报率。门控放行的行只有在两侧每一轮实际启动的二进制也相同时才是对照组（每次运行都会记录）；不同则该行是 **diverged**，算作结果；某次运行无法确定，或日志里有 `AUTOTUNE_REPLAY_FALLBACK` 且没有记录二进制时，该行是 unknown：门控比较的是记录下的 config，而不是这样一次运行实际启动的东西。
- **证据弱的结果用全新的轮次复测。** 差距小或噪声大的行，是用同一批样本既挑出来又判定的，会被高估。`ab_compare.py --recheck-list` 写出含这类行（|gap| < 2%、|gap| 小于样本变异系数的 2 倍、用 do_bench 计时、某一轮有一侧在测完该行之后失败，或只有配对显示出差异）的算子；`ab_run.py --add-rounds 6` 给它们补新的轮次，编号接在已有轮次之后；`ab_compare.py --confirm <起>-<止>`（`ab_run.py` 会打印这个范围）先用其余轮次判定每一行，faster/slower 的行只有在新轮次单独也能复现时才保留，单侧、`--confirm-alpha`（0.05）：秩检验，或新轮次全部同向。没能复现的行改为 `undecided`。新轮次少于 4 轮时什么结果都到不了 0.05，4 轮全部同向但两侧样本有重叠时也到不了（配对需要 5 轮）；这样的行报告为 `too_few_rounds` 并保持原判定，所以至少补 5 轮。补的轮次被中断或有失败时，加 `--from-round <起>` 重跑（`ab_run.py` 会打印这条命令）：不加的话，重跑会从当前最大轮次之后另起一段。

  ```bash
  python ab_compare.py runs/ab_dev --gate runs/gate_dev --recheck-list runs/ab_dev/recheck.txt
  python ab_run.py <相同的两侧设置、记录、--dtypes 与 --shape-file> --out runs/ab_dev --ops-file runs/ab_dev/recheck.txt --add-rounds 6
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
- 门控重编译测量实际用到的变体，比较 cubin 和 launch 元数据（shared memory、warp 数、scratch 内存、启动方式）。当且仅当两者都相同时变体算未变：测到的 kernel 是同一个二进制，无需再测。TTGIR 两个方向都说明不了问题：后续阶段可以在 TTGIR 不变的情况下改变二进制，重排过的 TTGIR 也可能得到同一个二进制。`variants.tsv` 逐个变体给出最先出现差异的阶段（cubin 仍然相同时加括号），TTGIR 是变了、只是重排还是没变，以及数量有变化的 TTGIR op。`--stage ttgir` 只比到 TTGIR，用于快速初看，此时任何 TTGIR 差异都算变化；不给 `--base` 时以记录时落盘的 ttgir 为基线，同样只能比到 TTGIR。
- 门控的 `rows.tsv` 逐个结果行标记是否变化：该行跑过的 kernel 有一个变了就算变（依据运行自己的 `rows.tsv`），所以只在某个 shape 的 kernel 上有变化的算子，其他行不会被算成结果，而是成为对照组。没有 kernel 的行只有在该运行用到的每个变体都能在某一行里找到时才放行，否则为 unknown。没有行跟踪的旧记录按算子判定。`changed_ops.txt` 与 `all_ops.txt` 里的算子写法与 `--ops` 接受的一致；单算子记录需要当前 `run_pytest.sh` 写进日志的 `spec=` 行。
- 门控放行的是记录下的变体、按记录下的编译选项编译的结果。A/B 的某一侧实际启动的是不是这些，要看运行本身：每次运行都记下各结果行启动的二进制（`rows.tsv` 的 `binaries` 列：cubin 的代码与 launch 元数据，不含行号信息），`ab_compare.py --gate` 把门控放行、但两侧运行了不同二进制的行报告为 **diverged**，算作结果而不是对照组。这种情况出现在算子库只在编译器支持时才设置某个编译选项（FlagGems-vllm 对 FlagTree 的 `pipeline_*` 选项就是这样，既按 kernel 设置，也作为调优空间多出来的一维），以及 replay 回退之后。某个构建有记录或另一个构建没有的编译选项时，`ir_gate.py` 会给出警告。这时把所有记录过的算子在两侧各 replay 一轮（`ab_run.py --ops-file <gate>/all_ops.txt --rounds 1 --gate <gate> ...`），写出有 diverged 行的算子（`ab_compare.py <ab> --gate <gate> --diverged-list <文件>`），再对两份清单一起做第 3 步：`cat <gate>/changed_ops.txt <文件> > ops.txt`，然后 `--ops-file ops.txt`（两份清单里都有的算子只跑一次）。
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
- `--op-timeout`（默认 1800 秒）杀掉卡住的算子，记为 `TIMEOUT`，不重试。测试失败和被信号杀掉最多重试两次，连续两次退出码相同就停止。重试复用该算子的目录：上一次的日志保留为 `run.attempt<n>.log`，IR 被替换。
- 指定 `--dtypes` 时，算子不支持该 dtype 就去掉限制重跑（`dtype_fallback`）；其他失败且没有任何 SUCCESS 行时，再不加限制跑最后一次（`dtype_rescue`）。
- 运行时的 skipif 与 xfail 记为 SKIP。只有 PASS 和 SKIP 会被标记为完成、供 `--resume` 复用。测试通过但没有打印 FlagGems 结果表（正确性测试、自定义输出格式的 benchmark）记为 FAIL：工具没有可读的结果。
- `ops_status.csv` 给批次目录里的每个算子一行（状态、所在卡、耗时、行数、备注），边跑边更新；同一目录续跑或再次运行时，这次没有重跑的算子保留原来的行。`summary.csv` 每个结果行一行，`batch_summary.py` 可为旧批次重新生成。解读结果时先看备注列：它统计 `no_cudagraph_rows`、`replay_fallback` 与 `cudagraph_oom_retry`，并标出 `shape_keys_unused`（该算子测的不是 shape 文件里的 shape）和 `ir_dump_failed`；带这些标记的 PASS 并不是看上去的那次测量。
- Ctrl-C（或 SIGTERM、关闭终端）会停掉正在跑的运行并以 130 退出；之后用 `--batch-dir <目录> --resume` 接着跑。用 `nohup` 启动时，批次照常不受终端关闭影响。
- 算子开始前，它的卡必须空闲：`nvidia-smi` 里没有进程，且已用显存不到 1 GiB（其他容器的进程不会列出）。同卡上的其他进程会同时干扰计时和调优，所以批次会等待（`--busy-gpu wait`，默认；`warn` 照常测量并在备注里记 `gpu_busy`）。卡号与 `nvidia-smi` 一致（`CUDA_DEVICE_ORDER=PCI_BUS_ID`）。
- 批量数字只用于筛查：多张卡同时跑会通过功耗和散热互相影响。可疑算子用[重复测量 A/B](#重复测量-ab) 确认。

**全量参考**（[测试版本](#测试版本)，7×H20，`--dtypes bfloat16`，单 shape 集）：`--all` 选出 1876 个算子，耗时 5 小时 55 分，合计 38 GPU 小时。PASS 1822、SKIP 54，10241 个结果行全部 SUCCESS。SKIP 都来自上游：skipif（缺 TransformerEngine 或 vLLM 参考实现、只支持其他厂商或架构、需要多 rank 或多卡启动）或 xfail。每个算子中位 46 秒（不需要重试的 43 秒，重跑过的 79 秒，多数是去掉 dtype 过滤后重跑），90 分位 112 秒，最长 24 分钟（`convolution_double_backward`）。因上游或环境问题失败、或不打印 FlagGems 结果表的算子列在 `ops/dismiss.txt` 里。多 shape 集上每个算子中位约一分钟，record 模式下最长要两小时（`chunk_gdn2`），所以要设 `--op-timeout 7200`。

### 多卡加速 sweep

shape 多时，墙上时间主要花在 autotune sweep 而不是测量上。`PARALLEL_WARMUP_GPUS=N` 让 sweep 在 N 张卡上并行，之后照常在一张卡上串行测量：

```bash
PARALLEL_WARMUP_GPUS=8 SHAPE_FILE=shape/DeepSeek-V4-Flash-p32768d1024.yaml bash run_pytest.sh
```

产物、replay 和结果表都不变。并行的分片会互相干扰，可能改变哪个 config 胜出，这时日志会警告：

```
WARNING 1 config key(s) got different winners across shards — parallel interference reached the sweep
```

没有冲突时 config 可信；冲突多时，下结论前去掉 `PARALLEL_WARMUP_GPUS` 重跑。设了 `REPLAY_FROM`、可见卡少于两张，或 shape 集里没有哪个条目有两个以上 shape 可分（单 shape 集）时，插件不起作用（[详情](docs/design.zh-CN.md#多卡并行-sweep)）。

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
- 键是 benchmark 的 op_name，不是测试名，而且两个仓库不总是一致：`test_fused_marlin_moe_w8a16_int8` 在 FlagGems 里是 `fused_marlin_moe_w8a16_int8`，在 FlagGems-vllm 里是 `fused_marlin_moe_w8a16`。`ops/inventory_*.csv` 的 `op_names` 列给出了大多数测试的键；该列为空说明 op_name 在测试里不是字面量，这时看 benchmark 源码，或看一次运行日志里的 `shapes of ...` 行。
- 同一个名字在两个仓库里含义不同时，把键限定到仓库：`vllm:<名字>` 只对 FlagGems-vllm 生效，并在那里优先于普通键；`main:<名字>` 对 FlagGems 同理。`fused_recurrent_gated_delta_rule` 在 FlagGems 里取序列长度，在 FlagGems-vllm 里取 case 名，所以共用的 shape 集里只有 `vllm:fused_recurrent_gated_delta_rule`，没有普通键。上游不认识限定键：`run_pytest.sh` 会把 shape 集按被测仓库解析（`shape_set.py`）后写成本次运行的 `shapes.yaml`，replay 沿用的也是它。
- 没有条目的算子跑上游的默认 shape，或者适用于它的类名键的 shape。对写死 shape 的算子来说就是全部 shape（marlin 是 16 个大 MoE shape，从几秒变成几分钟），所以这类算子新增或改名后要在两个集里都补上条目。shape 形式与通用 `Benchmark` 键不同的算子同样需要：`hc_combine` 读的是五元 shape，拿到该键给的二维 shape 就失败。
- 每次运行都会在日志里写明 shape 的来源（`shapes of <op_name>: 1 of the 1 of key '<键>'`）；当这个来源不可能是 shape 文件的本意时打印 `SHAPE_KEYS_UNUSED`：没有任何键适用于该算子、键写在了测试名而不是 op_name 上，或算子有条目但实际测的是它自己的 shape。回退到类名键不报告，所以在带有 `Benchmark` 键的 `all_ops_*` 集里，写死 shape 又没有条目的算子只体现为日志行 `shapes of <op_name>: its own N (class key 'Benchmark' not used)`。
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
├── rows.tsv             # 每个结果行：shape、它运行过的 kernel（cache hash）及其二进制指纹
├── ttgir/               # 实际用到的变体的 IR，按 shape 分组
│   ├── 1024x1024/softmax_kernel_inner/w4s3__bf16.ttgir
│   ├── index.tsv        # 所有编译变体：shape、文件、启动次数、constexpr、cache hash
│   └── naming.md        # 文件名缩写图例
└── ttir/<cache hash>/   # 实际用到的变体的 ttir 与编译参数，ir_gate.py 的输入
```

- 只写测量实际用到的变体；sweep 落选者只出现在 `index.tsv` 里。
- 文件名列出同一 (shape, kernel) 组内取值不同的 constexpr，并缩写（`BLOCK_SIZE_M` → `BSM`），后接 `w{warps}s{stages}`；仍重名时依次用 dtype、参数特化或 hash 区分（[详情](docs/design.zh-CN.md#ir-落盘)）。
- 在输入构造和测量之外启动的 kernel（如模块级探测）归到 `shape_unknown/`。
- `rows.tsv` 的顺序与 `run.log` 里的结果表一致：表的第 `i` 行对应它的测量启动过的 kernel（scope 为 `timed`），`ir_gate.py` 据此把变化归到行上。没有任何 kernel 是在计时函数里启动的行（不经过 `triton.testing.do_bench` 计时的 benchmark，或没加载 `_cudagraph_plugin` 的运行）得到为它的输入启动过的全部 kernel（scope 为 `row`）。`shape` 列是 benchmark 的输入迭代器当时走到的 shape；迭代器不遍历 shape 时是 `input<i>`，表示它的第 i 个输入。
- `rows.tsv` 的 `binaries` 列标识这些 kernel 作为二进制是什么：每个 kernel 一个指纹，覆盖 kernel 名、cubin 的代码（不含调试段）和 launch 元数据；没有启动 kernel 的行是 `-`，运行无法确定时是 `?`。cache hash 在任意两个编译器构建之间都不同，而两个构建生成同一个二进制时指纹相同，`ab_compare.py` 比较 A/B 两侧的正是它。
- IR 在进程退出时落盘，CUDA 报错导致会话结束时也会写；进程本身被杀（段错误、abort、超时被杀）时不会。

## 跟进上游更新

1. 更新两个仓库（fork 用 `git fetch upstream && git merge --ff-only upstream/<分支>`）。
2. 重建算子清单、刷新 shape 集：

   ```bash
   python gen_inventory.py --verify-collect
   python gen_inventory.py --refresh-shapes shape/all_ops_single.yaml --one-shape
   python gen_inventory.py --refresh-shapes shape/all_ops_multi.yaml
   ```

   之后检查所有 shape 集，尤其是自定义的（`all_ops_*` 以外的），因为 `--refresh-shapes` 从不重写它们；自己的 shape 集各加一个 `--check-shapes <文件>`。发现问题时退出码为 1，不写任何文件：

   ```bash
   python gen_inventory.py $(printf -- '--check-shapes %s ' shape/*.yaml)
   ```

   匹配不到当前 op_name 的键**在运行时被静默忽略**：上游回退到 benchmark 自带的硬编码 shape，运行看起来完全正常，但测的不是指定的 shape。该检查会报出这类键、挂在测试名上而其 benchmark 读的是另一个 op_name 的键，并以 note 形式列出：与集合中某个键共用同一个写死 shape 的 Benchmark 类、自己却没有条目的 op_name（在自定义集里通常是漏了某个仓库对该 benchmark 的叫法）。运行时，在没有类名键的集里同样的漂移表现为 `SHAPE_KEYS_UNUSED`；在 `all_ops_*` 集里算子会落到 `Benchmark` 键上，只有日志行 `shapes of ...: its own N` 能看出来，所以要跑这个检查。IR 落盘丢失表现为 `IR_DUMP_FAILED` / `ROWS_TSV_FAILED`。批量运行会把这些标记带进 `ops_status.csv` 的备注列。

   `--refresh-shapes` 只保留上游仍存在的 op_name 键，并同步 core_shapes 的键。改名算子的键会移到 `shape/<name>.unmatched.yaml`：找到新名（`grep -rn 'op_name=' $FLAGGEMS_DIR/benchmark/test_<op>.py`）后挪回，再删掉这个文件。
3. 跑一次全量筛查。失败的算子用上游自己的 pytest 复查（`cd $FLAGGEMS_DIR && python -m pytest benchmark/test_<op_file>.py::test_<op> --level core --mode kernel`，不带插件和 shape 文件）：仍然失败的是上游或环境问题，写进 `ops/dismiss.txt` 并注明原因与日期；不再失败的是 shape 集或插件的问题，要在这里修。也定期用 `--ops-file ops/dismiss.txt` 重跑清单里的算子，移出已恢复的。
4. 把[测试版本](#测试版本)和全量参考更新为本次筛查所用的 commit。

`ops/inventory_*.csv` 每个测试函数一行：`repo`、`op`、`op_file`、`file`、`marked_skip`（无条件 `pytest.mark.skip`）、`collected`（pytest collect 核实结果）和 `op_names`（在测试里找到的 op_name 字面量，名字是计算出来的则为空）。`marked_skip=yes` 或 `collected=no`（如 FlagGems-vllm 中只能在昇腾上导入的算子）的不进 `--all`。

## 常见问题

- **同一个算子在另一次 A/B 里 gap 不同。**

  先看两次用的是不是同一份记录。每份记录各自调优出 config，编译器改动的效果可能强烈依赖 config（有的算子差几倍）。gap 是某份记录的 config 下的效果，只在同一份记录上比较 A/B。

- **`ab_compare.py` 留下了 undecided 的行。**

  要么轮次不足 `--min-rounds`；要么差距超过阈值，但两个检验都还显示不出来：秩检验的 p 在 0.01 到 0.1 之间，各轮也不是全部同向。`--confirm` 没能复现的行同样是 undecided。用 `ab_run.py --until-decided` 并调高 `--max-rounds`；一直判不了的是小而不稳定的效果。

- **latency 和别人测的差很多。**

  先确认 GPU 没有被别的任务占用（`nvidia-smi`），再确认对方是否用了 CUDA graph 计时：M=1 这类小 shape 下 launch 开销占主导，普通 `do_bench` 可能高出 2 倍以上；大 shape 基本一致。

- **shape 相同、什么都没改，两次的 speedup 却不同。**

  两次 record 就会这样：每次都重新 sweep，计时噪声让它们选出不同 config。用其中一次 replay 另一次。

- **某个算子第一次跑特别慢。**

  要么是 `USE_FLAGTUNE=1`，首次运行要搜索整个 FlagTune 空间（marlin 超过十分钟，期间 `run.log` 停在测试名不动属正常）；要么是 libtuner 算子首次 sweep（mm 约 50 分钟）。winner 会存进 libtuner 缓存（FlagGems 是 `~/.flaggems/config_cache/`，FlagGems-vllm 是 `~/.flagdnn/config_cache/`；`FLAGGEMS_CACHE_DIR` 可同时改两者的位置），之后同样的 shape 几秒即可。默认 `USE_FLAGTUNE=0` 时单个 shape 的测量通常十秒左右，另加每次运行的固定开销（见下一条）；shape 多时用 `PARALLEL_WARMUP_GPUS`。第三种原因是算子写死了 shape 而 shape 集里没有它的条目：它会把写死的 shape 全跑一遍（日志里有 `SHAPE_KEYS_UNUSED`，或 `shapes of ...: its own`）。

- **每个算子都要 40 秒左右，哪怕很简单。**

  这是每次在新进程里运行的固定开销，换来的是算子之间互不影响：崩溃、调优状态和 Triton 缓存都各归各。其中大部分是 import 算子库：在 H20 机器上 `flag_gems` 约 18 秒，`flaggems_vllm` 约 23 秒。批量运行另有约 3 秒的空闲检查。

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
shape_set.py        shape 集：限定到单个仓库的键，每次运行按仓库解析
batch_summary.py    run.log 解析与 summary.csv
_*_plugin.py        pytest 插件；_binary_id.py、_fg_benchmark.py、_shape_scope.py、_term_style.py、_timing_scope.py 为共用模块
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
