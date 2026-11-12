import batch_summary

PRETTY_LOG = """\
Operator: softmax  (dtype=torch.bfloat16, mode=kernel, level=core)
Status     Torch Latency (ms)   Gems Latency (ms)   Gems Speedup  Size Detail
SUCCESS              0.010000            0.008000          1.250  [torch.Size([1024, 1024])]
FAILED                    N/A                 N/A            N/A  [torch.Size([64, 512, 512])]
"""

UPSTREAM_LOG = """\
Operator: mm  Performance Test (dtype=torch.float16, mode=kernel,level=core)
SUCCESS   0.003175   0.003074   1.033   36.843   [torch.Size([384, 384])]
"""


def test_parse_pretty_and_upstream_tables():
    rows = batch_summary.parse_run_log(PRETTY_LOG + UPSTREAM_LOG)
    assert [r["fg_op_name"] for r in rows] == ["softmax", "softmax", "mm"]
    assert rows[0]["dtype"] == "torch.bfloat16"
    assert (rows[0]["torch_lat"], rows[0]["gems_lat"], rows[0]["speedup"]) == (
        "0.010000", "0.008000", "1.250")
    assert rows[0]["size_detail"] == "[torch.Size([1024, 1024])]"
    assert rows[1]["row_status"] == "FAILED"
    assert rows[2]["extra_metrics"] == "36.843"


def test_classify():
    rows = batch_summary.parse_run_log(PRETTY_LOG)
    assert batch_summary.classify(0, rows[:1], "") == "PASS"
    assert batch_summary.classify(0, rows, "") == "FAIL"
    assert batch_summary.classify(1, rows[:1], "") == "FAIL"
    assert batch_summary.classify(0, [], "== 1 skipped in 0.2s ==") == "SKIP"
    assert batch_summary.classify(0, [], "== 1 passed in 0.2s ==") == "FAIL"


def test_skipped_count_reads_the_summary_line_wherever_it_is():
    summary = "======================== 1 xfailed, 1 warning in 34.63s ========================\n"
    assert batch_summary.pytest_skipped_count(summary) == 1
    # the IR dump prints after the summary, one line per shape: 200 shapes push it far
    # beyond any fixed window over the end of the log
    dump = "".join(f">>> [Dump]   {i}x256x7168x2048x8: 3\n" for i in range(200))
    assert len(dump) > 4000
    assert batch_summary.pytest_skipped_count(summary + dump) == 1
    assert batch_summary.classify(0, [], summary + dump) == "SKIP"
    # only the last summary counts (a retried run appends nothing, but logs can be joined)
    two = "== 3 skipped in 0.1s ==\n" + "== 1 passed, 2 skipped, 1 xfailed in 2.00s ==\n"
    assert batch_summary.pytest_skipped_count(two) == 3
    # "skipped" elsewhere in a long log is not a count of the session
    stray = "7 skipped shapes\n" + "x" * 5000 + "\n== 1 passed in 1.0s ==\n"
    assert batch_summary.pytest_skipped_count(stray) == 0
    assert batch_summary.pytest_skipped_count("") == 0


def test_marker_notes():
    text = ("BENCHMARK_DIRECT_NO_CUDAGRAPH op=x phase=capture\n" * 2
            + "AUTOTUNE_REPLAY_FALLBACK reason=key_missing\n")
    assert batch_summary.marker_notes(text) == "no_cudagraph_rows=2;replay_fallback=1"
    assert batch_summary.marker_notes("") == ""


def test_marker_notes_flag_what_makes_a_run_misleading():
    # a run that passed, but on other shapes than asked for, or without its IR
    text = ("[shape-override-plugin] SHAPE_KEYS_UNUSED mm has no entry -- ...\n"
            "[ir-dump-plugin] IR_DUMP_FAILED /runs/x: ...\n"
            "CUDAGRAPH_OOM_RETRY op=mm calls_per_graph=1\n")
    assert batch_summary.marker_notes(text) == "cudagraph_oom_retry=1;shape_keys_unused;ir_dump_failed"
    assert batch_summary.marker_notes("[ir-dump-plugin] ROWS_TSV_FAILED /runs/x") == "ir_dump_failed"
