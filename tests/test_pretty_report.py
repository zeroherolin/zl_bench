from types import SimpleNamespace

import _pretty_report_plugin as pretty
import batch_summary


def _metrics(latency, **kw):
    base = {"latency_base": 0.02, "latency": latency, "speedup": 0.02 / latency, "tflops": None, "gbps": None,
                "gbps_base": None, "error_msg": None, "shape_detail": ["torch.Size([8, 8])"]}
    base.update(kw)
    return SimpleNamespace(**base)


def _table(metrics, **kw):
    result = SimpleNamespace(op_name="softmax", dtype="torch.bfloat16", mode="kernel", level="core",
                             result=metrics, **kw)
    return pretty._pretty_str(result)


def test_table_parses_back_to_the_measured_numbers():
    text = _table([_metrics(0.01), _metrics(0.005, error_msg="boom")])
    rows = batch_summary.parse_run_log(text)
    assert [(r["fg_op_name"], r["dtype"], r["row_status"]) for r in rows] == [
        ("softmax", "torch.bfloat16", "SUCCESS"), ("softmax", "torch.bfloat16", "FAILED")]
    first = rows[0]
    assert (first["torch_lat"], first["gems_lat"], first["speedup"]) == ("0.020000", "0.010000", "2.000")
    assert first["size_detail"] == "['torch.Size([8, 8])']"


def test_optional_columns_follow_upstream():
    plain = _table([_metrics(0.01)])
    assert "TFLOPS" not in plain and "GBPS" not in plain and "vs Base" not in plain
    text = _table([_metrics(0.01, tflops=36.8, gbps=1.5, gbps_base=1.2, compared_speedup=1.07)])
    header = next(line for line in text.splitlines() if line.startswith("Status"))
    assert header.split("  Size Detail")[0].split() == [
        "Status", "Torch", "Latency", "(ms)", "Gems", "Latency", "(ms)", "Gems", "Speedup", "vs", "Base",
        "TFLOPS", "Torch", "GBPS", "Gems", "GBPS"]
    (row,) = batch_summary.parse_run_log(text)
    # the first three numbers are still torch / gems / speedup; the rest are extras
    assert (row["torch_lat"], row["gems_lat"], row["speedup"]) == ("0.020000", "0.010000", "2.000")
    assert row["extra_metrics"] == "1.070 36.800 1.200 1.500"


def test_inputs_shared_by_all_rows_are_folded_into_a_legend():
    shapes = [["torch.Size([1, 4096])", "torch.Size([8, 28672, 4096])", "torch.Size([8, 4096, 14336])"],
              ["torch.Size([16, 4096])", "torch.Size([8, 28672, 4096])", "torch.Size([8, 4096, 14336])"]]
    text = _table([_metrics(0.01, shape_detail=shapes[0]), _metrics(0.02, shape_detail=shapes[1])])
    assert f"2 inputs identical across all rows: {shapes[0][1]}, {shapes[0][2]}" in text
    rows = batch_summary.parse_run_log(text)
    assert [r["size_detail"] for r in rows] == ["torch.Size([1, 4096])", "torch.Size([16, 4096])"]


def test_empty_result_and_skipped_baseline():
    assert "(no results)" in _table([])
    text = _table([_metrics(0.01, latency_base=None, speedup=None)], native_baseline_skip_reason="no vLLM")
    assert "Native baseline: N/A (no vLLM)" in text
    (row,) = batch_summary.parse_run_log(text)
    assert (row["torch_lat"], row["gems_lat"], row["speedup"]) == ("N/A", "0.010000", "N/A")
