"""pytest plugin: readable, colorized benchmark result table.

Replaces FlagGems' BenchmarkResult.__str__ (benchmark/consts.py), whose rows
print every input's torch.Size(...) on every line — for the MoE ops that is 11
tensors and ~400 chars per row, ~90% of it identical across rows. This plugin:

- folds inputs identical across all rows into one legend line above the table,
  so each row's Size Detail keeps only what varies between rows;
- shapes keep their original torch.Size([...]) spelling (no compaction);
- colors the table title and Status (SUCCESS green / FAILED red);
- keeps the literal "SUCCESS"/"FAILED" words and upstream column titles, so
  run.log greps and __str__-wrapping hooks (vllm column renames) still work.

Color only reaches the live terminal: run_pytest.sh strips ANSI from run.log,
and _term_style disables color for non-tty stdout unless forced.
"""

from __future__ import annotations

import sys

from _term_style import BOLD, CYAN, DIM, GREEN, RED, paint, tag


def _shape_rows(metrics_list) -> list:
    """One list of per-input cell strings per metrics row (upstream spelling)."""
    rows = []
    for m in metrics_list:
        sd = m.shape_detail
        if isinstance(sd, (list, tuple)):
            rows.append([str(e) for e in sd])
        else:
            rows.append([str(sd) if sd is not None else "N/A"])
    return rows


def _factor_static(rows):
    """Fold columns identical across all rows into a legend.

    Only kicks in when it actually helps: >=2 rows of equal arity >=3, with
    >=2 static positions and at least one varying position left. Returns
    (legend_cells_or_None, per_row_varying_strings_or_None).
    """
    if len(rows) >= 2:
        arities = {len(r) for r in rows}
        if len(arities) == 1:
            n = next(iter(arities))
            if n >= 3:
                static = {i for i in range(n) if len({r[i] for r in rows}) == 1}
                if 2 <= len(static) < n:
                    legend = [rows[0][i] for i in sorted(static)]
                    kept = [
                        ", ".join(r[i] for i in range(n) if i not in static)
                        for r in rows
                    ]
                    return legend, kept
    return None, None


def _num_cell(value, fmt: str, width: int) -> str:
    s = f"{value:{fmt}}" if value is not None else "N/A"
    return f"{s:>{width}}"


def _pretty_str(self) -> str:
    metrics = self.result or []
    title = (
        "\n"
        + paint(f"Operator: {self.op_name}", BOLD, CYAN)
        + paint(f"  (dtype={self.dtype}, mode={self.mode}, level={self.level})", DIM)
        + "\n"
    )
    # Set by main-repo benchmarks marked skip_native: Torch columns read N/A.
    skip_reason = getattr(self, "native_baseline_skip_reason", None)
    if skip_reason:
        title += paint(f"Native baseline: N/A ({skip_reason})", DIM) + "\n"
    if not metrics:
        return title + "(no results)\n"

    # Same optional-column conditions as upstream.
    with_tflops = bool(metrics[0].tflops)
    with_gbps = metrics[0].gbps is not None
    # Upstream's --base-data adds a "vs Base" column after the speedup. The parsers of
    # run.log (batch_summary.py) read the first three numbers of a row by position, so
    # the column keeps upstream's place.
    with_base = any(getattr(m, "compared_speedup", None) is not None for m in metrics)

    legend, varying = _factor_static(_shape_rows(metrics))
    legend_line = ""
    if legend:
        legend_line = (
            paint(
                f"{len(legend)} inputs identical across all rows: "
                + ", ".join(legend),
                DIM,
            )
            + "\n"
        )
        shape_cells = varying
    else:
        # No folding possible: print the full shape_detail, upstream style.
        shape_cells = [
            str(m.shape_detail) if m.shape_detail is not None else "N/A"
            for m in metrics
        ]

    cols = [
        ("Status", 8, "<"),
        ("Torch Latency (ms)", 19, ">"),
        ("Gems Latency (ms)", 18, ">"),
        ("Gems Speedup", 13, ">"),
    ]
    if with_base:
        cols.append(("vs Base", 9, ">"))
    if with_tflops:
        cols.append(("TFLOPS", 13, ">"))
    if with_gbps:
        cols.append(("Torch GBPS", 12, ">"))
        cols.append(("Gems GBPS", 12, ">"))
    head_plain = (
        "  ".join(f"{name:{align}{width}}" for name, width, align in cols)
        + "  Size Detail"
    )
    header = paint(head_plain, BOLD) + "\n" + paint("-" * len(head_plain), DIM) + "\n"

    lines = []
    for m, shape_cell in zip(metrics, shape_cells, strict=True):
        ok = m.error_msg is None
        cells = [paint(f"{'SUCCESS' if ok else 'FAILED':<8}", GREEN if ok else RED)]
        cells.append(_num_cell(m.latency_base, ".6f", 19))
        cells.append(_num_cell(m.latency, ".6f", 18))
        cells.append(_num_cell(m.speedup, ".3f", 13))
        if with_base:
            cells.append(_num_cell(getattr(m, "compared_speedup", None), ".3f", 9))
        if with_tflops:
            cells.append(_num_cell(m.tflops, ".3f", 13))
        if with_gbps:
            cells.append(_num_cell(m.gbps_base, ".3f", 12))
            cells.append(_num_cell(m.gbps, ".3f", 12))
        lines.append("  ".join(cells) + "  " + shape_cell)

    return title + legend_line + header + "\n".join(lines) + "\n"


def pytest_configure(config):
    try:
        from benchmark import consts as fg_consts
    except Exception as exc:
        print(
            f"{tag('[pretty-report-plugin]')} disabled "
            f"(cannot import benchmark.consts: {exc})",
            file=sys.stderr, flush=True,
        )
        return
    fg_consts.BenchmarkResult.__str__ = _pretty_str
    print(
        f"{tag('[pretty-report-plugin]')} BenchmarkResult table -> folded + color "
        "(SUCCESS/FAILED words and column titles unchanged)",
        file=sys.stderr, flush=True,
    )
