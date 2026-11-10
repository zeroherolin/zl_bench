import _ir_dump_plugin as dump


def test_abbrev_and_fmt_value():
    assert dump._abbrev("BLOCK_SIZE_M") == "BSM"
    assert dump._abbrev("BLOCK_M2") == "BM2"
    assert dump._abbrev("SPLIT") == "SPLIT"
    assert dump._fmt_value(True) == "T"
    assert dump._fmt_value(1) == "1"  # not "T": 1 == True in Python
    assert dump._fmt_value(None) == "-"
    assert dump._fmt_value("triton.language.bfloat16") == "bfloat16"


def _variant(h, warps=4, stages=3, signature=None, **constexprs):
    return {"hash": h, "warps": warps, "stages": stages, "constexprs": constexprs,
            "signature": signature or {}, "attrs": {}}


def test_name_group_uses_only_varying_constexprs():
    items = [_variant("AAAA1111", BLOCK_SIZE_M=64, GROUP=8),
             _variant("BBBB2222", BLOCK_SIZE_M=128, GROUP=8)]
    named, legend = dump._name_group(items)
    assert sorted(name for _, name in named) == ["BSM128_w4s3.ttgir", "BSM64_w4s3.ttgir"]
    assert legend == {"BLOCK_SIZE_M": "BSM"}


def test_name_group_disambiguates_by_dtype_then_hash():
    by_dtype = [_variant("AAAA1111", signature={"x": "*fp16"}, B=1),
                _variant("BBBB2222", signature={"x": "*bf16"}, B=1)]
    named, _ = dump._name_group(by_dtype)
    assert sorted(name for _, name in named) == ["w4s3__bf16.ttgir", "w4s3__fp16.ttgir"]

    by_hash = [_variant("AAAA1111", B=1), _variant("BBBB2222", B=1)]
    named, _ = dump._name_group(by_hash)
    assert sorted(name for _, name in named) == ["w4s3__AAAA1111.ttgir", "w4s3__BBBB2222.ttgir"]


def test_rows_tsv_prefers_the_timed_kernels(tmp_path, monkeypatch):
    monkeypatch.setattr(dump, "_tables",
                        [("softmax", "torch.float16", [(0, "4x4"), (1, "8x8"), (None, "x")])])
    monkeypatch.setattr(dump, "_row_timed", {0: {"H2", "H1"}})
    monkeypatch.setattr(dump, "_row_any", {0: {"H1", "H2", "H3"}, 1: {"H4"}})
    dump._write_rows(tmp_path / "rows.tsv")
    lines = (tmp_path / "rows.tsv").read_text().splitlines()
    assert lines[0] == "op_name\tdtype\trow\tshape\tkernels\tscope"
    assert lines[1:] == ["softmax\ttorch.float16\t0\t4x4\tH1,H2\ttimed",
                         "softmax\ttorch.float16\t1\t8x8\tH4\trow",
                         "softmax\ttorch.float16\t2\tx\t?\t-"]


def test_short_label_is_unchanged():
    assert dump._label_of([64, 256, 1, 1, 8]) == "64x256x1x1x8"
    assert dump._label_of("mm") == "mm"


def test_long_label_fits_a_directory_name_and_stays_distinct():
    # a varlen attention case carries its cu_seqlens: the plain join is far over the
    # 255-byte limit, and a name the filesystem rejects used to fail the whole dump
    a = dump._label_of(list(range(300)))
    b = dump._label_of(list(range(1, 301)))
    assert len(a) <= dump._MAX_LABEL and len(b) <= dump._MAX_LABEL
    assert a != b
    assert a.startswith("0x1x2x3x")   # still readable at the front


def test_dump_failure_is_reported_not_swallowed(tmp_path, capsys, monkeypatch):
    # the dump runs from an atexit hook, after pytest called the run a pass; an error
    # there must be visible, or the missing IR only surfaces in a later gate
    def boom(cache_dir, run_dir):
        raise OSError(36, "File name too long")

    monkeypatch.setattr(dump, "_dump_ir", boom)
    dump._dump_ir_guarded(str(tmp_path), str(tmp_path))
    err = capsys.readouterr().err
    assert "IR_DUMP_FAILED" in err
    assert "File name too long" in err


def test_rows_tsv_survives_a_failing_ir_dump(tmp_path, capsys, monkeypatch):
    # rows.tsv is what attributes result rows to kernels; the much larger IR copy
    # failing (a rejected path, a cache file that went away) must not take it along
    written = []

    def boom(cache_dir, run_dir):
        raise OSError(36, "File name too long")

    monkeypatch.setattr(dump, "_write_rows", written.append)
    monkeypatch.setattr(dump, "_dump_ir", boom)
    dump._dump_ir_guarded(str(tmp_path), str(tmp_path))
    assert written, "rows.tsv was not written when the IR dump failed"
    err = capsys.readouterr().err
    assert "IR_DUMP_FAILED" in err and "ROWS_TSV_FAILED" not in err
