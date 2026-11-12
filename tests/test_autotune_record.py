import json
import sys
import types

import pytest

import _autotune_record_plugin as rec

KERNEL = "pkg.ops.mm::mm_kernel"


class Config:
    """triton.Config, as far as the plugin reads and builds it."""

    def __init__(self, kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None,
                 ir_override=None):
        self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages
        self.num_ctas, self.maxnreg = num_ctas, maxnreg
        self.pre_hook, self.ir_override = pre_hook, ir_override


class Tuner:
    """An autotuner: run() "tunes" by taking its last config for a key it has not seen."""
    arg_names = ["x", "BLOCK"]
    keys = ["BLOCK"]

    def __init__(self, configs):
        self.configs, self.cache = configs, {}
        self.fn = type("fn", (), {"__module__": "pkg.ops.mm"})()
        self.__name__ = "mm_kernel"

    def run(self, *args, **kwargs):
        key = (kwargs["BLOCK"],)
        if key not in self.cache:
            self.cache[key] = self.configs[-1]
        return self.cache[key]


class Recording(Tuner):
    run = rec._record_run(Tuner.run)


class Replaying(Tuner):
    run = rec._replay_run(Tuner.run)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Empty plugin state, and a stand-in for triton: the tests need its Config only, so
    they run where Triton is not installed (CI)."""
    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(Config=Config))
    monkeypatch.setattr(rec, "_record_map", {})
    monkeypatch.setattr(rec, "_replay_map", {})
    monkeypatch.setattr(rec, "_unrecorded", set())
    monkeypatch.setattr(rec, "_dump_done", False)
    monkeypatch.setenv("FLAGGEMS_PERF_CURRENT_OP", "mm")


SMALL, LARGE = Config({"BM": 64}, num_warps=4, num_stages=3), Config({"BM": 128}, num_warps=8, num_stages=2)


def test_record_then_replay_pins_the_config(tmp_path, capsys):
    Recording([SMALL, LARGE]).run(None, BLOCK=32)
    rec._dump_record(str(tmp_path))
    recorded = json.loads((tmp_path / "mm.json").read_text())
    assert recorded == {KERNEL: {'["32"]': {
        "kwargs": {"BM": 128}, "num_warps": 8, "num_stages": 2, "num_ctas": 1, "maxnreg": None}}}

    assert rec._load_replay_dir(str(tmp_path)) == 1
    tuner = Replaying([LARGE, SMALL])  # left alone, it would pick SMALL
    used = tuner.run(None, BLOCK=32)
    assert (used.kwargs, used.num_warps, used.num_stages) == ({"BM": 128}, 8, 2)
    assert "AUTOTUNE_REPLAY_FALLBACK" not in capsys.readouterr().out
    # a key the record does not hold is tuned on the spot, and the log says so
    assert tuner.run(None, BLOCK=64) is SMALL
    assert "AUTOTUNE_REPLAY_FALLBACK reason=key_missing" in capsys.readouterr().out


def test_a_warm_cache_does_not_keep_replay_from_injecting():
    # LibTuner's cache is a sqlite db that outlives the process: the key is already there
    rec._replay_map[KERNEL] = {'["32"]': {"kwargs": {"BM": 128}, "num_warps": 8, "num_stages": 2}}
    tuner = Replaying([LARGE, SMALL])
    tuner.cache[("32",)] = tuner.cache[(32,)] = SMALL
    assert tuner.run(None, BLOCK=32).kwargs == {"BM": 128}


def test_a_missing_record_is_reported(tmp_path, capsys):
    assert rec._load_replay_dir(str(tmp_path)) == 0
    assert "AUTOTUNE_REPLAY_FALLBACK reason=record_missing" in capsys.readouterr().out


def test_config_round_trip_keeps_an_ir_override():
    cfg = Config({"BM": 64}, num_warps=2, num_stages=1, maxnreg=96, ir_override="/k/mm.ttgir")
    entry = rec._serialize_config(cfg)
    assert entry["ir_override"] == "/k/mm.ttgir"
    back = rec._deserialize_config(json.loads(json.dumps(entry)))
    assert (back.kwargs, back.num_warps, back.num_stages, back.maxnreg, back.ir_override) == (
        {"BM": 64}, 2, 1, 96, "/k/mm.ttgir")
    # an entry without the field loads as a config without an override
    plain = rec._serialize_config(Config({"BM": 64}))
    assert "ir_override" not in plain
    assert rec._deserialize_config(plain).ir_override is None


def test_a_config_with_a_pre_hook_is_recorded_only_for_libtuner():
    hooked = Config({"BM": 64}, pre_hook=lambda nargs: None)
    assert rec._serialize_config(hooked) is None
    assert rec._serialize_config(hooked, allow_pre_hook=True)["kwargs"] == {"BM": 64}


def test_a_recorded_config_that_no_longer_compiles_falls_back(capsys):
    rec._replay_map[KERNEL] = {'["32"]': {"kwargs": {"BM": 128}, "num_warps": 8, "num_stages": 2}}

    class Failing(Tuner):
        def run(self, *args, **kwargs):
            cfg = self.cache.get((kwargs["BLOCK"],))
            if cfg is not None and cfg.kwargs == {"BM": 128}:
                raise RuntimeError("ptxas: out of resources")
            return super().run(*args, **kwargs)

    Failing.run = rec._replay_run(Failing.run)
    used = Failing([LARGE, SMALL]).run(None, BLOCK=32)
    assert used is SMALL  # tuned on the spot once the recorded config was evicted
    assert "AUTOTUNE_REPLAY_FALLBACK reason=compile_RuntimeError\n" in capsys.readouterr().out


def test_a_kernel_the_record_does_not_have_is_reported_once(capsys):
    # The record holds another kernel of the op; this one tunes for itself. Its configs
    # may not have been storable (a pre_hook on a plain autotuner), or the op launches a
    # kernel now that it did not when the record was made.
    rec._replay_map["pkg.ops.mm::other_kernel"] = {}
    tuner = Replaying([SMALL, LARGE])
    assert tuner.run(None, BLOCK=32) is LARGE  # tuned live
    tuner.run(None, BLOCK=64)
    out = capsys.readouterr().out
    assert out.count("AUTOTUNE_REPLAY_FALLBACK reason=kernel_missing") == 1
    # a kernel with a single config is not tuned: nothing to replay, nothing to report
    Replaying([SMALL]).run(None, BLOCK=32)
    assert capsys.readouterr().out == ""


def test_a_missing_key_of_a_recorded_kernel_is_reported_each_time(capsys):
    rec._replay_map[KERNEL] = {}  # the kernel is in the record, without this key
    tuner = Replaying([SMALL, LARGE])
    tuner.run(None, BLOCK=32)
    tuner.run(None, BLOCK=64)
    assert capsys.readouterr().out.count("AUTOTUNE_REPLAY_FALLBACK reason=key_missing") == 2
