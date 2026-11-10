import _shape_override_plugin as plugin


class FakeBenchmark:
    def __init__(self, op_name, shapes):
        self.op_name, self._shapes = op_name, shapes

    def init_user_config(self):
        self.shapes = list(self._shapes)  # what a hardcoded set_shapes() leaves behind


def _patched(monkeypatch, entries):
    monkeypatch.setattr(plugin, "_entries", entries)

    class Bench(FakeBenchmark):
        pass

    plugin._patch_init_user_config(Bench)
    return Bench


def test_yaml_shapes_override_hardcoded(monkeypatch):
    Bench = _patched(monkeypatch, {"moe": {"shapes": [[1, 2, 3]], "shape_desc": "M, N, K"}})
    b = Bench("moe", [(9, 9, 9), (8, 8, 8)])
    b.init_user_config()
    assert b.shapes == [(1, 2, 3)]
    assert b.shape_desc == "M, N, K"


def test_ops_without_entry_keep_their_shapes(monkeypatch):
    Bench = _patched(monkeypatch, {"other": {"shapes": [[1]]}})
    b = Bench("moe", [(9, 9, 9)])
    b.init_user_config()
    assert b.shapes == [(9, 9, 9)]


def test_arity_normalization_is_preserved(monkeypatch):
    Bench = _patched(monkeypatch, {"mm": {"shapes": [[2, 64, 64, 64]]}})
    b = Bench("mm", [(64, 64, 64)])  # subclass normalized (B, M, N, K) -> (M, N, K)
    b.init_user_config()
    assert b.shapes == [(64, 64, 64)]


def test_standard_input_fn_signature():
    assert plugin._takes_shape_dtype_device(lambda shape, dtype, device: None)
    assert not plugin._takes_shape_dtype_device(lambda b, m, n, k, dtype, device, t: None)
    assert not plugin._takes_shape_dtype_device(None)


def _session(monkeypatch, capsys, entries, ops_that_ran):
    """Run the end-of-session check for a given yaml and a given set of ops."""
    monkeypatch.setattr(plugin, "_entries", entries)
    monkeypatch.setattr(plugin, "_looked_up", set())
    monkeypatch.setattr(plugin, "_ops_run", set())
    Bench = _patched(monkeypatch, entries)
    for op_name in ops_that_ran:
        Bench(op_name, [(9, 9, 9)]).init_user_config()
    plugin.pytest_sessionfinish(None, 0)
    return capsys.readouterr().err


def test_renamed_key_is_reported(monkeypatch, capsys):
    # the op ran but the yaml still carries its old name: it silently used its own shapes
    err = _session(monkeypatch, capsys, dict(grouped_topk_old=dict(shapes=[[1, 2]])), ["grouped_topk"])
    assert "SHAPE_KEYS_UNUSED" in err and "grouped_topk" in err


def test_matching_key_is_silent(monkeypatch, capsys):
    err = _session(monkeypatch, capsys, dict(grouped_topk=dict(shapes=[[1, 2]])), ["grouped_topk"])
    assert "SHAPE_KEYS_UNUSED" not in err


def test_keys_of_ops_that_did_not_run_are_not_reported(monkeypatch, capsys):
    # a whole-suite set holds keys for every op; running one of them is not a problem
    entries = dict(mm=dict(shapes=[[1, 2]]), softmax=dict(shapes=[[3]]))
    assert "SHAPE_KEYS_UNUSED" not in _session(monkeypatch, capsys, entries, ["mm"])


def test_no_shape_file_is_silent(monkeypatch, capsys):
    assert "SHAPE_KEYS_UNUSED" not in _session(monkeypatch, capsys, dict(), ["mm"])
