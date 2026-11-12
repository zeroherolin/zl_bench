import os
import subprocess
import sys
from pathlib import Path

import yaml

import shape_set

ROOT = Path(__file__).resolve().parent.parent
SET = """\
# a comment the author wants to keep
softmax:
  shapes:
  - [1024, 1024]
bmm:
  shape_desc: B, M, N, K
  shapes:
  - [1, 1, 1024, 4096]
vllm:bmm:
  shapes:
  - [4, 129, 17, 257]
vllm:recurrent:
  shapes:
  - [decode64]
main:only_main:
  shapes:
  - [8]
"""


def test_split_key():
    assert shape_set.split_key("softmax") == (None, "softmax")
    assert shape_set.split_key("vllm:softmax") == ("vllm", "softmax")
    assert shape_set.split_key(3) == (None, "3")  # a yaml key need not be a string


def test_resolve_applies_the_repo_s_keys_and_drops_the_others():
    entries = yaml.safe_load(SET)
    vllm = shape_set.resolve(entries, "vllm")
    assert sorted(vllm) == ["bmm", "recurrent", "softmax"]
    assert vllm["bmm"]["shapes"] == [[4, 129, 17, 257]]  # the scoped key beats the plain one
    main = shape_set.resolve(entries, "main")
    assert sorted(main) == ["bmm", "only_main", "softmax"]
    assert main["bmm"]["shapes"] == [[1, 1, 1024, 4096]]
    # resolving is idempotent: a resolved set has no scopes left
    assert shape_set.resolve(vllm, "vllm") == vllm and shape_set.resolve(vllm, "main") == vllm


def test_write_resolved(tmp_path):
    src, dst = tmp_path / "set.yaml", tmp_path / "out" / "shapes.yaml"
    dst.parent.mkdir()
    src.write_text(SET)
    assert shape_set.write_resolved(src, "vllm", dst) is True
    assert yaml.safe_load(dst.read_text()) == shape_set.resolve(yaml.safe_load(SET), "vllm")
    assert "as the repo 'vllm' sees it" in dst.read_text().splitlines()[0]


def test_a_set_without_scopes_is_copied_verbatim(tmp_path):
    # curated sets carry comments and anchors; they must reach the run dir untouched
    text = "# keep me\nmm: &s\n  shape_desc: a:b\n  shapes:\n  - [1]\nbmm: *s\n"
    src, dst = tmp_path / "set.yaml", tmp_path / "shapes.yaml"
    src.write_text(text)
    assert shape_set.write_resolved(src, "main", dst) is False
    assert dst.read_text() == text
    # a replayed run archives the shapes.yaml of the run it replays: same file, no error
    assert shape_set.write_resolved(dst, "main", dst) is False and dst.read_text() == text


def test_an_unreadable_set_is_left_to_upstream(tmp_path):
    src, dst = tmp_path / "set.yaml", tmp_path / "shapes.yaml"
    src.write_text("vllm:mm:\n  shapes: [a, b\n")
    assert shape_set.write_resolved(src, "main", dst) is False
    assert dst.read_text() == src.read_text()


def test_a_quoted_scoped_key_is_resolved_too(tmp_path):
    src, dst = tmp_path / "set.yaml", tmp_path / "shapes.yaml"
    src.write_text('mm:\n  shapes:\n  - [1]\n"vllm:mm":\n  shapes:\n  - [2]\n')
    assert shape_set.write_resolved(src, "vllm", dst) is True
    assert yaml.safe_load(dst.read_text()) == {"mm": {"shapes": [[2]]}}


def test_a_scope_that_names_no_repo_is_pointed_out(tmp_path, capsys):
    src, dst = tmp_path / "set.yaml", tmp_path / "shapes.yaml"
    src.write_text("mm:\n  shapes:\n  - [1]\nvlm:mm:\n  shapes:\n  - [2]\n")
    assert shape_set.write_resolved(src, "vllm", dst) is True
    assert "scopes vlm:mm to a repo other than main / vllm" in capsys.readouterr().err
    assert yaml.safe_load(dst.read_text()) == {"mm": {"shapes": [[1]]}}  # it applies to no run
    src.write_text(SET)
    shape_set.write_resolved(src, "vllm", dst)
    assert capsys.readouterr().err == ""


def test_a_run_whose_shape_set_cannot_be_prepared_says_so_in_its_log(tmp_path):
    # run_pytest.sh stops before pytest; under run_batch.py the log is all that is kept
    repo, out = tmp_path / "repo", tmp_path / "out"
    (repo / "benchmark").mkdir(parents=True)
    (repo / "benchmark" / "test_x.py").write_text("")
    (out / "shapes.yaml").mkdir(parents=True)  # the archive cannot be written
    src = tmp_path / "set.yaml"
    src.write_text(SET)
    env = dict(os.environ, OP="x", OP_FILE="x", FLAGGEMS_DIR=str(repo), OUT_DIR=str(out),
               SHAPE_FILE=str(src), REPLAY_FROM="")
    env.pop("BASH_ENV", None)  # a container's shell init has no part in this (and may be slow)
    run = subprocess.run(["bash", str(ROOT / "run_pytest.sh")], env=env, capture_output=True, text=True)
    assert run.returncode == 2 and "cannot prepare the shape set" in run.stderr
    log = (out / "run.log").read_text()
    assert "cannot prepare the shape set" in log and "Traceback" not in log


def test_repo_label_and_cli(tmp_path):
    vllm = tmp_path / "anything"
    (vllm / "src" / "flaggems_vllm").mkdir(parents=True)
    assert shape_set.repo_label(vllm) == "vllm"
    assert shape_set.repo_label(tmp_path / "FlagGems") == "main"
    src, dst = tmp_path / "set.yaml", tmp_path / "shapes.yaml"
    src.write_text(SET)
    cmd = [sys.executable, str(ROOT / "shape_set.py"), "resolve", str(src), str(vllm), str(dst)]
    subprocess.run(cmd, check=True)
    assert yaml.safe_load(dst.read_text())["recurrent"]["shapes"] == [["decode64"]]
    label = [sys.executable, str(ROOT / "shape_set.py"), "label", str(vllm)]
    assert subprocess.run(label, capture_output=True, text=True, check=True).stdout == "vllm\n"
    cmd[3] = str(tmp_path / "nope.yaml")
    missing = subprocess.run(cmd, capture_output=True, text=True)
    assert missing.returncode != 0 and "no shape file" in missing.stderr


def test_the_shipped_sets_resolve_for_both_repos():
    for path in sorted((ROOT / "shape").glob("*.yaml")):
        entries = yaml.safe_load(path.read_text())
        for repo in ("main", "vllm"):
            view = shape_set.resolve(entries, repo)
            assert view and all(shape_set.split_key(k)[0] is None for k in view), path.name
            assert all(isinstance(v, dict) and v.get("shapes") for v in view.values()), path.name
