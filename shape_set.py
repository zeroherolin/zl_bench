#!/usr/bin/env python3
"""Shape sets: which entry of a shape yaml applies to which repo.

A shape set is one yaml for both upstream repos, keyed by op_name or Benchmark class
name. The repos share most names, and now and then one name with two meanings:
fused_recurrent_gated_delta_rule takes a sequence length in FlagGems and a case name in
FlagGems-vllm, so whichever benchmark is handed the other repo's entry fails to unpack
it. A key can therefore be scoped to a repo with a `<repo>:` prefix:

    softmax:                        # both repos
    vllm:fused_recurrent_gated_delta_rule:    # FlagGems-vllm only; beats the plain key there

Upstream reads the file itself and knows nothing of scopes, so every run gets the set
resolved for its repo; run_pytest.sh writes it to <run dir>/shapes.yaml:

    python shape_set.py resolve <set.yaml> <repo checkout> <out.yaml>

`python shape_set.py label <repo checkout>` prints the repo's label, main or vllm.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import yaml

SCOPE = ":"
REPOS = ("main", "vllm")  # the repo labels a key can be scoped to (repo_label)
# libyaml's loader where PyYAML was built with it: the all-ops sets take a tenth of the time
_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def split_key(key) -> Tuple[Optional[str], str]:
    """(repo scope or None, name) of a top-level key."""
    scope, sep, name = str(key).partition(SCOPE)
    return (scope, name) if sep else (None, str(key))


def resolve(entries: dict, repo: str) -> dict:
    """The set as `repo` sees it: plain keys, overridden by the keys scoped to `repo`;
    the keys scoped to another repo are left out. A set without scopes is unchanged."""
    out = {key: val for key, val in entries.items() if split_key(key)[0] is None}
    for key, val in entries.items():
        scope, name = split_key(key)
        if scope == repo:
            out[name] = val
    return out


class _PlainDumper(yaml.SafeDumper):
    """Writes a list that occurs twice in full both times. PyYAML would emit an anchor
    and aliases instead (&id001 / *id001), numbered in dump order, so one added key
    renumbers the rest of the file and a shape reads as *id007."""

    def ignore_aliases(self, data):
        return True


def dump_shapes(data: dict) -> str:
    """A shape set as text: sorted keys, one shape per line."""
    return yaml.dump(data, Dumper=_PlainDumper, sort_keys=True, default_flow_style=None,
                     allow_unicode=True)


def write_resolved(src: Path, repo: str, dst: Path) -> bool:
    """Write the set at src, resolved for repo, to dst. Returns whether anything was
    resolved; a set without scoped keys is copied as it is, comments included, and so
    is a file that cannot be read as a set (upstream then reports it in its own words).
    """
    try:
        entries = yaml.load(src.read_text(), Loader=_Loader)
    except (yaml.YAMLError, UnicodeDecodeError):
        entries = None
    if not isinstance(entries, dict) or all(split_key(k)[0] is None for k in entries):
        if not (dst.exists() and src.samefile(dst)):
            shutil.copyfile(src, dst)
        return False
    unknown = sorted(str(k) for k in entries if split_key(k)[0] not in (None,) + REPOS)
    if unknown:
        print(f"[shape-set] warning: {src.name} scopes {', '.join(unknown)} to a repo other than "
              f"{' / '.join(REPOS)}: such a key applies to no run", file=sys.stderr, flush=True)
    dst.write_text(f"# {src.name} as the repo '{repo}' sees it: its scoped keys applied, the\n"
                   "# other repo's left out (shape_set.py).\n" + dump_shapes(resolve(entries, repo)))
    return True


def repo_label(repo_dir: Path) -> str:
    """The label of an upstream checkout, as used in scopes, the inventory and
    run_batch.py's [repo:]op: "vllm" for FlagGems-vllm (package flaggems_vllm), else
    "main". Told by the package the checkout ships, so it holds wherever it lives."""
    return "vllm" if (repo_dir / "src" / "flaggems_vllm").is_dir() else "main"


USAGE = """usage: shape_set.py resolve <set.yaml> <repo checkout> <out.yaml>
   or: shape_set.py label <repo checkout>"""


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "label":
        print(repo_label(Path(sys.argv[2])))
        return
    if len(sys.argv) != 5 or sys.argv[1] != "resolve":
        raise SystemExit(USAGE)
    src, repo_dir, dst = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    if not src.is_file():
        raise SystemExit(f"no shape file {src}")
    try:
        write_resolved(src, repo_label(repo_dir), dst)
    except OSError as exc:
        raise SystemExit(f"cannot write the shape set for the run: {exc}") from None


if __name__ == "__main__":
    main()
