"""What a launch of a compiled kernel runs, as one comparable id.

Two sides of an A/B run the same binary when the kernel's name, the code of its cubin
and the launch metadata Triton derives agree. That is what ir_gate.py compares between
two builds offline; this module gives the same answer for the kernels a run really
launched, so that _ir_dump_plugin can record it per result row and ab_compare.py can
hold the gate's verdict against it.

Neither the Triton cache hash nor the cubin's bytes will do. The cache hash covers the
compiler's own sources and so differs between any two builds. The cubin carries line
info unless it is turned off, and line info names source files by path, among them
files of the Triton install itself: two builds in different directories emit cubins
that differ in .debug_line and nowhere else. So the id covers the sections of the
cubin other than debug info. It errs towards "different": two images that only one of
which has line info at all (TRITON_DISABLE_LINE_INFO on one side) differ in their
section tables and get two ids, which costs a control row and never makes one.
"""

from __future__ import annotations

import hashlib
import json
import struct

# What the launcher reads from the metadata Triton derives during compilation (the
# CudaLauncher of third_party/nvidia/backend/driver.py, and the shared memory and warps
# of the kernel): a difference changes the launch even when the code is identical.
LAUNCH_META = ("shared", "num_warps", "num_ctas", "cluster_dims", "tmem_size",
               "global_scratch_size", "global_scratch_align",
               "profile_scratch_size", "profile_scratch_align",
               "launch_cooperative_grid", "launch_pdl")

_SHT_NOBITS = 8


def _is_debug(name: str) -> bool:
    return ".debug" in name or name.startswith(".nv_debug")


def code_digest(cubin: bytes) -> "hashlib._Hash":
    """A sha256 over the cubin's sections, debug info left out: name, type and content
    of each, in file order. Bytes that are not an ELF64 image are hashed as they are."""
    h = hashlib.sha256()
    try:
        if cubin[:4] != b"\x7fELF" or cubin[4] != 2 or cubin[5] != 1:
            raise ValueError("not a little-endian ELF64 image")
        shoff = struct.unpack_from("<Q", cubin, 0x28)[0]
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", cubin, 0x3A)
        heads = [struct.unpack_from("<IIQQQQIIQQ", cubin, shoff + i * shentsize) for i in range(shnum)]
        names_off, names_size = heads[shstrndx][4], heads[shstrndx][5]
        names = cubin[names_off:names_off + names_size]
        parts = []
        for name_off, kind, _flags, _addr, off, size, *_rest in heads:
            name = names[name_off:names.index(b"\0", name_off)].decode(errors="replace")
            if _is_debug(name):
                continue
            body = b"" if kind == _SHT_NOBITS else cubin[off:off + size]
            parts.append((name, kind, body))
    except (ValueError, IndexError, struct.error):
        h.update(cubin)
        return h
    for name, kind, body in parts:
        h.update(f"{name}\0{kind}\0{len(body)}\0".encode())
        h.update(body)
    return h


def fingerprint(kernel: str, cubin: bytes, meta: dict) -> str:
    """The id of a compiled variant: kernel name, code (code_digest) and launch metadata."""
    h = code_digest(cubin)
    h.update(kernel.encode())
    launch = {k: meta[k] for k in LAUNCH_META if k in meta}
    h.update(json.dumps(launch, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]
