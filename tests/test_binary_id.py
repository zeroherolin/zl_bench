import struct

import _binary_id


def _elf(sections):
    """A minimal ELF64 image with the given (name, type, content) sections."""
    names = b"\0"
    offsets = []
    for name, _kind, _body in sections + [(".shstrtab", 3, b"")]:
        offsets.append(len(names))
        names += name.encode() + b"\0"
    bodies = [body for _n, _k, body in sections] + [names]
    kinds = [kind for _n, kind, _b in sections] + [3]
    data = bytearray(64)
    heads = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]  # the null section
    for name_off, kind, body in zip(offsets, kinds, bodies, strict=True):
        heads.append(struct.pack("<IIQQQQIIQQ", name_off, kind, 0, 0, len(data), len(body), 0, 0, 1, 0))
        data += body
    shoff = len(data)
    data += b"".join(heads)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", data, 0x28, shoff)
    struct.pack_into("<HHH", data, 0x3A, 64, len(heads), len(heads) - 1)
    return bytes(data)


TEXT = (".text.kernel", 1, b"\x01\x02\x03\x04" * 8)
INFO = (".nv.info", 0x70000000, b"regs=40")
META = {"shared": 1024, "num_warps": 4, "num_stages": 3}


def test_line_info_does_not_change_the_id():
    # Two builds installed in different directories emit the same code and a
    # .debug_line that names their own source paths: that is still one binary.
    a = _elf([TEXT, INFO, (".debug_line", 1, b"/build/a/triton/x.py:12"), (".rela.debug_line", 4, b"A" * 24)])
    b = _elf([TEXT, INFO, (".debug_line", 1, b"/other/b/triton/x.py:12"), (".rela.debug_line", 4, b"B" * 24)])
    assert a != b
    assert _binary_id.fingerprint("k", a, META) == _binary_id.fingerprint("k", b, META)
    # An image without any line info is not the same file layout (its section name table
    # differs), so it gets another id: the id errs towards "different", which costs a
    # control row, never a wrong one.
    assert _binary_id.fingerprint("k", a, META) != _binary_id.fingerprint("k", _elf([TEXT, INFO]), META)


def test_code_name_and_launch_metadata_change_the_id():
    base = _binary_id.fingerprint("k", _elf([TEXT, INFO]), META)
    other_code = _elf([(".text.kernel", 1, b"\x01\x02\x03\x05" * 8), INFO])
    other_info = _elf([TEXT, (".nv.info", 0x70000000, b"regs=48")])
    renamed = _elf([(".text.other", 1, TEXT[2]), INFO])
    ids = [base,
           _binary_id.fingerprint("k", other_code, META),
           _binary_id.fingerprint("k", other_info, META),
           _binary_id.fingerprint("k", renamed, META),
           _binary_id.fingerprint("k2", _elf([TEXT, INFO]), META),
           _binary_id.fingerprint("k", _elf([TEXT, INFO]), dict(META, shared=2048))]
    assert len(set(ids)) == len(ids)
    # num_stages is a compile option, not a launch parameter: same code, same launch
    assert _binary_id.fingerprint("k", _elf([TEXT, INFO]), dict(META, num_stages=2)) == base


def test_bytes_that_are_not_elf_are_hashed_whole():
    fp = _binary_id.fingerprint
    assert fp("k", b"not an elf", META) != fp("k", b"not an elg", META)
    whole = _elf([TEXT, INFO])
    truncated = whole[:80]  # an ELF magic, but the section table is gone
    other = truncated[:-1] + bytes([truncated[-1] ^ 1])  # hashed whole: one byte tells them apart
    assert len({fp("k", truncated, META), fp("k", other, META), fp("k", whole, META)}) == 3


def test_a_big_endian_image_is_hashed_whole():
    # the section table of such an image would be read as garbage; the id still tells
    # two images apart
    fp = _binary_id.fingerprint
    big = bytearray(_elf([TEXT, INFO]))
    big[5] = 2
    other = bytearray(big)
    other[-1] ^= 1
    assert fp("k", bytes(big), META) != fp("k", bytes(other), META)
    assert fp("k", bytes(big), META) != fp("k", _elf([TEXT, INFO]), META)
