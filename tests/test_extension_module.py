"""The extension module asks the loader for no static TLS.

A shared object marked ``DF_STATIC_TLS`` must fit its whole thread-local block into the small
reserve glibc keeps for libraries loaded after startup. The core is always such a library
(Python loads it with ``dlopen``), and so is anything imported after it: with the flag set, the
OpenMP runtime that scikit-learn and torch load could no longer find room, and ``import
simplipy`` followed by ``import sklearn`` failed with "cannot allocate memory in static TLS
block". The flag comes from initial-exec thread-local accesses, which the allocator would make
unless built with local-dynamic TLS (Cargo.toml).
"""
import struct
import sys

import pytest

import simplipy._core as core

PT_DYNAMIC = 2
DT_NULL = 0
DT_FLAGS = 30
DF_STATIC_TLS = 0x10


def dynamic_flags(path: str) -> int:
    """``DT_FLAGS`` of a 64-bit little-endian ELF shared object (0 when it has none)."""
    with open(path, 'rb') as fh:
        data = fh.read()
    if data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
        raise ValueError(f'{path} is not a 64-bit little-endian ELF file')
    phoff, = struct.unpack_from('<Q', data, 0x20)
    phentsize, phnum = struct.unpack_from('<HH', data, 0x36)
    for i in range(phnum):
        p_type, _flags, p_offset, _vaddr, _paddr, p_filesz = struct.unpack_from('<IIQQQQ', data, phoff + i * phentsize)
        if p_type != PT_DYNAMIC:
            continue
        for j in range(p_offset, p_offset + p_filesz, 16):
            tag, value = struct.unpack_from('<qQ', data, j)
            if tag == DT_NULL:
                break
            if tag == DT_FLAGS:
                return value
    return 0


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='DF_STATIC_TLS is an ELF flag')
def test_the_core_needs_no_static_tls():
    assert not dynamic_flags(core.__file__) & DF_STATIC_TLS, (
        f'{core.__file__} is marked DF_STATIC_TLS: a library imported after simplipy that needs '
        'static TLS (OpenMP in scikit-learn or torch) can fail to load')
