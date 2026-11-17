"""Settings, gates and inputs the Python suites of UForm share.

Declares no test of its own; the suites import what they need from here, never from each other.

Run:
    python -m pytest test/python
    UFORM_SEED=random python -m pytest test/python
    python -m pytest test/python -k "encode and deferred"

File: test/python/base.py
Author: Ash Vardanian
Date: November 16, 2026
"""

import importlib.util
import os
import random
import struct
import zlib
from dataclasses import dataclass
from typing import NoReturn

import pytest

import uform

# region Shared Settings
_UINT64_MASK = 2**64 - 1


def env_text(variable: str) -> str | None:
    """The stripped text of the environment `variable`, or None when unset or blank."""
    text = os.environ.get(variable, "").strip()
    return text or None


def exit_unparsed(variable: str, text: str, expected: str) -> NoReturn:
    """Stops the session on a setting that does not parse, naming it as the C and C++ suites do."""
    raise SystemExit(f'{variable}="{text}" does not parse, expected {expected}')


def env_seed(variable: str) -> int:
    """An unsigned 64-bit seed: 42 when unset, drawn afresh for `random`."""
    text = env_text(variable)
    if text is None:
        return 42
    if text == "random":
        return random.SystemRandom().getrandbits(64)
    if not text.isdigit() or int(text) > _UINT64_MASK:
        exit_unparsed(variable, text, "an unsigned integer or random")
    return int(text)


def env_real(variable: str, fallback: float) -> float:
    """A positive real, `fallback` when unset."""
    text = env_text(variable)
    if text is None:
        return fallback
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not value > 0 or value == float("inf"):
        exit_unparsed(variable, text, "a positive number like 0.1 or 10")
    return value


def env_count(variable: str, fallback: int) -> int:
    """A whole number, `fallback` when unset."""
    text = env_text(variable)
    if text is None:
        return fallback
    if not text.isdigit():
        exit_unparsed(variable, text, "a whole number")
    return int(text)


def env_flag(variable: str, fallback: bool = False) -> bool:
    """A switch, `fallback` when unset: 0, false, off and no turn it off."""
    text = env_text(variable)
    if text is None:
        return fallback
    return text not in {"0", "false", "off", "no"}


def mix(value: int) -> int:
    """SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def stream_key(seed: int, name: str, index: int = 0) -> int:
    """The `index`-th stream of the family `name` under `seed`, identical in C, C++, Rust and Mojo."""
    hashed = 0xCBF29CE484222325
    for byte in name.encode():
        hashed = ((hashed ^ byte) * 0x100000001B3) & _UINT64_MASK
    return mix((mix(seed ^ hashed) + index) & _UINT64_MASK)


# endregion Shared Settings


@dataclass(frozen=True)
class Settings:
    """Every `UFORM_*` variable the Python suites read, parsed once at import."""

    seed: int
    scale: float


SETTINGS = Settings(seed=env_seed("UFORM_SEED"), scale=env_real("UFORM_SCALE", 1.0))

BACKEND_AXES: set[str] = set()
"""The axes the seed and the case cap leave out: none, as every suite runs on the one library it imports."""

MAX_CASES_PER_TEST = 12
"""The most cases one test function may expand to, today's largest, so a new axis cannot multiply a
matrix past it unnoticed.
"""

numpy_available = importlib.util.find_spec("numpy") is not None
"""Whether NumPy is installed, which the README Quickstarts use and the suites themselves never import."""

requires_cpu_only_build = pytest.mark.skipif(
    "serial" not in uform.capabilities_compiled(), reason="needs a build without GPU backends"
)
"""Marks a test of the refusals a build without GPUs answers in place of the device verbs."""


def png(side: int = 8) -> bytes:
    """A valid `side` x `side` RGB PNG of a gradient."""
    rows = b"".join(b"\0" + bytes((row * 32 + byte * 8) & 0xFF for byte in range(side * 3)) for row in range(side))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
