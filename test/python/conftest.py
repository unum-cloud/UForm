"""Session-wide configuration for the Python suites of UForm.

Holds only what pytest itself loads: the report header and the handles every module shares, so a module
asks for an executor, a model or an arena by name and imports nothing but `uform` and `base`. The suites
run against the real library with the tiny checkpoint `UFORM_TEST_CHECKPOINT` names, and against the stub
library, which loads any bytes as a model; shapes always come from the model's own description.

File: test/python/conftest.py
Author: Ash Vardanian
Date: October 18, 2026
"""

import collections
import enum
import os
import platform
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from base import BACKEND_AXES, MAX_CASES_PER_TEST, SETTINGS, stream_key

import uform

WAIT_LIMIT = 120
"""How long a test waits for what it expects to happen, in seconds. A passing test never waits it out, so it
bounds hangs rather than speed, and leaves room for sanitizers and emulated CPUs."""


def pytest_report_header() -> list[str]:
    """What this run is actually exercising, printed where pytest prints its own header."""
    return [
        f"UForm {uform.version()} from {uform.lib.__file__}",
        f"- Platform: {platform.platform()}",
        f"- Capabilities compiled: {uform.capabilities_compiled()}",
        f"- Capabilities available: {uform.capabilities_available()}",
        f"- Checkpoint: {os.environ.get('UFORM_TEST_CHECKPOINT', 'none, the stub loads any bytes')}",
        f"- Seed: {SETTINGS.seed}, scale: {SETTINGS.scale:g}, filter: {SETTINGS.filter.pattern or 'none'}",
    ]


# region Shared Pytest
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Refuses a test whose cases outgrow MAX_CASES_PER_TEST, then deselects what the filter setting leaves out."""
    variants: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for item in items:
        callspec = getattr(item, "callspec", None)
        parameters = dict(callspec.params) if callspec is not None else {}
        parameters.pop("__pytest_repeat_step_number", None)
        kept = {key: value for key, value in parameters.items() if key not in BACKEND_AXES}
        variants[(item.module.__name__, getattr(item, "originalname", item.name))].add(repr(kept))
    oversized = [
        f"{module}::{name} has {len(cases)}"
        for (module, name), cases in variants.items()
        if len(cases) > MAX_CASES_PER_TEST
    ]
    if oversized:
        raise pytest.UsageError(f"over MAX_CASES_PER_TEST={MAX_CASES_PER_TEST}: " + "; ".join(oversized))
    deselected = [item for item in items if not SETTINGS.filter.selects(item.nodeid)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if SETTINGS.filter.selects(item.nodeid)]


def pytest_make_parametrize_id(config: pytest.Config, val: object, argname: str) -> str | None:
    """Names enum parameters as the code spells them, `Backend.LINUX_URING618`, so an id greps to its source."""
    return f"{type(val).__name__}.{val.name}" if isinstance(val, enum.Enum) else None


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    """Share inputs across backends while keeping other parameters and repeat steps distinct."""
    callspec = getattr(request.node, "callspec", None)
    parameters = dict(callspec.params) if callspec is not None else {}
    repeat_step = parameters.pop("__pytest_repeat_step_number", 0)
    name = request.node.originalname or request.node.name
    name += repr({key: value for key, value in parameters.items() if key not in BACKEND_AXES})
    return stream_key(SETTINGS.seed, name, repeat_step)


@pytest.fixture
def rng(seed: int) -> random.Random:
    """A generator private to this test, so a neighbour's draws cannot shift this one's."""
    return random.Random(seed)


@pytest.fixture
def unraisable(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Collects what the interpreter reports as unraisable, like a finalizer's exception, for the test to check."""
    seen: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    return seen


# endregion Shared Pytest


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """The tiny checkpoint `UFORM_TEST_CHECKPOINT` names, or a file of stand-in bytes only the stub loads."""
    named = os.environ.get("UFORM_TEST_CHECKPOINT")
    if named:
        return Path(named)
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"stand-in checkpoint bytes")
    return path


@pytest.fixture
def wait_limit() -> int:
    """`WAIT_LIMIT`, for a timeout a test hands the library."""
    return WAIT_LIMIT


@pytest.fixture
def executor() -> Iterator[uform.Executor]:
    """Host threads private to one test, closed once it ends, so a test's deferred calls never leak into the next."""
    with uform.Executor.for_ram() as pool:
        yield pool


@pytest.fixture
def model(checkpoint: Path) -> uform.Model:
    """A model loaded on the calling thread from the session's checkpoint."""
    return uform.Model.load_ram(checkpoint)


@pytest.fixture
def arena(executor: uform.Executor) -> uform.Arena:
    """An arena made where `executor` computes."""
    return uform.Arena(executor)
