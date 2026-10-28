"""Session-wide configuration for the Python suites of UForm.

Holds only what pytest itself loads: the report header and the handles every module shares, so a module
asks for an executor, a model or an arena by name and imports nothing but `uform`. The suites run against
the real library with the tiny checkpoint `UFORM_TEST_CHECKPOINT` names, and against the stub library,
which loads any bytes as a model; shapes always come from the model's own description.
"""

import os
import platform
from collections.abc import Iterator
from pathlib import Path

import pytest

import uform


def pytest_report_header() -> list[str]:
    """What this run is actually exercising, printed where pytest prints its own header."""
    return [
        f"- Platform: {platform.platform()}",
        f"- UForm: {uform.version()} from {uform.lib.__file__}",
        f"- Capabilities compiled: {uform.capabilities_compiled()}",
        f"- Capabilities available: {uform.capabilities_available()}",
        f"- Checkpoint: {os.environ.get('UFORM_TEST_CHECKPOINT', 'none, the stub loads any bytes')}",
    ]


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
