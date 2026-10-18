"""The vocabulary and the model lifecycle: enums, failures, executors, loads, receivers and progress.

Run with `uv run pytest test/python/model.py` once the extension is built against the stub library or the
engine; every check reads its expectations from the C contract, never from the stub's choices.
"""

from __future__ import annotations

import asyncio
from enum import IntEnum, IntFlag
from pathlib import Path

import pytest

import uform

FAILING_STATUSES = [status for status in uform.Status if status not in (uform.Status.SUCCESS, uform.Status.DEFERRED)]


def test_version_and_capabilities_are_reported():
    """The version is the package's, and both capability lists are non-empty strings."""
    assert uform.version() == uform.__version__
    assert uform.capabilities_compiled()
    assert uform.capabilities_available()


@pytest.mark.parametrize("name", ["Status", "Scalar", "Mood", "Modality", "Objective", "Negatives", "MultimemTier"])
def test_modes_are_int_enums_of_the_package(name: str):
    """Every C enum is an `IntEnum` of the `uform` module, never a string."""
    mirrored = getattr(uform, name)
    assert issubclass(mirrored, IntEnum)
    assert mirrored.__module__ == "uform"


def test_enum_values_follow_the_header():
    """Members carry the C values, gaps in the numbering included."""
    assert uform.Status.SHAPE_MISMATCH == 3 and uform.Status.IO == 7 and uform.Status.INVALID_ARGUMENT == 17
    assert uform.Scalar.BF16 == 3 and uform.Scalar.E4M3 == 6 and uform.Scalar.U1X8 == 22
    assert uform.Modality.AUDIO == 3 and uform.Objective.JOINT == 2 and uform.MultimemTier.PEER_READ == 1
    assert "UNRECOGNIZED" not in uform.Status.__members__


def test_outputs_is_a_flag_set():
    """`Outputs` combines with `|`, one flag per output field of the encode verbs."""
    assert issubclass(uform.Outputs, IntFlag)
    both = uform.Outputs.EMBEDDINGS | uform.Outputs.STATES
    assert int(both) == 3 and uform.Outputs.STATES in both


def test_every_failing_status_raises_its_own_class():
    """One `Error` subclass per failing status, deriving the built-in that names the same fault."""
    subclasses = {
        value
        for value in vars(uform).values()
        if isinstance(value, type) and issubclass(value, uform.Error) and value is not uform.Error
    }
    assert len(subclasses) == len(FAILING_STATUSES)
    assert issubclass(uform.Error, RuntimeError)
    assert issubclass(uform.ShapeMismatchError, ValueError)
    assert issubclass(uform.InputOutputError, OSError)
    assert issubclass(uform.OutOfMemoryError, MemoryError)
    assert issubclass(uform.UnsupportedError, NotImplementedError)
    assert not issubclass(uform.ContendedError, ValueError)


def test_raised_failures_carry_their_status(tmp_path: Path):
    """A failure the library reports raises the subclass of its status, whose `status` names it."""
    with pytest.raises(uform.InputOutputError) as caught:
        uform.Model.load_ram(tmp_path / "missing.safetensors")
    assert caught.value.status is uform.Status.IO
    assert isinstance(caught.value, OSError)


def test_unknown_enum_values_are_refused(checkpoint: Path):
    """An integer no member carries is an unknown kind, and a bool or a string is no member at all."""
    with pytest.raises(uform.UnknownKindError):
        uform.Model.load_ram(checkpoint, quantization=99)
    with pytest.raises(TypeError):
        uform.Model.load_ram(checkpoint, quantization="bf16")


def test_executor_drains_and_closes(model: uform.Model):
    """An executor drains in either mood and closes on leaving its `with` block, refusing later receivers."""
    with uform.Executor.for_ram(threads=1, deferred_limit=4) as executor:
        executor.drain(uform.Mood.SPINNING)
        executor.drain()
    with pytest.raises(uform.InvalidArgumentError):
        model.on(executor)


def test_executor_is_built_through_its_residency():
    """`Executor()` names no residency, so only the class methods build one."""
    with pytest.raises(TypeError):
        uform.Executor()


def test_borrowed_pools_are_unsupported():
    """`on_pool` mirrors the C verb, which reports unsupported until borrowing is designed across projects."""
    with pytest.raises(uform.UnsupportedError) as caught:
        uform.Executor.on_pool(0)
    assert isinstance(caught.value, NotImplementedError)


def test_load_ram_and_describe(checkpoint: Path):
    """A load from a path answers a model whose description carries its geometry."""
    model = uform.Model.load_ram(checkpoint, quantization=uform.Scalar.BF16)
    description = model.describe()
    dimensions, context, quantization, memory_usage, capabilities = description
    assert dimensions > 0 and context > 0 and memory_usage > 0
    assert quantization is uform.Scalar.BF16
    assert isinstance(capabilities, str)


def test_load_ram_buffer_matches_the_path(checkpoint: Path):
    """A load from the checkpoint's bytes describes the same geometry as a load from its path."""
    from_path = uform.Model.load_ram(checkpoint).describe()
    from_buffer = uform.Model.load_ram_buffer(checkpoint.read_bytes()).describe()
    assert (from_path.dimensions, from_path.context) == (from_buffer.dimensions, from_buffer.context)


def test_load_refuses_a_quantization_the_blocks_never_run_in(checkpoint: Path):
    """Blocks run in bf16 or e4m3, so f32 is declared but unsupported."""
    with pytest.raises(uform.UnsupportedError):
        uform.Model.load_ram(checkpoint, quantization=uform.Scalar.F32)


def test_load_through_an_executor(checkpoint: Path, executor: uform.Executor):
    """A load may run on an executor, and answers the same model."""
    assert uform.Model.load_ram(checkpoint, executor=executor).describe().context > 0


def test_progress_cancels_a_load(checkpoint: Path):
    """A handler answering False stops the load, which raises `CancelledError`, a `uform.Error`."""
    reports: list[tuple[int, int]] = []

    def stop(completed: int, total: int) -> bool:
        reports.append((completed, total))
        return False

    with pytest.raises(uform.CancelledError) as caught:
        uform.Model.load_ram(checkpoint, progress=stop)
    assert isinstance(caught.value, uform.Error) and caught.value.status is uform.Status.CANCELLED
    assert reports and all(0 < completed <= total for completed, total in reports)


def test_progress_exception_outranks_the_cancellation(checkpoint: Path):
    """A handler that raises stops the load, and its own exception reaches the caller."""

    def explode(completed: int, total: int) -> bool:
        raise LookupError("from the handler")

    with pytest.raises(LookupError, match="from the handler"):
        uform.Model.load_ram(checkpoint, progress=explode)


def test_vram_residency_reports_its_absence(checkpoint: Path):
    """A build without GPUs refuses every VRAM verb with `DeviceError`."""
    if "serial" not in uform.capabilities_compiled():
        pytest.skip("the build carries a GPU backend")
    with pytest.raises(uform.DeviceError):
        uform.Executor.for_vram([0])
    with pytest.raises(uform.DeviceError):
        uform.Model.load_vram(checkpoint, devices=[0])
    with pytest.raises(uform.DeviceError):
        uform.Model.load_vram_buffer(checkpoint.read_bytes())


def test_receivers_answer_alike(model: uform.Model, executor: uform.Executor):
    """The model, `on` and `deferred` run one verb with one answer."""
    direct = model.describe()
    assert model.on(executor).describe() == direct
    call = model.deferred(executor).describe()
    assert isinstance(call, uform.Deferred)
    assert call.wait() == direct
    assert call.wait() == direct, "a final call answers as many times as it is asked"
    assert call.status is uform.Status.SUCCESS


def test_deferred_awaits(model: uform.Model, executor: uform.Executor):
    """`await` runs `wait()` on the event loop's default executor, so the coroutine suspends rather than blocks."""

    async def describe() -> uform.ModelDescription:
        return await model.deferred(executor).describe()

    assert asyncio.run(describe()) == model.describe()


def test_receivers_are_built_through_the_target(model: uform.Model, executor: uform.Executor):
    """A receiver is answered by `on` or `deferred` and names its target type in subscripts."""
    with pytest.raises(TypeError):
        uform.On()
    assert repr(model.on(executor)).startswith("On(")
    assert uform.On[uform.Model] is not None


def test_model_is_built_through_a_load():
    """`Model()` names no checkpoint, so only the loads build one."""
    with pytest.raises(TypeError):
        uform.Model()
