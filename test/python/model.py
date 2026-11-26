"""The vocabulary and the model lifecycle: enums, failures, executors, loads, receivers and progress.

Every check reads its expectations from the C contract, never from the stub's choices, so the module runs
against the stub library and the engine alike.

Run:
    uv run pytest test/python/model.py

File: test/python/model.py
Author: Ash Vardanian
Date: October 18, 2026
"""

import asyncio
from enum import IntEnum, IntFlag
from pathlib import Path

import pytest
from base import requires_cpu_only_build

import uform

FAILING_STATUSES = [status for status in uform.Status if status not in (uform.Status.SUCCESS, uform.Status.DEFERRED)]


def test_capabilities_are_reported():
    """Both capability lists are non-empty strings."""
    assert uform.capabilities_compiled()
    assert uform.capabilities_available()


@pytest.mark.parametrize(
    "name",
    [
        "Status",
        "Scalar",
        "Mood",
        "CPUClass",
        "GPUClass",
        "GPUVendor",
        "Modality",
        "Objective",
        "Negatives",
        "PairLossForm",
        "LateInteractionSide",
        "MultimemTier",
    ],
)
def test_modes_are_int_enums_of_the_package(name: str):
    """Every C enum is an `IntEnum` of the `uform` module, never a string."""
    mirrored = getattr(uform, name)
    assert issubclass(mirrored, IntEnum)
    assert mirrored.__module__ == "uform"


def test_enum_values_follow_the_header():
    """Members mirror the C values, gaps in the numbering included."""
    assert uform.Status.SHAPE_MISMATCH == 3
    assert uform.Status.IO == 7
    assert uform.Status.INVALID_ARGUMENT == 17
    assert uform.Status.UNINITIALIZED == 12
    assert issubclass(uform.UninitializedError, uform.Error)
    assert uform.Scalar.BF16 == 3
    assert uform.Scalar.E4M3 == 6
    assert uform.Scalar.U1X8 == 22
    assert uform.Modality.AUDIO == 3
    assert uform.Objective.JOINT == 2
    assert uform.MultimemTier.PEER_READ == 1
    assert "UNRECOGNIZED" not in uform.Status.__members__


def test_classes_are_spelled_by_the_library():
    """Class members take the library's own spellings and fixed values, and the unknown class is no member."""
    assert uform.CPUClass.SERIAL == 1
    assert uform.CPUClass.MONOVERSIONED == 2
    assert uform.CPUClass.IBM_POWER10 == 18
    assert uform.CPUClass.INTEL_HASWELL == 3
    assert uform.CPUClass.WASM64_EXTENDED == 20
    assert uform.GPUClass.NVIDIA_HOPPER == 3
    assert uform.GPUClass.APPLE_METAL10 == 12
    assert "UNKNOWN" not in uform.CPUClass.__members__
    assert "UNKNOWN" not in uform.GPUClass.__members__


def test_gpu_devices_list_the_drivable_ones():
    """Every listed device names its vendor and a class, and a build without GPUs lists none."""
    devices = uform.gpu_devices()
    assert isinstance(devices, tuple)
    assert all(isinstance(device.vendor, uform.GPUVendor) and device.memory_bytes > 0 for device in devices)
    if "serial" in uform.capabilities_compiled():
        assert devices == ()


def test_outputs_is_a_flag_set():
    """`Outputs` combines with `|`, one flag per output field of the encode verbs."""
    assert issubclass(uform.Outputs, IntFlag)
    both = uform.Outputs.EMBEDDINGS | uform.Outputs.STATES
    assert int(both) == 3
    assert uform.Outputs.STATES in both
    assert uform.Outputs.TOKEN_VECTORS == 4


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
    """An integer no member holds is an unknown kind, and a bool or a string is no member at all."""
    with pytest.raises(uform.UnknownKindError):
        uform.Model.load_ram(checkpoint, quantization=99)
    with pytest.raises(TypeError):
        uform.Model.load_ram(checkpoint, quantization="bf16")


def test_executor_drains_and_closes(model: uform.Model):
    """An executor drains in either mood and closes on leaving its `with` block, refusing later receivers."""
    with uform.Executor.for_ram(threads=1) as executor:
        executor.drain(uform.Mood.SPINNING)
        executor.drain()
    with pytest.raises(uform.UninitializedError):
        model.on(executor)


def test_executor_is_built_through_its_residency():
    """`Executor()` names no residency, so only the class methods build one."""
    with pytest.raises(TypeError):
        uform.Executor()


def test_a_loaded_model_describes_itself_on_every_receiver(checkpoint: Path):
    """A load from a path answers a model whose description holds its geometry."""
    model = uform.Model.load_ram(checkpoint, quantization=uform.Scalar.BF16)
    description = model.describe()
    assert description.dimensions > 0
    assert description.context > 0
    assert description.memory_usage > 0
    assert description.quantization is uform.Scalar.BF16
    assert isinstance(description.cpu_class, uform.CPUClass)
    assert description.gpu_class is None


def test_a_ceiling_loads_the_best_class_under_it(checkpoint: Path):
    """A serial ceiling packs for serial, and one of a family this CPU does not run falls to serial."""
    serial = uform.Model.load_ram(checkpoint, cpu_class_ceiling=uform.CPUClass.SERIAL)
    assert serial.describe().cpu_class is uform.CPUClass.SERIAL
    foreign = uform.Model.load_ram(checkpoint, cpu_class_ceiling=uform.CPUClass.WASM64_EXTENDED)
    assert foreign.describe().cpu_class is uform.CPUClass.SERIAL
    with pytest.raises(uform.UnknownKindError):
        uform.Model.load_ram(checkpoint, cpu_class_ceiling=0)


def test_load_ram_buffer_matches_the_path(checkpoint: Path):
    """A load from the checkpoint's bytes describes the same geometry as a load from its path."""
    from_path = uform.Model.load_ram(checkpoint).describe()
    from_buffer = uform.Model.load_ram_buffer(checkpoint.read_bytes()).describe()
    assert (from_path.dimensions, from_path.context) == (from_buffer.dimensions, from_buffer.context)


def test_load_refuses_a_quantization_the_blocks_never_run_in(checkpoint: Path):
    """Blocks run in bf16 or e4m3, so f32 is declared but unsupported."""
    with pytest.raises(uform.UnsupportedError):
        uform.Model.load_ram(checkpoint, quantization=uform.Scalar.F32)


def test_a_load_runs_through_an_executor(checkpoint: Path, executor: uform.Executor):
    """A load may run on an executor, and answers the same model."""
    assert uform.Model.load_ram(checkpoint, executor=executor).describe().context > 0


def test_a_stopping_handler_cancels_the_call(checkpoint: Path):
    """A handler answering False stops the load, which raises `CancelledError`, a `uform.Error`."""
    reports: list[tuple[int, int]] = []

    def stop(completed: int, total: int) -> bool:
        reports.append((completed, total))
        return False

    with pytest.raises(uform.CancelledError) as caught:
        uform.Model.load_ram(checkpoint, progress=stop)
    assert isinstance(caught.value, uform.Error)
    assert caught.value.status is uform.Status.CANCELLED
    assert reports
    assert all(0 < completed <= total for completed, total in reports)


def test_a_raising_handler_reports_and_cancels(checkpoint: Path, unraisable: list[object]):
    """A handler that raises goes to `sys.unraisablehook` and stops the load, which raises `CancelledError`."""

    def explode(completed: int, total: int) -> bool:
        raise LookupError("from the handler")

    with pytest.raises(uform.CancelledError):
        uform.Model.load_ram(checkpoint, progress=explode)
    assert len(unraisable) == 1
    assert isinstance(unraisable[0].exc_value, LookupError)


@requires_cpu_only_build
def test_vram_residency_reports_its_absence(checkpoint: Path):
    """A build without GPUs refuses every VRAM verb with `DeviceError`."""
    with pytest.raises(uform.DeviceError):
        uform.Executor.for_vram([0])
    with pytest.raises(uform.DeviceError):
        uform.Model.load_vram(checkpoint, devices=[0])
    with pytest.raises(uform.DeviceError):
        uform.Model.load_vram_buffer(checkpoint.read_bytes())


def test_receivers_answer_alike(model: uform.Model, executor: uform.Executor):
    """The model, `on` and `deferred` run one verb with one answer."""
    direct = model.describe()
    assert model.on(executor).describe().dimensions == direct.dimensions
    call = model.deferred(executor).describe()
    assert isinstance(call, uform.Deferred)
    answer = call.wait()
    assert isinstance(answer, uform.ModelDescription)
    assert answer.context == direct.context
    assert call.wait() is answer, "a final call answers as many times as it is asked"
    assert call.status is uform.Status.SUCCESS


def test_deferred_descriptions_await(model: uform.Model, executor: uform.Executor):
    """`await` polls the call between turns of the event loop, so the coroutine suspends and no thread blocks."""

    async def describe() -> uform.ModelDescription:
        return await model.deferred(executor).describe()

    assert asyncio.run(describe()).dimensions == model.describe().dimensions


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
