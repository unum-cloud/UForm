"""The six encode verbs: outputs selection, shapes, stats, tapes, arenas, deferred calls and progress.

Texts are tokenizer-free, a byte per token, so their shapes follow from the inputs and the model's own
description; media shapes are only checked for consistency.

Run:
    uv run pytest test/python/encode.py

File: test/python/encode.py
Author: Ash Vardanian
Date: October 18, 2026
"""

import asyncio
import ctypes
import io
import wave
from array import array
from typing import NoReturn

import pytest
from base import numpy_available, png, pyarrow_available

import uform

if numpy_available:
    import numpy
if pyarrow_available:
    import pyarrow

TEXTS = ["first document", "the second", ""]


class DLDevice(ctypes.Structure):
    """DLPack's `DLDevice`, to hand-build a capsule no installed library would export here."""

    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    """DLPack's `DLDataType`."""

    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    """DLPack's `DLTensor`."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensorVersioned(ctypes.Structure):
    """DLPack's `DLManagedTensorVersioned`, with no deleter, as the test owns it."""

    _fields_ = [
        ("version", ctypes.c_uint32 * 2),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", DLTensor),
    ]


class DLPackExporter:
    """An object exporting `tensor` through a versioned DLPack capsule, as a GPU library would."""

    def __init__(self, tensor: DLManagedTensorVersioned):
        """Keeps `tensor` alive as long as the exporter."""
        self.tensor = tensor

    def __dlpack__(self, **_: object) -> object:
        """A capsule naming the tensor in place, whatever the consumer asks for."""
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.restype = ctypes.py_object
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        return capsule_new(ctypes.addressof(self.tensor), b"dltensor_versioned", None)


DEVICE_BYTES = (ctypes.c_uint8 * 8)()
"""Host bytes a hand-built capsule claims live on a GPU; the binding must refuse them before reading."""


def wav(samples: int = 1600) -> bytes:
    """A valid 16 kHz mono 16-bit WAV of a sawtooth."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(array("h", [(sample % 64) * 512 - 16384 for sample in range(samples)]).tobytes())
    return buffer.getvalue()


def test_embeddings_are_kept_alone_by_default(model: uform.Model, arena: uform.Arena):
    """Without `outputs`, a call keeps the pooled embeddings alone, one row of `dimensions` per item."""
    dimensions = model.describe().dimensions
    embeddings, states, token_vectors = model.text_encode(TEXTS, arena)
    assert states is None
    assert token_vectors is None
    view = memoryview(embeddings)
    assert view.shape == (len(TEXTS), dimensions)
    assert view.format == "f"
    assert len(embeddings) == len(TEXTS)


def test_states_alone_skip_the_embeddings(model: uform.Model, arena: uform.Arena):
    """`Outputs.STATES` keeps one row per byte, split by `count + 1` offsets, and skips the embeddings."""
    encoded = model.text_encode(TEXTS, arena, outputs=uform.Outputs.STATES)
    assert encoded.embeddings is None
    offsets = memoryview(encoded.states.offsets).tolist()
    assert offsets == [0, 14, 24, 24]
    assert memoryview(encoded.states).shape == (24, model.describe().dimensions)


def test_both_outputs_land_from_one_forward(model: uform.Model, arena: uform.Arena):
    """Both flags keep both surfaces, landing in the one arena."""
    encoded = model.text_encode(TEXTS, arena, outputs=uform.Outputs.EMBEDDINGS | uform.Outputs.STATES)
    assert len(encoded.embeddings) == len(TEXTS)
    assert len(encoded.states) == sum(len(text) for text in TEXTS)


def test_token_vectors_follow_the_states_or_are_refused(model: uform.Model, arena: uform.Arena):
    """Token vectors are unit rows of `token_width` on the states' offsets, and a model without the head refuses them."""
    token_width = model.describe().token_width
    outputs = uform.Outputs.STATES | uform.Outputs.TOKEN_VECTORS
    if not token_width:
        with pytest.raises(uform.UnsupportedError):
            model.text_encode(TEXTS, arena, outputs=outputs)
        return
    encoded = model.text_encode(TEXTS, arena, outputs=outputs)
    assert memoryview(encoded.token_vectors.offsets).tolist() == memoryview(encoded.states.offsets).tolist()
    rows = memoryview(encoded.token_vectors).tolist()
    assert len(rows) == len(encoded.states)
    assert all(len(row) == token_width for row in rows)
    assert all(abs(sum(value * value for value in row) - 1) < 0.02 for row in rows)


def test_choices_sum_to_one_or_are_refused(model: uform.Model, arena: uform.Arena):
    """Each text's probabilities follow its choices and sum to one; a model without the byte decoder refuses."""
    texts, choices = ["keep or drop? ", "pick a neighbour: ", "stop? "], ["yn", "ABC", "s"]
    groups = model.describe().choice_temperature_groups
    if not groups:
        with pytest.raises(uform.UnsupportedError):
            model.text_choose(texts, choices, arena)
        return
    stats = uform.TextEncodeStats()
    probabilities = model.text_choose(texts, choices, arena, temperature_groups=[0, 0, 0], stats=stats)
    offsets, values = memoryview(probabilities.offsets).tolist(), memoryview(probabilities).tolist()
    assert offsets == [0, 2, 5, 6]
    assert stats.truncated == 0
    assert all(abs(sum(values[begin:end]) - 1) < 1e-3 for begin, end in zip(offsets, offsets[1:]))
    with pytest.raises(uform.InvalidArgumentError):
        model.text_choose(texts, choices, arena, temperature_groups=[0, groups, 0])


def test_choices_need_one_entry_per_text(model: uform.Model, arena: uform.Arena):
    """Choices and groups pair with the texts, so a count of either that differs is refused before the call."""
    with pytest.raises(uform.ShapeMismatchError):
        model.text_choose(["one", "two"], ["yn"], arena)
    with pytest.raises(uform.ShapeMismatchError):
        model.text_choose(["one"], ["yn"], arena, temperature_groups=[0, 0])


def test_naming_no_output_is_refused(model: uform.Model, arena: uform.Arena):
    """An empty flag set passes neither pointer, which the library refuses."""
    with pytest.raises(uform.InvalidArgumentError):
        model.text_encode(TEXTS, arena, outputs=uform.Outputs(0))
    with pytest.raises(uform.InvalidArgumentError):
        model.text_encode(TEXTS, arena, outputs=8)


@pytest.mark.parametrize(
    ("scalar", "format", "item_bytes"),
    [(uform.Scalar.F32, "f", 4), (uform.Scalar.F16, "e", 2), (uform.Scalar.BF16, "B", 1)],
)
def test_outputs_scalar_picks_the_exported_format(
    model: uform.Model, arena: uform.Arena, scalar: uform.Scalar, format: str, item_bytes: int
):
    """Formattable scalars export a (count, dimensions) matrix, and `bf16` a (count, row_bytes) run of bytes."""
    dimensions = model.describe().dimensions
    view = memoryview(model.text_encode(TEXTS, arena, outputs_scalar=scalar).embeddings)
    assert view.format == format
    assert view.itemsize == item_bytes
    assert view.nbytes == len(TEXTS) * dimensions * (2 if scalar != uform.Scalar.F32 else 4)


def test_texts_past_the_context_are_truncated(model: uform.Model, arena: uform.Arena):
    """A text past the context keeps its first `context` tokens, and the stats count it as truncated."""
    context = model.describe().context
    stats = uform.TextEncodeStats()
    encoded = model.text_encode(["x" * (context + 9), "y"], arena, outputs=uform.Outputs.STATES, stats=stats)
    assert (stats.truncated, stats.tokens) == (1, context + 1)
    assert memoryview(encoded.states.offsets).tolist() == [0, context, context + 1]


def test_stats_type_is_checked(model: uform.Model, arena: uform.Arena):
    """Each verb fills its own report type, and a stranger is refused before the call."""
    with pytest.raises(TypeError):
        model.text_encode(TEXTS, arena, stats=uform.ImageEncodeStats())
    assert uform.ImageEncodeStats().tokens == 0


def test_a_tape_pair_is_read_in_place(model: uform.Model, arena: uform.Arena):
    """A (data, offsets) pair of buffers encodes exactly as the sequence it lays out."""
    data = "".join(TEXTS).encode()
    offsets = array("Q", [0, 14, 24, 24])
    packed = bytes(memoryview(model.text_encode((data, offsets), arena).embeddings))
    gathered = bytes(memoryview(model.text_encode([text.encode() for text in TEXTS], arena).embeddings))
    assert packed == gathered


def test_outputs_cross_as_dlpack(model: uform.Model, arena: uform.Arena):
    """Host outputs export DLPack, join a tape pair as a buffer would, and keep their arena alive while exported."""
    encoded = model.text_encode(TEXTS, arena, outputs=uform.Outputs.STATES)
    assert encoded.states.offsets.__dlpack_device__() == (1, 0)
    data = "".join(TEXTS).encode()
    packed = bytes(memoryview(model.text_encode((data, encoded.states.offsets), uform.Arena()).embeddings))
    gathered = bytes(memoryview(model.text_encode([text.encode() for text in TEXTS], uform.Arena()).embeddings))
    assert packed == gathered
    capsule = encoded.states.__dlpack__()
    del encoded, arena
    del capsule


def test_an_arena_with_an_export_open_refuses_calls(model: uform.Model, arena: uform.Arena):
    """A DLPack capsule over an output keeps its arena from rewinding until the consumer lets go."""
    capsule = model.text_encode(TEXTS, arena).embeddings.__dlpack__()
    with pytest.raises(uform.ContendedError):
        model.text_encode(TEXTS, arena)
    del capsule
    model.text_encode(TEXTS, arena)


def test_a_live_buffer_view_refuses_reuse_of_its_arena(model: uform.Model, arena: uform.Arena):
    """A memoryview counts as an export, so reusing its arena raises ContendedError until the view is released."""
    view = memoryview(model.text_encode(TEXTS, arena).embeddings)
    with pytest.raises(uform.ContendedError):
        model.text_encode(TEXTS, arena)
    view.release()
    model.text_encode(TEXTS, arena)


def test_a_result_of_an_earlier_call_is_stale(model: uform.Model, arena: uform.Arena):
    """Once a later call rewinds the arena, an earlier result refuses every read of its memory."""
    earlier = model.text_encode(TEXTS, arena).embeddings
    later = model.text_encode(TEXTS, arena).embeddings
    with pytest.raises(ValueError):  # noqa: PT011 - the stale result raises the built-in alone
        memoryview(earlier)
    with pytest.raises(ValueError):  # noqa: PT011 - the stale result raises the built-in alone
        earlier.__dlpack__()
    assert len(earlier) == len(later)
    assert bytes(memoryview(later))


def test_a_deferred_call_naming_the_arena_makes_earlier_results_stale(
    model: uform.Model, executor: uform.Executor, arena: uform.Arena
):
    """A deferred call takes its arena as it is submitted, so an earlier result over the arena reads as stale at once."""
    earlier = model.text_encode(TEXTS, arena).embeddings
    pending = model.deferred(executor).text_encode(TEXTS[:1], arena)
    with pytest.raises(ValueError):  # noqa: PT011 - the stale result raises the built-in alone
        memoryview(earlier)
    assert len(memoryview(pending.wait().embeddings)) == 1


def test_calls_without_an_arena_own_a_fresh_one(model: uform.Model, executor: uform.Executor):
    """A call naming no arena writes into a fresh one its result owns, so results of separate calls stay valid."""
    first = model.text_encode(TEXTS).embeddings
    second = model.on(executor).text_encode(TEXTS, outputs=uform.Outputs.EMBEDDINGS).embeddings
    third = model.deferred(executor).text_encode(TEXTS).wait().embeddings
    assert bytes(memoryview(first)) == bytes(memoryview(second)) == bytes(memoryview(third))


def test_closing_an_executor_stops_its_verbs_and_keeps_results(model: uform.Model):
    """Leaving an executor's `with` block stops its verbs, and results over its arenas stay readable."""
    with uform.Executor.for_ram() as pool:
        embeddings = model.on(pool).text_encode(TEXTS).embeddings
    assert bytes(memoryview(embeddings))
    with pytest.raises(uform.UninitializedError):
        model.on(pool).text_encode(TEXTS)


def test_a_tape_pair_past_its_data_is_refused(model: uform.Model, arena: uform.Arena):
    """Offsets ending past the data would read out of bounds, so the binding refuses them."""
    with pytest.raises(uform.ShapeMismatchError):
        model.text_encode((b"abc", array("Q", [0, 9])), arena)


def test_one_text_is_not_a_batch(model: uform.Model, arena: uform.Arena):
    """A lone `str` or `bytes` would split into characters, so it is refused."""
    with pytest.raises(TypeError):
        model.text_encode("one text", arena)
    with pytest.raises(TypeError):
        model.image_encode(b"one file", arena)


def test_an_arena_serves_one_call_at_a_time(model: uform.Model, executor: uform.Executor, arena: uform.Arena):
    """The library refuses a call naming an arena a running call holds, and the binding only forwards the refusal."""
    refused: list[uform.Error] = []

    def contend(completed: int, total: int) -> bool:
        try:
            model.text_encode(TEXTS, arena)
        except uform.ContendedError as error:
            refused.append(error)
        return True

    assert len(model.on(executor).text_encode(TEXTS, arena, progress=contend).embeddings) == len(TEXTS)
    assert refused
    assert all(error.status is uform.Status.CONTENDED for error in refused)
    model.text_encode(TEXTS, arena)


def test_deferred_matches_blocking(model: uform.Model, executor: uform.Executor):
    """A deferred call answers exactly what the blocking one does."""
    blocking = bytes(memoryview(model.on(executor).text_encode(TEXTS, uform.Arena(executor)).embeddings))
    call = model.deferred(executor).text_encode(TEXTS, uform.Arena(executor))
    assert bytes(memoryview(call.wait().embeddings)) == blocking


def test_deferred_encodings_await(model: uform.Model, executor: uform.Executor):
    """`await` polls the call between turns of the event loop, which stays free for other coroutines."""

    async def encode() -> uform.Encoded:
        call = model.deferred(executor).image_encode([png(), bytearray(png(16))], uform.Arena(executor))
        return await call

    assert len(asyncio.run(encode()).embeddings) == 2


def test_a_final_deferred_call_gives_back_its_arena(model: uform.Model, executor: uform.Executor, arena: uform.Arena):
    """Once a deferred call is final, its executor and its arena serve the next call at once."""
    call = model.deferred(executor).text_encode(TEXTS, arena)
    call.wait()
    model.on(executor).text_encode(TEXTS, arena)


def test_a_dropped_deferred_call_waits(model: uform.Model, executor: uform.Executor, arena: uform.Arena):
    """Dropping a `Deferred` waits for its call, then lets go of the arena."""
    model.deferred(executor).text_encode(TEXTS, arena)
    assert len(model.text_encode(TEXTS, arena).embeddings) == len(TEXTS)


def test_progress_reports_and_cancels(model: uform.Model, arena: uform.Arena):
    """Reports never decrease and end at the total; a handler answering False raises `CancelledError`."""
    reports: list[tuple[int, int]] = []
    model.text_encode(["a"] * 8, arena, progress=lambda completed, total: reports.append((completed, total)))
    assert reports
    assert reports[-1] == (8, 8)
    assert [completed for completed, _ in reports] == sorted(completed for completed, _ in reports)
    with pytest.raises(uform.CancelledError):
        model.text_encode(["a"] * 8, arena, progress=lambda completed, total: completed < 4, progress_steps=4)


def test_progress_cancels_a_deferred_call(
    model: uform.Model, executor: uform.Executor, arena: uform.Arena, unraisable: list[object]
):
    """A deferred call's handler runs on the executor's thread and reports its exception, then cancels.

    One that raises goes to `sys.unraisablehook`, and the call raises `CancelledError` at the wait.
    """

    def explode(completed: int, total: int) -> bool:
        raise LookupError("from the handler")

    call = model.deferred(executor).text_encode(TEXTS, arena, progress=explode)
    with pytest.raises(uform.CancelledError):
        call.wait()
    assert len(unraisable) == 1
    assert isinstance(unraisable[0].exc_value, LookupError)
    model.text_encode(TEXTS, arena)


def test_pixels_encode_with_their_tokens_split_by_offsets(model: uform.Model, arena: uform.Arena):
    """Pixels arrive as [height, width, 3] uint8 buffers, and their tokens split by offsets."""
    images = [
        memoryview(bytearray(17 * 9 * 3)).cast("B", (17, 9, 3)),
        memoryview(bytearray(8 * 8 * 3)).cast("B", (8, 8, 3)),
    ]
    stats = uform.ImageEncodeStats()
    encoded = model.image_encode_pixels(
        images, arena, outputs=uform.Outputs.EMBEDDINGS | uform.Outputs.STATES, stats=stats
    )
    assert len(encoded.embeddings) == 2
    offsets = memoryview(encoded.states.offsets).tolist()
    assert len(offsets) == 3
    assert offsets[-1] == len(encoded.states) == stats.tokens


def test_image_encode_pixels_refuses_other_shapes(model: uform.Model, arena: uform.Arena):
    """A buffer that is not [height, width, 3] uint8 is refused before the call."""
    with pytest.raises(uform.ShapeMismatchError):
        model.image_encode_pixels([bytearray(27)], arena)
    with pytest.raises(uform.ShapeMismatchError):
        model.image_encode_pixels([memoryview(bytearray(36)).cast("B", (3, 3, 4))], arena)


@pytest.mark.parametrize(("verb", "file"), [("image_encode", png()), ("audio_encode", wav())])
def test_file_verbs_take_any_bytes_like_entry(model: uform.Model, arena: uform.Arena, verb: str, file: bytes):
    """Every file verb takes bytes-like entries and lands one embedding per file."""
    files = [file, bytearray(file), memoryview(file)]
    encoded = getattr(model, verb)(files, arena, outputs=uform.Outputs.EMBEDDINGS | uform.Outputs.STATES)
    assert len(encoded.embeddings) == len(files)
    offsets = memoryview(encoded.states.offsets).tolist()
    assert offsets[-1] == len(encoded.states)


def test_an_undecodable_file_fails_whole(model: uform.Model, arena: uform.Arena):
    """A file the engine cannot decode fails whole as malformed; the stub decodes nothing, so it encodes any bytes."""
    try:
        encoded = model.video_encode([b"\x00" * 100], arena)
    except uform.MalformedError:
        return
    assert len(encoded.embeddings) == 1


def test_samples_encode_from_buffers_or_a_tape_pair(model: uform.Model, arena: uform.Arena):
    """Decoded samples arrive as 1-D float32 buffers, or as a (data, offsets) pair counted in samples."""
    recordings = [array("f", [0.0] * 1000), array("f", [0.5] * 300)]
    gathered = bytes(memoryview(model.audio_encode_samples(recordings, 16000, arena).embeddings))
    data = array("f", [0.0] * 1000 + [0.5] * 300)
    packed = bytes(memoryview(model.audio_encode_samples((data, array("Q", [0, 1000, 1300])), 16000, arena).embeddings))
    assert gathered == packed
    with pytest.raises(uform.InvalidArgumentError):
        model.audio_encode_samples(recordings, 0, arena)
    with pytest.raises(uform.ShapeMismatchError):
        model.audio_encode_samples([array("d", [0.0])], 16000, arena)


def test_a_receiver_offers_its_targets_verbs(model: uform.Model, executor: uform.Executor):
    """One receiver type serves every target, binding the verbs of the target it holds and no others."""
    on = model.on(executor)
    assert on.describe().dimensions == model.describe().dimensions
    assert isinstance(model.deferred(executor).text_encode, type(model.text_encode))
    with pytest.raises(AttributeError):
        on.step([uform.Modality.TEXT], ["a"])


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
@pytest.mark.parametrize("arrow_type", ["string", "large_string", "binary", "large_binary"])
def test_arrow_strings_encode_as_their_sequence(model: uform.Model, arena: uform.Arena, arrow_type: str):
    """An Arrow array of strings or binaries encodes exactly as the Python sequence it holds."""
    batch = pyarrow.array([text.encode() for text in TEXTS], getattr(pyarrow, arrow_type)())
    packed = bytes(memoryview(model.text_encode(batch, arena).embeddings))
    gathered = bytes(memoryview(model.text_encode(TEXTS, uform.Arena()).embeddings))
    assert packed == gathered


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_a_sliced_arrow_array_reads_its_own_entries(model: uform.Model, arena: uform.Arena):
    """An Arrow slice starts at its own offset, so it encodes as the entries it shows."""
    batch = pyarrow.array(["skipped", *TEXTS], pyarrow.large_string()).slice(1)
    packed = bytes(memoryview(model.text_encode(batch, arena).embeddings))
    assert packed == bytes(memoryview(model.text_encode(TEXTS, uform.Arena()).embeddings))


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_an_arrow_stream_of_chunks_encodes_as_one_batch(model: uform.Model, arena: uform.Arena):
    """A chunked Arrow stream gathers its chunks into one batch, in order."""
    chunked = pyarrow.chunked_array([TEXTS[:1], TEXTS[1:]], pyarrow.string())
    packed = bytes(memoryview(model.text_encode(chunked, arena).embeddings))
    assert packed == bytes(memoryview(model.text_encode(TEXTS, uform.Arena()).embeddings))


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_arrow_nulls_are_refused(model: uform.Model, arena: uform.Arena):
    """A null entry has no text to encode, so the binding refuses the batch."""
    with pytest.raises(uform.InvalidArgumentError):
        model.text_encode(pyarrow.array(["text", None]), arena)


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_arrow_lists_of_float32_encode_as_samples(model: uform.Model, arena: uform.Arena):
    """An Arrow list of float32 samples encodes as the buffers it lays out."""
    recordings = [[0.25 * (sample % 4) for sample in range(1600)], [0.5] * 800]
    batch = pyarrow.array(recordings, pyarrow.list_(pyarrow.float32()))
    packed = bytes(memoryview(model.audio_encode_samples(batch, 16000, arena).embeddings))
    gathered = [array("f", recording) for recording in recordings]
    assert packed == bytes(memoryview(model.audio_encode_samples(gathered, 16000, uform.Arena()).embeddings))


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
@pytest.mark.parametrize("arrow_type", ["string", "large_string"])
def test_arrow_strings_are_read_in_place(model: uform.Model, arena: uform.Arena, arrow_type: str):
    """An Arrow array over an arena's own output is refused by the call rewinding it, so its bytes were never copied."""
    embeddings = model.text_encode(TEXTS, arena).embeddings
    with memoryview(embeddings) as view:
        anchor = ctypes.c_char.from_buffer(view.cast("B"))
        address, size = ctypes.addressof(anchor), view.nbytes
        del anchor
    data = pyarrow.foreign_buffer(address, size)  # holds no export, so only the input check can refuse
    width = 4 if arrow_type == "string" else 8
    offsets = pyarrow.py_buffer((0).to_bytes(width, "little") + (4).to_bytes(width, "little"))
    batch = pyarrow.Array.from_buffers(getattr(pyarrow, arrow_type)(), 1, [None, offsets, data])
    with pytest.raises(uform.InvalidArgumentError):
        model.text_encode(batch, arena)


class ArrowOnly:
    """Exports a PyArrow array through `__arrow_c_array__` alone, as a producer without DLPack does."""

    def __init__(self, array: object) -> None:
        """Keeps `array` alive as long as the wrapper."""
        self.array = array

    def __arrow_c_array__(self, requested_schema: object | None = None) -> tuple[object, object]:
        """The schema and array capsules of the wrapped array."""
        return self.array.__arrow_c_array__(requested_schema)


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_a_tape_pair_takes_arrow_offsets(model: uform.Model, arena: uform.Arena):
    """A pair's offsets cross through Arrow when the producer exports nothing else, as through the buffer protocol."""
    data = "".join(TEXTS).encode()
    offsets = ArrowOnly(pyarrow.array([0, 14, 24, 24], pyarrow.uint64()))
    packed = bytes(memoryview(model.text_encode((data, offsets), arena).embeddings))
    assert packed == bytes(memoryview(model.text_encode((data, array("Q", [0, 14, 24, 24])), uform.Arena()).embeddings))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_outputs_round_trip_through_numpy_dlpack(model: uform.Model, arena: uform.Arena):
    """NumPy reads an output in place through DLPack, as the buffer protocol shows it."""
    encoded = model.text_encode(TEXTS, arena, outputs=uform.Outputs.EMBEDDINGS | uform.Outputs.STATES)
    embeddings = numpy.from_dlpack(encoded.embeddings)
    assert embeddings.shape == (len(TEXTS), model.describe().dimensions)
    assert embeddings.tobytes() == bytes(memoryview(encoded.embeddings))
    assert numpy.from_dlpack(encoded.states.offsets).tolist() == memoryview(encoded.states.offsets).tolist()
    assert not hasattr(encoded.embeddings, "__cuda_array_interface__")


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_numpy_inputs_cross_dlpack_even_read_only(model: uform.Model, arena: uform.Arena):
    """A NumPy pair crosses through its versioned DLPack capsule, a read-only one included."""
    data = numpy.frombuffer("".join(TEXTS).encode(), dtype=numpy.uint8)
    offsets = numpy.array([0, 14, 24, 24], dtype=numpy.uint64)
    assert not data.flags.writeable
    packed = bytes(memoryview(model.text_encode((data, offsets), arena).embeddings))
    assert packed == bytes(memoryview(model.text_encode(TEXTS, uform.Arena()).embeddings))


def test_dlpack_stream_argument_is_checked(model: uform.Model, arena: uform.Arena):
    """Host outputs take no stream but None or -1, and a stream that is no integer is a type error."""
    embeddings = model.text_encode(TEXTS, arena).embeddings
    assert embeddings.__dlpack__(stream=None) is not None
    assert embeddings.__dlpack__(stream=-1) is not None
    with pytest.raises(ValueError) as refused:  # noqa: PT011 - the standard names only the built-in
        embeddings.__dlpack__(stream=1)
    assert refused.type is ValueError
    with pytest.raises(TypeError):
        embeddings.__dlpack__(stream="default")
    with pytest.raises(TypeError):
        embeddings.__dlpack__(stream=True)
    with pytest.raises(BufferError):
        embeddings.__dlpack__(dl_device=(2, 0))


def test_device_memory_is_refused_on_host_arenas(model: uform.Model, arena: uform.Arena):
    """A DLPack tensor in CUDA memory cannot feed a call on host pages, which would read it on the host."""
    tensor = DLManagedTensorVersioned()
    shape = ctypes.c_int64(len(DEVICE_BYTES))
    tensor.version[0] = 1
    tensor.dl_tensor.data = ctypes.addressof(DEVICE_BYTES)
    tensor.dl_tensor.device = DLDevice(2, 0)  # kDLCUDA
    tensor.dl_tensor.ndim = 1
    tensor.dl_tensor.dtype = DLDataType(1, 8, 1)  # uint8
    tensor.dl_tensor.shape = ctypes.pointer(shape)
    exporter = DLPackExporter(tensor)
    with pytest.raises(uform.DeviceError):
        model.text_encode((exporter, array("Q", [0, len(DEVICE_BYTES)])), arena)


class RefusingExporter(bytearray):
    """Bytes whose DLPack export refuses with `BufferError`, as NumPy before 2.1 does for read-only arrays."""

    def __dlpack__(self, **_: object) -> NoReturn:
        """Refuses every export."""
        raise BufferError("read-only")


def test_a_refused_dlpack_export_falls_back_to_the_buffer_protocol(model: uform.Model, arena: uform.Arena):
    """A producer refusing its DLPack export with `BufferError` is read through the buffer protocol instead."""
    data = RefusingExporter("".join(TEXTS).encode())
    packed = bytes(memoryview(model.text_encode((data, array("Q", [0, 14, 24, 24])), arena).embeddings))
    assert packed == bytes(memoryview(model.text_encode(TEXTS, uform.Arena()).embeddings))


def test_metal_tensors_are_refused(model: uform.Model, arena: uform.Arena):
    """No Metal exchange is supported yet, so a DLPack tensor on a Metal device is refused as unsupported."""
    shape = ctypes.c_int64(len(DEVICE_BYTES))
    tensor = DLManagedTensorVersioned()
    tensor.version[0] = 1
    tensor.dl_tensor.data = ctypes.addressof(DEVICE_BYTES)
    tensor.dl_tensor.device = DLDevice(8, 0)  # kDLMetal
    tensor.dl_tensor.ndim = 1
    tensor.dl_tensor.dtype = DLDataType(1, 8, 1)  # uint8
    tensor.dl_tensor.shape = ctypes.pointer(shape)
    with pytest.raises(uform.UnsupportedError):
        model.text_encode((DLPackExporter(tensor), array("Q", [0, len(DEVICE_BYTES)])), arena)


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_outputs_round_trip_through_arrow(model: uform.Model, arena: uform.Arena):
    """Rows export as fixed-size lists, ragged rows as large lists of them, and offsets as uint64, all in place."""
    dimensions = model.describe().dimensions
    encoded = model.text_encode(TEXTS, arena, outputs=uform.Outputs.EMBEDDINGS | uform.Outputs.STATES)
    embeddings = pyarrow.array(encoded.embeddings)
    assert (embeddings.type.list_size, embeddings.type.value_type) == (dimensions, pyarrow.float32())
    assert embeddings.values.buffers()[1].to_pybytes() == bytes(memoryview(encoded.embeddings))
    states = pyarrow.chunked_array(encoded.states)
    assert pyarrow.types.is_large_list(states.type)
    assert states.type.value_type.list_size == memoryview(encoded.states).shape[1]
    assert [len(item) for item in states.chunk(0)] == [
        end - start
        for start, end in zip(memoryview(encoded.states.offsets)[:-1], memoryview(encoded.states.offsets)[1:])
    ]
    assert pyarrow.array(encoded.states.offsets).type == pyarrow.uint64()
    schema, device_array = encoded.embeddings.__arrow_c_device_array__()
    imported = pyarrow.Array._import_from_c_device_capsule(schema, device_array)
    assert imported.equals(embeddings)
    with pytest.raises(uform.ContendedError):
        model.text_encode(TEXTS, arena)
    del embeddings, states, imported
    model.text_encode(TEXTS, arena)


@pytest.mark.skipif(not pyarrow_available, reason="PyArrow is not installed")
def test_bf16_rows_have_no_arrow_type(model: uform.Model, arena: uform.Arena):
    """Rows of bf16 cross DLPack and buffers as their bytes, but Arrow names no such type, so it is refused."""
    embeddings = model.text_encode(TEXTS, arena, outputs_scalar=uform.Scalar.BF16).embeddings
    with pytest.raises(BufferError):
        pyarrow.array(embeddings)
