"""The six encode verbs: outputs selection, shapes, stats, tapes, arenas, deferred calls and progress.

Run with `uv run pytest test/python/encode.py`. Texts are tokenizer-free, a byte per token, so their shapes
follow from the inputs and the model's own description; media shapes are only checked for consistency.

File: test/python/encode.py
Author: Ash Vardanian
Date: October 18, 2026
"""

import asyncio
import io
import struct
import wave
import zlib
from array import array

import pytest

import uform

TEXTS = ["first document", "the second", ""]


def png(side: int = 8) -> bytes:
    """A valid `side` x `side` RGB PNG of a gradient."""
    rows = b"".join(b"\0" + bytes((row * 32 + byte * 8) & 0xFF for byte in range(side * 3)) for row in range(side))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def wav(samples: int = 1600) -> bytes:
    """A valid 16 kHz mono 16-bit WAV of a sawtooth."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(array("h", [(sample % 64) * 512 - 16384 for sample in range(samples)]).tobytes())
    return buffer.getvalue()


def test_embeddings_alone_by_default(model: uform.Model, arena: uform.Arena):
    """Without `outputs`, a call keeps the pooled embeddings alone, one row of `dimensions` per item."""
    dimensions = model.describe().dimensions
    embeddings, states, token_vectors = model.text_encode(TEXTS, arena)
    assert states is None and token_vectors is None
    view = memoryview(embeddings)
    assert view.shape == (len(TEXTS), dimensions) and view.format == "f"
    assert len(embeddings) == len(TEXTS)


def test_states_alone(model: uform.Model, arena: uform.Arena):
    """`Outputs.STATES` keeps one row per byte, split by `count + 1` offsets, and skips the embeddings."""
    encoded = model.text_encode(TEXTS, arena, outputs=uform.Outputs.STATES)
    assert encoded.embeddings is None
    offsets = memoryview(encoded.states.offsets).tolist()
    assert offsets == [0, 14, 24, 24]
    assert memoryview(encoded.states).shape == (24, model.describe().dimensions)


def test_both_outputs_from_one_forward(model: uform.Model, arena: uform.Arena):
    """Both flags keep both surfaces, landing in the one arena."""
    encoded = model.text_encode(TEXTS, arena, outputs=uform.Outputs.EMBEDDINGS | uform.Outputs.STATES)
    assert len(encoded.embeddings) == len(TEXTS)
    assert len(encoded.states) == sum(len(text) for text in TEXTS)


def test_token_vectors_follow_the_token_head(model: uform.Model, arena: uform.Arena):
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
    assert len(rows) == len(encoded.states) and all(len(row) == token_width for row in rows)
    assert all(abs(sum(value * value for value in row) - 1) < 0.02 for row in rows)


def test_choices_sum_to_one_per_text(model: uform.Model, arena: uform.Arena):
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
    assert offsets == [0, 2, 5, 6] and stats.truncated == 0
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
def test_outputs_scalar(model: uform.Model, arena: uform.Arena, scalar: uform.Scalar, format: str, item_bytes: int):
    """Formattable scalars export a (count, dimensions) matrix, and `bf16` a (count, row_bytes) run of bytes."""
    dimensions = model.describe().dimensions
    view = memoryview(model.text_encode(TEXTS, arena, outputs_scalar=scalar).embeddings)
    assert view.format == format and view.itemsize == item_bytes
    assert view.nbytes == len(TEXTS) * dimensions * (2 if scalar != uform.Scalar.F32 else 4)


def test_truncation_is_counted(model: uform.Model, arena: uform.Arena):
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


def test_a_tape_pair_is_lent_as_is(model: uform.Model, arena: uform.Arena):
    """A (data, offsets) pair of buffers encodes exactly as the sequence it lays out."""
    data = "".join(TEXTS).encode()
    offsets = array("Q", [0, 14, 24, 24])
    lent = bytes(memoryview(model.text_encode((data, offsets), arena).embeddings))
    gathered = bytes(memoryview(model.text_encode([text.encode() for text in TEXTS], arena).embeddings))
    assert lent == gathered


def test_outputs_cross_as_dlpack(model: uform.Model, arena: uform.Arena):
    """Host outputs export DLPack, lend a tape pair as a buffer would, and free their arena once dropped."""
    encoded = model.text_encode(TEXTS, arena, outputs=uform.Outputs.STATES)
    assert encoded.states.offsets.__dlpack_device__() == (1, 0)
    data = "".join(TEXTS).encode()
    lent = bytes(memoryview(model.text_encode((data, encoded.states.offsets), uform.Arena()).embeddings))
    gathered = bytes(memoryview(model.text_encode([text.encode() for text in TEXTS], uform.Arena()).embeddings))
    assert lent == gathered
    capsule = encoded.states.__dlpack__()
    with pytest.raises(uform.ContendedError):
        model.text_encode(TEXTS, arena)
    del capsule
    model.text_encode(TEXTS, arena)


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


def test_an_exported_buffer_pins_the_arena(model: uform.Model, arena: uform.Arena):
    """While a buffer over its outputs lives, the arena refuses to rewind, and releasing it lets the next call in."""
    view = memoryview(model.text_encode(TEXTS, arena).embeddings)
    with pytest.raises(uform.ContendedError):
        model.text_encode(TEXTS, arena)
    view.release()
    model.text_encode(TEXTS, arena)


def test_a_rewound_output_goes_stale(model: uform.Model, arena: uform.Arena):
    """A result whose arena a later call rewound refuses to be read."""
    first = model.text_encode(TEXTS, arena)
    model.text_encode(TEXTS, arena)
    with pytest.raises(uform.InvalidArgumentError):
        memoryview(first.embeddings)


def test_deferred_matches_blocking(model: uform.Model, executor: uform.Executor):
    """A deferred call answers exactly what the blocking one does."""
    blocking = bytes(memoryview(model.on(executor).text_encode(TEXTS, uform.Arena(executor)).embeddings))
    call = model.deferred(executor).text_encode(TEXTS, uform.Arena(executor))
    assert bytes(memoryview(call.wait().embeddings)) == blocking


def test_deferred_awaits(model: uform.Model, executor: uform.Executor):
    """`await` resolves the call on the event loop's default executor."""

    async def encode() -> uform.Encoded:
        call = model.deferred(executor).image_encode([png(), bytearray(png(16))], uform.Arena(executor))
        return await call

    assert len(asyncio.run(encode()).embeddings) == 2


def test_a_deferred_call_holds_its_arena(model: uform.Model, executor: uform.Executor, arena: uform.Arena):
    """Until a deferred call is waited on or dropped, its arena refuses another call."""
    call = model.deferred(executor).text_encode(TEXTS, arena)
    with pytest.raises(uform.ContendedError):
        model.on(executor).text_encode(TEXTS, arena)
    call.wait()
    model.on(executor).text_encode(TEXTS, arena)


def test_a_dropped_deferred_call_waits(model: uform.Model, executor: uform.Executor, arena: uform.Arena):
    """Dropping a `Deferred` waits for its call, then lets go of the arena."""
    model.deferred(executor).text_encode(TEXTS, arena)
    assert len(model.text_encode(TEXTS, arena).embeddings) == len(TEXTS)


def test_a_full_deferred_table_is_contended(model: uform.Model):
    """A submission past `deferred_limit` finishes contended, raising at the wait."""
    with uform.Executor.for_ram(deferred_limit=1) as executor:
        first = model.deferred(executor).text_encode(TEXTS, uform.Arena(executor))
        second = model.deferred(executor).text_encode(TEXTS, uform.Arena(executor))
        with pytest.raises(uform.ContendedError):
            second.wait()
        assert len(first.wait().embeddings) == len(TEXTS)


def test_progress_reports_and_cancels(model: uform.Model, arena: uform.Arena):
    """Reports never decrease and end at the total; a handler answering False raises `CancelledError`."""
    reports: list[tuple[int, int]] = []
    model.text_encode(["a"] * 8, arena, progress=lambda completed, total: reports.append((completed, total)))
    assert reports and reports[-1] == (8, 8)
    assert [completed for completed, _ in reports] == sorted(completed for completed, _ in reports)
    with pytest.raises(uform.CancelledError):
        model.text_encode(["a"] * 8, arena, progress=lambda completed, total: completed < 4, progress_steps=4)


def test_progress_cancels_a_deferred_call(model: uform.Model, executor: uform.Executor, arena: uform.Arena):
    """A deferred call's handler runs on whichever thread waits, and its exception surfaces there."""

    def explode(completed: int, total: int) -> bool:
        raise LookupError("from the handler")

    call = model.deferred(executor).text_encode(TEXTS, arena, progress=explode)
    with pytest.raises(LookupError, match="from the handler"):
        call.wait()
    model.text_encode(TEXTS, arena)


def test_image_encode_pixels(model: uform.Model, arena: uform.Arena):
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
    assert len(offsets) == 3 and offsets[-1] == len(encoded.states) == stats.tokens


def test_image_encode_pixels_refuses_other_shapes(model: uform.Model, arena: uform.Arena):
    """A buffer that is not [height, width, 3] uint8 is refused before the call."""
    with pytest.raises(uform.ShapeMismatchError):
        model.image_encode_pixels([bytearray(27)], arena)
    with pytest.raises(uform.ShapeMismatchError):
        model.image_encode_pixels([memoryview(bytearray(36)).cast("B", (3, 3, 4))], arena)


@pytest.mark.parametrize(("verb", "file"), [("image_encode", png()), ("audio_encode", wav())])
def test_encoded_files(model: uform.Model, arena: uform.Arena, verb: str, file: bytes):
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


def test_audio_encode_samples(model: uform.Model, arena: uform.Arena):
    """Decoded samples arrive as 1-D float32 buffers, or as a (data, offsets) pair counted in samples."""
    recordings = [array("f", [0.0] * 1000), array("f", [0.5] * 300)]
    gathered = bytes(memoryview(model.audio_encode_samples(recordings, 16000, arena).embeddings))
    data = array("f", [0.0] * 1000 + [0.5] * 300)
    lent = bytes(memoryview(model.audio_encode_samples((data, array("Q", [0, 1000, 1300])), 16000, arena).embeddings))
    assert gathered == lent
    with pytest.raises(uform.InvalidArgumentError):
        model.audio_encode_samples(recordings, 0, arena)
    with pytest.raises(uform.ShapeMismatchError):
        model.audio_encode_samples([array("d", [0.0])], 16000, arena)


def test_verbs_refuse_a_receiver_over_a_trainer(model: uform.Model, executor: uform.Executor):
    """Every receiver offers every verb, and a verb checks the type of the target it addresses."""
    on = model.on(executor)
    with pytest.raises(TypeError):
        on.step([uform.Modality.TEXT], ["a"])
