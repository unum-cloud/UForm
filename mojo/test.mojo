"""The Mojo binding's tests, registered in `main` since Mojo has no test discovery.

The pure-logic group needs no library. The rest call the library and are skipped when it is not on
the loader path, and the model checks also need the tiny checkpoint `UFORM_TEST_CHECKPOINT` names,
which `uform_tiny_checkpoint` writes; without one they are skipped.

File: mojo/test.mojo
Author: Ash Vardanian
Date: November 8, 2026
"""

from std.ffi import c_int
from std.math import sqrt
from std.os import getenv
from std.runtime._asyncrt import create_raising_task
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from uform import (
    Arena,
    Continuation,
    CPUClass,
    EncodeConfig,
    EncodeStats,
    Error,
    Executor,
    GPUClass,
    GPUDevice,
    GPUVendor,
    LateInteractionSide,
    Library,
    Mesh,
    Modality,
    Model,
    Mood,
    MultimemTier,
    Negatives,
    Objective,
    Outputs,
    PairLossForm,
    Progress,
    RamOptions,
    Residency,
    ScalarKind,
    Status,
    TapeView,
    TextChooseConfig,
    Trainer,
    TrainerInitVramConfig,
)
from uform.types import NULL_ADDRESS


# region Shared Testing
def _skipped(reason: String):
    """Announces a skipped test, so a run shows what it left out rather than passing in silence."""
    print("skipped:", reason)


def _split_mix_bits(value: UInt64) -> UInt64:
    """SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits."""
    var mixed = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EB
    return mixed ^ (mixed >> 31)


def _stream_key(seed: UInt64, name: String, index: UInt64 = 0) -> UInt64:
    """The `index`-th stream of the family `name` under `seed`, identical in C, C++, Rust and Python."""
    var hashed: UInt64 = 0xCBF29CE484222325
    for byte in name.as_bytes():
        hashed = (hashed ^ UInt64(byte)) * 0x100000001B3
    return _split_mix_bits(_split_mix_bits(seed ^ hashed) + index)


# endregion Shared Testing


comptime library_missing_k = "the shared library is not on the loader path"
comptime checkpoint_missing_k = "needs UFORM_TEST_CHECKPOINT"

# region Pure Logic


def test_outputs_combine_as_flags() raises:
    """`Outputs` combines with `|`, and each flag is contained only where it was named."""
    var both = Outputs.EMBEDDINGS | Outputs.STATES
    assert_true(both.contains(Outputs.EMBEDDINGS) and both.contains(Outputs.STATES))
    assert_false(both.contains(Outputs.TOKEN_VECTORS))


def test_scalar_kinds_follow_their_element_types() raises:
    """Each Mojo element type maps to the kind that stores it, and the rest to `UNKNOWN`."""
    assert_true(ScalarKind.of[DType.float32]() == ScalarKind.F32)
    assert_true(ScalarKind.of[DType.bfloat16]() == ScalarKind.BF16)
    assert_true(ScalarKind.of[DType.uint8]() == ScalarKind.U8)
    assert_true(ScalarKind.of[DType.uint16]() == ScalarKind.UNKNOWN)


def test_tapes_refuse_offsets_past_their_elements() raises:
    """A tape whose last offset ends past its elements is refused before any call reads it."""
    var data: List[UInt8] = [1, 2, 3]
    var offsets: List[UInt] = [0, 2, 4]
    var refused = Status.SUCCESS
    try:
        _ = TapeView.of(data, offsets)
    except error:
        refused = error.status
    assert_true(refused == Status.SHAPE_MISMATCH, "offsets past the elements must be refused")
    var fitting: List[UInt] = [0, 1, 3]
    assert_equal(len(TapeView.of(data, fitting)), 2)


# endregion Pure Logic

# region Engine


def _library_is_present() raises -> Bool:
    """Whether the shared library is on the loader path; any other failure to load it raises rather than skips."""
    try:
        _ = Library()
        return True
    except error:
        if error.status != Status.LIBRARY_MISSING:
            raise error
        return False


def test_library_reports_its_version_and_capabilities() raises:
    """The loaded engine is of the binding's major, and both capability lists are non-empty."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    assert_equal(library.version()[0], 4)
    assert_true(library.version_name().byte_length() > 0)
    assert_true(library.capabilities_compiled().byte_length() > 0)
    assert_true(library.capabilities_available().byte_length() > 0)


def test_names_round_trip_through_the_library_and_refuse_strangers() raises:
    """Every enumerator's spelling reads back as the same enumerator, and a spelling no table holds is refused."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    assert_equal(library.status_name(Status.SHAPE_MISMATCH), "shape-mismatch")
    assert_true(library.status_from_name("io") == Status.IO)
    assert_equal(library.scalar_name(ScalarKind.F32), "f32")
    assert_true(library.scalar_from_name("bf16") == ScalarKind.BF16)
    assert_equal(library.mood_name(Mood.SPINNING), "spinning")
    assert_true(library.mood_from_name("parked") == Mood.PARKED)
    assert_equal(library.cpu_class_name(CPUClass.INTEL_HASWELL), "intel-haswell")
    assert_true(library.cpu_class_from_name("intel-haswell") == CPUClass.INTEL_HASWELL)
    assert_equal(library.gpu_class_name(GPUClass.NVIDIA_HOPPER), "nvidia-hopper")
    assert_true(library.gpu_class_from_name("nvidia-hopper") == GPUClass.NVIDIA_HOPPER)
    assert_equal(library.modality_name(Modality.AUDIO), "audio")
    assert_true(library.modality_from_name("image") == Modality.IMAGE)
    assert_equal(library.objective_name(Objective.JOINT), "joint")
    assert_true(library.objective_from_name("contrastive") == Objective.CONTRASTIVE)
    assert_equal(library.negatives_name(Negatives.NAMED), "named")
    assert_true(library.negatives_from_name("ambient") == Negatives.AMBIENT)
    assert_equal(library.multimem_tier_name(MultimemTier.PEER_READ), "peer-read")
    assert_true(library.multimem_tier_from_name("peer-read") == MultimemTier.PEER_READ)
    for code in range(4):
        var form = PairLossForm(c_int(code))
        assert_true(library.pair_loss_form_from_name(library.pair_loss_form_name(form)) == form)
        var side = LateInteractionSide(c_int(code))
        assert_true(library.late_interaction_side_from_name(library.late_interaction_side_name(side)) == side)
        var vendor = GPUVendor(c_int(code))
        assert_true(library.gpu_vendor_from_name(library.gpu_vendor_name(vendor)) == vendor)
    assert_equal(library.pair_loss_form_name(PairLossForm.SIGMOID), "sigmoid")
    assert_equal(library.late_interaction_side_name(LateInteractionSide.QUERY_AND_DOCUMENT), "query-and-document")
    assert_equal(library.gpu_vendor_name(GPUVendor.NVIDIA), "nvidia")
    assert_equal(library.status_name(Status.UNINITIALIZED), "uninitialized")
    var refused = Status.SUCCESS
    try:
        _ = library.mood_from_name("sleepy")
    except error:
        refused = error.status
    assert_true(refused == Status.INVALID_ARGUMENT, "an unrecognized spelling must be refused, not read as a default")


def test_gpu_devices_count_what_the_process_drives() raises:
    """The listing fills no more entries than it is given, and a host-only build counts none."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var devices = List[GPUDevice]()
    for _ in range(4):
        devices.append(GPUDevice())
    var count = library.gpu_devices(devices)
    assert_true(count >= 0)
    for position in range(min(count, len(devices))):
        assert_true(devices[position].memory_bytes > 0 and devices[position].name().byte_length() > 0)


def test_executors_and_arenas_open_and_drain() raises:
    """A RAM executor drains, and arenas open in host pages on the calling thread and on the executor."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, RamOptions(), threads=2)
    executor.drain(Mood.SPINNING)
    executor.drain()
    var inline_arena = Arena(library)
    var pooled_arena = Arena(executor)
    assert_true(inline_arena.describe().residency == Residency.RAM and pooled_arena.describe().device == -1)


def test_a_missing_checkpoint_is_refused_with_its_status() raises:
    """Loading a path that does not exist raises `IO`."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var refused = Status.SUCCESS
    try:
        _ = Model.load_ram(Executor(library), "/nonexistent/uform/model.safetensors")
    except error:
        refused = error.status
    assert_true(refused == Status.IO, "a missing checkpoint must be refused")


def test_undecodable_bytes_are_refused() raises:
    """A buffer holding no safetensors checkpoint is refused rather than loaded."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var bytes = "stand-in checkpoint bytes".as_bytes()
    var refused = Status.SUCCESS
    try:
        _ = Model.load_ram_buffer(Executor(library), bytes)
    except error:
        refused = error.status
    assert_true(refused != Status.SUCCESS and refused != Status.DEFERRED, "stand-in bytes must be refused")


def test_a_device_executor_needs_a_device() raises:
    """Without a drivable GPU, a device executor is refused rather than made."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var none = List[GPUDevice]()
    if library.gpu_devices(none) > 0:
        _skipped("needs a machine without a GPU")
        return
    var refused = Status.SUCCESS
    try:
        _ = Executor.for_vram(library)
    except error:
        refused = error.status
    assert_true(refused == Status.UNSUPPORTED or refused == Status.DEVICE, "a device executor needs a device")


def test_training_verbs_refuse_a_host_executor() raises:
    """Every training constructor needs a device executor, and a RAM one is refused with `DEVICE`; a
    verb on a null trainer or mesh is refused with `UNINITIALIZED` before anything else."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=1)
    var statuses = List[Status]()
    try:
        _ = Mesh.bootstrap(executor)
    except error:
        statuses.append(error.status)
    try:
        _ = Trainer.init_vram(executor, TrainerInitVramConfig())
    except error:
        statuses.append(error.status)
    var trainer = Trainer[ImmUntrackedOrigin](library=library, address=NULL_ADDRESS, mesh=None)
    var modalities: List[Modality] = [Modality.TEXT]
    var contents = "a".as_bytes()
    var offsets: List[UInt] = [0, 1]
    var arena = Arena(executor)
    try:
        trainer.on(executor).step(modalities, TapeView.of(contents, offsets))
    except error:
        statuses.append(error.status)
    try:
        _ = trainer.on(executor).step(modalities, TapeView.of(contents, offsets), arena)
    except error:
        statuses.append(error.status)
    try:
        trainer.on(executor).save("uform-mojo-refused.safetensors")
    except error:
        statuses.append(error.status)
    var call = trainer.deferred(executor).save("uform-mojo-refused.safetensors")
    try:
        _ = call.wait()
    except error:
        statuses.append(error.status)
    assert_equal(len(statuses), 6, "every training verb must be refused")
    for position in range(len(statuses)):
        if position < 2:
            assert_true(statuses[position] == Status.DEVICE, "a RAM executor is refused as the wrong device")
        else:
            assert_true(statuses[position] == Status.UNINITIALIZED, "a null trainer is refused before its executor")
    var refused = Status.SUCCESS
    try:
        _ = Mesh(library=library, address=NULL_ADDRESS).describe()
    except error:
        refused = error.status
    assert_true(refused == Status.UNINITIALIZED, "a null mesh is refused before the build's own refusal")


# endregion Engine

# region Model


def _checkpoint() -> String:
    """The tiny checkpoint the suites share, or empty when none is named."""
    return getenv("UFORM_TEST_CHECKPOINT")


def test_a_loaded_model_describes_itself_on_every_receiver() raises:
    """A loaded model reports a geometry, the same from its path as from its bytes."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var model = Model.load_ram(executor, path)
    var description = model.describe()
    assert_true(description.dimensions > 0 and description.context > 0)
    assert_true(description.quantization == ScalarKind.BF16)
    var bytes = open(path, "r").read_bytes()
    var from_buffer = Model.load_ram_buffer(executor, bytes).on(executor).describe()
    assert_equal(Int(from_buffer.dimensions), Int(description.dimensions))


def test_a_ceiling_loads_the_best_class_under_it() raises:
    """A serial ceiling loads serial, and a ceiling of a family this CPU does not run falls to serial."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    for ceiling in [CPUClass.SERIAL, CPUClass.WASM64_EXTENDED]:
        var model = Model.load_ram(executor, path, RamOptions(cpu_class_ceiling=ceiling))
        assert_true(model.describe().cpu_class == CPUClass.SERIAL)


def test_texts_land_one_embedding_per_item_and_one_state_per_byte() raises:
    """Embeddings land one row per text at the model's width, and states one row per byte."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var model = Model.load_ram(executor, path)
    var dimensions = Int(model.describe().dimensions)
    var texts = "a caturban dog".as_bytes()
    var offsets: List[UInt] = [0, 5, 14]
    var arena = Arena(executor)
    var stats = EncodeStats()
    var encoded = model.on(executor).text_encode(
        TapeView.of(texts, offsets),
        Outputs.EMBEDDINGS | Outputs.STATES,
        arena,
        EncodeConfig(stats=Pointer(to=stats)),
    )
    var embeddings = encoded.embeddings().value()
    assert_equal(len(embeddings), 2)
    assert_equal(embeddings.columns(), dimensions)
    assert_equal(len(embeddings.row[DType.float32](1)), dimensions)
    assert_equal(len(embeddings.values[DType.float32]()), 2 * dimensions)
    var states = encoded.states().value()
    assert_equal(Int(states.offsets()[2]), Int(stats.tokens))
    assert_false(Bool(encoded.token_vectors()))


def test_a_later_call_retires_earlier_rows() raises:
    """A second call on the same arena rewinds it, so reading the first call's rows is refused."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var model = Model.load_ram(Executor(library), path)
    var texts = "first".as_bytes()
    var offsets: List[UInt] = [0, 5]
    var arena = Arena(library)
    var first = model.text_encode(TapeView.of(texts, offsets), Outputs.EMBEDDINGS, arena).embeddings().value()
    _ = model.text_encode(TapeView.of(texts, offsets), Outputs.EMBEDDINGS, arena)
    var refused = Status.SUCCESS
    try:
        _ = first.row[DType.float32](0)
    except error:
        refused = error.status
    assert_true(refused == Status.INVALID_ARGUMENT, "a rewound arena's rows must be refused")


def test_deferred_encoding_matches_the_blocking_one() raises:
    """A deferred call reads `DEFERRED` or final, and its waited rows equal a blocking call's."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var model = Model.load_ram(executor, path)
    var texts = "a deferred caption".as_bytes()
    var offsets: List[UInt] = [0, UInt(len(texts))]
    var blocking_arena = Arena(executor)
    var deferred_arena = Arena(executor)
    var blocking = model.on(executor).text_encode(TapeView.of(texts, offsets), Outputs.EMBEDDINGS, blocking_arena)
    var call = model.deferred(executor).text_encode(TapeView.of(texts, offsets), Outputs.EMBEDDINGS, deferred_arena)
    var status = call.status()
    assert_true(status == Status.DEFERRED or status == Status.SUCCESS)
    var waited = call.wait().embeddings().value()
    var expected = blocking.embeddings().value()
    for position in range(expected.columns()):
        assert_equal(waited.row[DType.float32](0)[position], expected.row[DType.float32](0)[position])
    _ = blocking_arena^  # the rows read their arenas, which Mojo would free at their last use
    _ = deferred_arena^


async def _encode_both[
    texts_origin: ImmOrigin, //
](
    first: Model, second: Model, first_executor: Executor, second_executor: Executor, texts: TapeView[texts_origin]
) raises -> Int:
    """Encodes the same texts on two models at once, each on its own executor and arena, and counts the rows."""
    var first_arena = Arena(first_executor)
    var second_arena = Arena(second_executor)
    var first_call = first.deferred(first_executor).text_encode(texts, Outputs.EMBEDDINGS, first_arena)
    var second_call = second.deferred(second_executor).text_encode(texts, Outputs.EMBEDDINGS, second_arena)
    ref first_encoded = await first_call
    ref second_encoded = await second_call
    return len(first_encoded.embeddings().value()) + len(second_encoded.embeddings().value())


def test_awaited_fan_out_answers_both_models() raises:
    """Two deferred encodings submitted before either is awaited both answer inside one task."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var first_executor = Executor.for_ram(library, threads=2)
    var second_executor = Executor.for_ram(library, threads=2)
    var first = Model.load_ram(first_executor, path)
    var second = Model.load_ram(second_executor, path)
    var texts = "onetwo".as_bytes()
    var offsets: List[UInt] = [0, 3, 6]
    var both = create_raising_task(
        _encode_both(first, second, first_executor, second_executor, TapeView.of(texts, offsets))
    )
    assert_equal(both^.wait(), 4, "two rows from each model")


async def _encode_pixels_once[
    pixels_origin: ImmOrigin, heights_origin: ImmOrigin, widths_origin: ImmOrigin, //
](
    model: Model,
    executor: Executor,
    pixels: TapeView[pixels_origin],
    heights: Span[UInt32, heights_origin],
    widths: Span[UInt32, widths_origin],
) raises -> Int:
    """Awaits one deferred pixel encoding and answers the status code it failed with, or zero when it succeeded."""
    var arena = Arena(executor)
    var call = model.deferred(executor).image_encode_pixels(pixels, heights, widths, Outputs.EMBEDDINGS, arena)
    try:
        ref encoded = await call
        _ = len(encoded.embeddings().value())
    except error:
        return Int(error.status.code)
    return 0


def test_awaited_failures_raise() raises:
    """A deferred call the binding refuses raises its failure through `await` and the task's wait."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var model = Model.load_ram(executor, path)
    var pixels = List[UInt8](length=12, fill=0)
    var offsets: List[UInt] = [0, 12]
    var heights = List[UInt32]()
    var widths: List[UInt32] = [2]
    var failing = create_raising_task(
        _encode_pixels_once(model, executor, TapeView.of(pixels, offsets), heights, widths)
    )
    assert_equal(failing^.wait(), Int(Status.SHAPE_MISMATCH.code), "the refusal reaches the awaiting task")


def test_arenas_free_before_their_executor() raises:
    """An arena refers to the executor that made it, so the executor outlives it past its own last use."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var model = Model.load_ram(Executor(library), path)
    var executor = Executor.for_ram(library, threads=2)
    var arena = Arena(executor)
    var texts = "onetwo".as_bytes()
    var offsets: List[UInt] = [0, 3, 6]
    for _ in range(2):
        var encoded = model.text_encode(TapeView.of(texts, offsets), Outputs.EMBEDDINGS, arena)
        assert_equal(len(encoded.embeddings().value()), 2, "the arena serves calls after its executor's last use")


@fieldwise_init
struct _Reentry(Movable):
    """The model and executor a handler calls again on, and how its calls ended."""

    var model: Pointer[Model, MutUntrackedOrigin]
    var executor: Pointer[Executor, MutUntrackedOrigin]
    var calls: Int
    var refusals: Int


def _call_again(completed: Int, total: Int, reentry: Pointer[_Reentry, MutAnyOrigin]) -> Continuation:
    """A progress handler calling `describe` on the executor its own call holds."""
    reentry[].calls += 1
    try:
        _ = reentry[].model[].on(reentry[].executor[]).describe()
    except error:
        if error.status == Status.CONTENDED:
            reentry[].refusals += 1
    return Continuation.CONTINUE


def test_a_held_executor_refuses_other_calls() raises:
    """A call made from the handler of a call holding the executor raises `CONTENDED` without running."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var model = Model.load_ram(executor, path)
    var texts = "onetwo".as_bytes()
    var offsets: List[UInt] = [0, 3, 6]
    var arena = Arena(executor)
    var reentry = _Reentry(
        Pointer(to=model).unsafe_origin_cast[MutUntrackedOrigin](),
        Pointer(to=executor).unsafe_origin_cast[MutUntrackedOrigin](),
        0,
        0,
    )
    var progress = Progress.of[_call_again](Pointer(to=reentry))
    var encoded = model.on(executor).text_encode(
        TapeView.of(texts, offsets), Outputs.EMBEDDINGS, arena, EncodeConfig(progress=progress)
    )
    assert_equal(len(encoded.embeddings().value()), 2)
    assert_true(reentry.calls > 0, "the handler must hear a report")
    assert_equal(reentry.refusals, reentry.calls)
    _ = model.on(executor).describe()


def _stop_at_first_report(completed: Int, total: Int, reports: Pointer[Int, MutAnyOrigin]) -> Continuation:
    """A progress handler counting its reports and cancelling at the first."""
    reports[] += 1
    return Continuation.CANCEL


def test_a_stopping_handler_cancels_the_call() raises:
    """A handler answering `CANCEL` stops the call, which finishes `CANCELLED` after one report."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var model = Model.load_ram(Executor(library), path)
    var texts = "onetwothreefour".as_bytes()
    var offsets: List[UInt] = [0, 3, 6, 11, 15]
    var arena = Arena(library)
    var reports = 0
    var progress = Progress.of[_stop_at_first_report](Pointer(to=reports), steps=4)
    var refused = Status.SUCCESS
    try:
        _ = model.text_encode(TapeView.of(texts, offsets), Outputs.EMBEDDINGS, arena, EncodeConfig(progress=progress))
    except error:
        refused = error.status
    assert_true(refused == Status.CANCELLED, "a cancelling handler must stop the call")
    assert_equal(reports, 1)


def test_choices_sum_to_one_or_are_refused() raises:
    """Each item's probabilities sum to one over its legal choices, or a model without the byte decoder refuses them."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var model = Model.load_ram(Executor(library), path)
    var states = "yes or no? ".as_bytes()
    var state_offsets: List[UInt] = [0, UInt(len(states))]
    var choices = "yn".as_bytes()
    var choice_offsets: List[UInt] = [0, 2]
    var groups: List[UInt32] = [0]
    var arena = Arena(library)
    if model.describe().choice_temperature_groups == 0:
        var refused = Status.SUCCESS
        try:
            _ = model.text_choose(TapeView.of(states, state_offsets), TapeView.of(choices, choice_offsets), arena)
        except error:
            refused = error.status
        assert_true(refused == Status.UNSUPPORTED, "a model without the byte decoder must refuse choices")
        return
    var probabilities = model.text_choose(
        TapeView.of(states, state_offsets),
        TapeView.of(choices, choice_offsets),
        arena,
        TextChooseConfig(temperature_groups=groups),
    )
    var item = probabilities.item[DType.float32](0)
    assert_equal(len(item), 2)
    assert_true(abs(item[0] + item[1] - 1) < 1e-3)


def test_pixels_and_samples_encode_without_codecs() raises:
    """Decoded pixels and samples encode into one embedding each."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var model = Model.load_ram(Executor(library), path)
    var arena = Arena(library)
    var pixels = List[UInt8](length=16 * 16 * 3, fill=128)
    var pixel_offsets: List[UInt] = [0, UInt(len(pixels))]
    var heights: List[UInt32] = [16]
    var widths: List[UInt32] = [16]
    var image = model.image_encode_pixels(
        TapeView.of(pixels, pixel_offsets), heights, widths, Outputs.EMBEDDINGS, arena
    )
    assert_equal(len(image.embeddings().value()), 1)
    var samples = List[Float32](length=4096, fill=0.25)
    var sample_offsets: List[UInt] = [0, UInt(len(samples))]
    var recording = model.audio_encode_samples(TapeView.of(samples, sample_offsets), 16000, Outputs.EMBEDDINGS, arena)
    assert_equal(len(recording.embeddings().value()), 1)


def test_mismatched_pixel_shapes_are_refused_before_the_call() raises:
    """Heights and widths must name every item, which the binding checks before crossing."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var model = Model.load_ram(Executor(library), path)
    var arena = Arena(library)
    var pixels = List[UInt8](length=12, fill=0)
    var offsets: List[UInt] = [0, 12]
    var heights = List[UInt32]()
    var widths: List[UInt32] = [2]
    var refused = Status.SUCCESS
    try:
        _ = model.image_encode_pixels(TapeView.of(pixels, offsets), heights, widths, Outputs.EMBEDDINGS, arena)
    except error:
        refused = error.status
    assert_true(refused == Status.SHAPE_MISMATCH, "missing heights must be refused")


# endregion Model


# region Quickstart


def _cosine(text: Span[Float32, _], image: Span[Float32, _]) -> Float32:
    var dot: Float32 = 0  # embeddings come unnormalized
    var text_norm: Float32 = 0
    var image_norm: Float32 = 0
    for i in range(len(image)):
        dot += text[i] * image[i]
        text_norm += text[i] * text[i]
        image_norm += image[i] * image[i]
    return dot / sqrt(text_norm * image_norm)


def test_readme_quickstart() raises:
    """The Quickstart of `mojo/README.md`, line for line, but loading the suite's checkpoint in place
    of `model.safetensors` and an 8 x 8 PNG in place of reading `cat.png`; a build without host
    decoders refuses the photo, which the README's build decodes."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var path = _checkpoint()
    if path.byte_length() == 0:
        _skipped(checkpoint_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library)
    var model = Model.load_ram(executor, path)
    var texts_arena = Arena(executor)
    var images_arena = Arena(executor)

    var captions: List[String] = ["a cat on a mat", "a sunrise over the bay"]
    var caption_offsets: List[UInt] = [0, 14, 36]
    # fmt: off
    var photo: List[UInt8] = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 8, 0, 0, 0, 8,
        8, 2, 0, 0, 0, 75, 109, 41, 220, 0, 0, 0, 75, 73, 68, 65, 84, 120, 156, 99, 96, 224, 16, 144,
        80, 208, 48, 176, 112, 240, 8, 136, 72, 200, 40, 168, 104, 232, 152, 48, 99, 193, 138, 13, 59, 24, 176, 9,
        30, 56, 113, 225, 6, 3, 54, 193, 7, 47, 62, 252, 96, 192, 42, 8, 52, 158, 1, 155, 32, 200, 120, 6,
        108, 130, 32, 227, 25, 176, 9, 130, 140, 103, 192, 38, 8, 50, 30, 0, 179, 100, 93, 1, 16, 63, 179, 153,
        0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ]
    # fmt: on
    var photo_offsets: List[UInt] = [0, UInt(len(photo))]
    var tape = TapeView.of("a cat on a mata sunrise over the bay".as_bytes(), caption_offsets)
    var texts = model.on(executor).text_encode(tape, Outputs.EMBEDDINGS, texts_arena).embeddings().value()
    var best = -1
    var refused = Status.SUCCESS
    try:
        var images = model.on(executor).image_encode(
            TapeView.of(photo, photo_offsets), Outputs.EMBEDDINGS, images_arena
        )
        var image = images.embeddings().value().row[DType.float32](0)
        best = 1 if _cosine(texts.row[DType.float32](1), image) > _cosine(texts.row[DType.float32](0), image) else 0
    except error:
        refused = error.status
    if best >= 0:
        assert_true(captions[best].byte_length() > 0)
    else:
        assert_true(refused == Status.UNSUPPORTED, "only a build without host decoders refuses the photo")


# endregion Quickstart


def main() raises:
    var suite = TestSuite()
    suite.test[test_outputs_combine_as_flags]()
    suite.test[test_scalar_kinds_follow_their_element_types]()
    suite.test[test_tapes_refuse_offsets_past_their_elements]()
    suite.test[test_library_reports_its_version_and_capabilities]()
    suite.test[test_names_round_trip_through_the_library_and_refuse_strangers]()
    suite.test[test_gpu_devices_count_what_the_process_drives]()
    suite.test[test_executors_and_arenas_open_and_drain]()
    suite.test[test_a_missing_checkpoint_is_refused_with_its_status]()
    suite.test[test_undecodable_bytes_are_refused]()
    suite.test[test_a_device_executor_needs_a_device]()
    suite.test[test_training_verbs_refuse_a_host_executor]()
    suite.test[test_a_loaded_model_describes_itself_on_every_receiver]()
    suite.test[test_a_ceiling_loads_the_best_class_under_it]()
    suite.test[test_texts_land_one_embedding_per_item_and_one_state_per_byte]()
    suite.test[test_a_later_call_retires_earlier_rows]()
    suite.test[test_deferred_encoding_matches_the_blocking_one]()
    suite.test[test_awaited_fan_out_answers_both_models]()
    suite.test[test_awaited_failures_raise]()
    suite.test[test_arenas_free_before_their_executor]()
    suite.test[test_a_stopping_handler_cancels_the_call]()
    suite.test[test_a_held_executor_refuses_other_calls]()
    suite.test[test_choices_sum_to_one_or_are_refused]()
    suite.test[test_pixels_and_samples_encode_without_codecs]()
    suite.test[test_mismatched_pixel_shapes_are_refused_before_the_call]()
    suite.test[test_readme_quickstart]()
    suite^.run()
