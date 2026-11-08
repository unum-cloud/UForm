"""The Mojo binding's tests, registered in `main` since Mojo has no test discovery.

The pure-logic group needs no engine. The rest call the engine and are skipped when nothing is on
the loader path, and the model checks also need the tiny checkpoint `UFORM_TEST_CHECKPOINT` names,
which `uform_tiny_checkpoint` writes; without one they are skipped, as the Python and Rust suites
skip theirs against the engine.

File: mojo/test.mojo
Author: Ash Vardanian
Date: November 8, 2026
"""

from std.os import getenv
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
    Library,
    Mesh,
    Modality,
    Model,
    Mood,
    MultimemTier,
    Negatives,
    Objective,
    Outputs,
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


def _engine_is_present() raises -> Bool:
    """Whether anything answers on the loader path; a library missing a symbol fails rather than skips."""
    try:
        _ = Library()
        return True
    except error:
        if error.status != Status.LIBRARY_MISSING:
            raise error
        return False


def test_library_reports_its_version_and_capabilities() raises:
    """The loaded engine is of the binding's major, and both capability lists are non-empty."""
    if not _engine_is_present():
        print("skipped: libuform_shared is not on the loader path")
        return
    var library = Library()
    assert_equal(library.version()[0], 4)
    assert_true(library.version_name().byte_length() > 0)
    assert_true(library.capabilities_compiled().byte_length() > 0)
    assert_true(library.capabilities_available().byte_length() > 0)


def test_names_round_trip_through_the_engine_tables() raises:
    """Every enumerator's spelling reads back as the same enumerator."""
    if not _engine_is_present():
        print("skipped: libuform_shared is not on the loader path")
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


def test_gpu_devices_count_what_the_process_drives() raises:
    """The listing fills no more entries than it is given, and a host-only build counts none."""
    if not _engine_is_present():
        print("skipped: libuform_shared is not on the loader path")
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
    if not _engine_is_present():
        print("skipped: libuform_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, RamOptions(), threads=2)
    executor.drain(Mood.SPINNING)
    executor.drain()
    var inline_arena = Arena(library)
    var pooled_arena = Arena(executor)
    assert_true(inline_arena.describe().residency == Residency.RAM and pooled_arena.describe().device == -1)


def test_a_missing_checkpoint_is_refused_with_its_status() raises:
    """Loading a path that does not exist raises `IO`, and the error names the engine and the verb."""
    if not _engine_is_present():
        print("skipped: libuform_shared is not on the loader path")
        return
    var library = Library()
    var refused = Status.SUCCESS
    var text = String()
    try:
        _ = Model.load_ram(Executor(library), "/nonexistent/uform/model.safetensors")
    except error:
        refused = error.status
        text = String(error)
    assert_true(refused == Status.IO, "a missing checkpoint must be refused")
    assert_true(text.startswith("UForm: io [uform_model_load_ram]"), text)


def test_undecodable_bytes_are_refused() raises:
    """A buffer holding no safetensors checkpoint is refused rather than loaded."""
    if not _engine_is_present():
        print("skipped: libuform_shared is not on the loader path")
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
    if not _engine_is_present():
        print("skipped: libuform_shared is not on the loader path")
        return
    var library = Library()
    var none = List[GPUDevice]()
    if library.gpu_devices(none) > 0:
        print("skipped: this machine drives a GPU")
        return
    var refused = Status.SUCCESS
    try:
        _ = Executor.for_vram(library)
    except error:
        refused = error.status
    assert_true(refused == Status.UNSUPPORTED or refused == Status.DEVICE, "a device executor needs a device")


def test_training_verbs_refuse_a_host_executor() raises:
    """Every training verb needs a device executor, and a RAM one is refused with `DEVICE`."""
    if not _engine_is_present():
        print("skipped: libuform_shared is not on the loader path")
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
    for status in statuses:
        assert_true(status == Status.DEVICE, "a RAM executor is refused as the wrong device")
    var refused = Status.SUCCESS
    try:
        _ = Mesh(library=library, address=NULL_ADDRESS).describe()
    except error:
        refused = error.status
    assert_true(refused == Status.UNSUPPORTED, "a build without the trainer describes no mesh")


# endregion Engine

# region Model


def _checkpoint() -> String:
    """The tiny checkpoint the suites share, or empty when none is named."""
    return getenv("UFORM_TEST_CHECKPOINT")


def test_model_describes_a_loaded_checkpoint() raises:
    """A loaded model reports a geometry, the same from its path as from its bytes."""
    var path = _checkpoint()
    if not _engine_is_present() or path.byte_length() == 0:
        print("skipped: UFORM_TEST_CHECKPOINT names no checkpoint")
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


def test_text_encode_lands_rows_in_the_arena() raises:
    """Embeddings land one row per text at the model's width, and states one row per byte."""
    var path = _checkpoint()
    if not _engine_is_present() or path.byte_length() == 0:
        print("skipped: UFORM_TEST_CHECKPOINT names no checkpoint")
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
    var path = _checkpoint()
    if not _engine_is_present() or path.byte_length() == 0:
        print("skipped: UFORM_TEST_CHECKPOINT names no checkpoint")
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
    var path = _checkpoint()
    if not _engine_is_present() or path.byte_length() == 0:
        print("skipped: UFORM_TEST_CHECKPOINT names no checkpoint")
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
    var waited = call.wait().embeddings().value().row[DType.float32](0)
    var expected = blocking.embeddings().value().row[DType.float32](0)
    for position in range(len(expected)):
        assert_equal(waited[position], expected[position])


def _stop_at_first_report(completed: Int, total: Int, reports: Pointer[Int, MutAnyOrigin]) -> Continuation:
    """A progress handler counting its reports and cancelling at the first."""
    reports[] += 1
    return Continuation.CANCEL


def test_progress_cancels_an_encoding() raises:
    """A handler answering `CANCEL` stops the call, which finishes `CANCELLED` after one report."""
    var path = _checkpoint()
    if not _engine_is_present() or path.byte_length() == 0:
        print("skipped: UFORM_TEST_CHECKPOINT names no checkpoint")
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


def test_text_choose_answers_a_probability_per_choice() raises:
    """Each item's probabilities sum to one over its legal choices."""
    var path = _checkpoint()
    if not _engine_is_present() or path.byte_length() == 0:
        print("skipped: UFORM_TEST_CHECKPOINT names no checkpoint")
        return
    var library = Library()
    var model = Model.load_ram(Executor(library), path)
    if model.describe().choice_temperature_groups == 0:
        print("skipped: the checkpoint carries no byte decoder")
        return
    var states = "yes or no? ".as_bytes()
    var state_offsets: List[UInt] = [0, UInt(len(states))]
    var choices = "yn".as_bytes()
    var choice_offsets: List[UInt] = [0, 2]
    var groups: List[UInt32] = [0]
    var arena = Arena(library)
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
    var path = _checkpoint()
    if not _engine_is_present() or path.byte_length() == 0:
        print("skipped: UFORM_TEST_CHECKPOINT names no checkpoint")
        return
    var library = Library()
    var model = Model.load_ram(Executor(library), path)
    var arena = Arena(library)
    var pixels = List[UInt8](length=16 * 16 * 3, fill=128)
    var pixel_offsets: List[UInt] = [0, UInt(len(pixels))]
    var heights: List[UInt32] = [16]
    var widths: List[UInt32] = [16]
    var image = model.image_encode_pixels(TapeView.of(pixels, pixel_offsets), heights, widths, Outputs.EMBEDDINGS, arena)
    assert_equal(len(image.embeddings().value()), 1)
    var samples = List[Float32](length=4096, fill=0.25)
    var sample_offsets: List[UInt] = [0, UInt(len(samples))]
    var recording = model.audio_encode_samples(TapeView.of(samples, sample_offsets), 16000, Outputs.EMBEDDINGS, arena)
    assert_equal(len(recording.embeddings().value()), 1)


def test_mismatched_pixel_shapes_are_refused_before_the_call() raises:
    """Heights and widths must name every item, which the binding checks before crossing."""
    var path = _checkpoint()
    if not _engine_is_present() or path.byte_length() == 0:
        print("skipped: UFORM_TEST_CHECKPOINT names no checkpoint")
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


def main() raises:
    var suite = TestSuite()
    suite.test[test_outputs_combine_as_flags]()
    suite.test[test_scalar_kinds_follow_their_element_types]()
    suite.test[test_tapes_refuse_offsets_past_their_elements]()
    suite.test[test_library_reports_its_version_and_capabilities]()
    suite.test[test_names_round_trip_through_the_engine_tables]()
    suite.test[test_gpu_devices_count_what_the_process_drives]()
    suite.test[test_executors_and_arenas_open_and_drain]()
    suite.test[test_a_missing_checkpoint_is_refused_with_its_status]()
    suite.test[test_undecodable_bytes_are_refused]()
    suite.test[test_a_device_executor_needs_a_device]()
    suite.test[test_training_verbs_refuse_a_host_executor]()
    suite.test[test_model_describes_a_loaded_checkpoint]()
    suite.test[test_text_encode_lands_rows_in_the_arena]()
    suite.test[test_a_later_call_retires_earlier_rows]()
    suite.test[test_deferred_encoding_matches_the_blocking_one]()
    suite.test[test_progress_cancels_an_encoding]()
    suite.test[test_text_choose_answers_a_probability_per_choice]()
    suite.test[test_pixels_and_samples_encode_without_codecs]()
    suite.test[test_mismatched_pixel_shapes_are_refused_before_the_call]()
    suite^.run()
