"""Training: the keyword-built configuration structures, the mesh, the trainer and its step report.

Run with `uv run pytest test/python/train.py`. Training is GPU-only, so on a build without GPUs the
constructors raise `DeviceError`, and these suites check everything the binding validates before that.
"""

from __future__ import annotations

from array import array
from pathlib import Path

import pytest

import uform


def test_structures_default_to_zero():
    """A field left out is zero, which the library derives as its header documents."""
    batch = uform.BatchShape(max_rows_per_device=64)
    assert batch.max_rows_per_device == 64 and batch.max_columns_per_row == 0
    assert batch.image is None
    assert uform.Masking().token_mask_probability == 0.0
    assert uform.Contrastive().negatives is uform.Negatives.AMBIENT
    assert uform.Contrastive().pair_loss is uform.PairLossForm.DERIVED
    assert uform.LateInteraction(weight=0.5).pair_loss is uform.PairLossForm.DERIVED
    assert (uform.Choices().weight, uform.Choices().brier_weight) == (0.0, 0.0)
    assert batch.max_choices_per_item == 0
    assert uform.MeshOptions().directory is None


def test_nested_structures_take_tuples():
    """A nested structure takes its own type or a tuple of its fields in order."""
    batch = uform.BatchShape(image=(16,), video=uform.VideoShape(patch_side=16, tube_frames=2), audio=(400, 16000))
    assert batch.image.patch_side == 16 and batch.image.channels == 0
    assert batch.video.tube_frames == 2
    assert (batch.audio.chunk_samples, batch.audio.sample_rate) == (400, 16000)
    assert repr(batch.image) == "ImageShape(patch_side=16, channels=0)"


def test_structures_refuse_what_their_fields_cannot_hold():
    """Unknown names, positional values, negative sizes and strangers in nested or enum fields are refused."""
    with pytest.raises(TypeError):
        uform.Schedule(stepz=10)
    with pytest.raises(TypeError):
        uform.Schedule(10)
    with pytest.raises(uform.InvalidArgumentError):
        uform.Schedule(steps=-1)
    with pytest.raises(TypeError):
        uform.BatchShape(image=uform.AudioShape())
    with pytest.raises(TypeError):
        uform.BatchShape(image=(1, 2, 3))
    with pytest.raises(TypeError):
        uform.Masking(text_weight="1.0")
    with pytest.raises(TypeError):
        uform.Contrastive(negatives="named")
    with pytest.raises(uform.UnknownKindError):
        uform.Contrastive(negatives=7)


def test_relations_take_edges_or_tuples():
    """Each edge set takes an `Edges` or an (offsets, ids[, weights]) tuple of buffers, kept as given."""
    offsets, ids = array("Q", [0, 1, 2]), array("I", [1, 0])
    relations = uform.Relations(attract=(offsets, ids), repel=uform.Edges(offsets=offsets, ids=ids))
    assert isinstance(relations.attract, uform.Edges) and relations.attract.ids is ids
    assert relations.repel.weights is None and relations.ignore is None


def test_mesh_job_is_a_name(executor: uform.Executor, tmp_path: Path):
    """The job is a name, a `str`, refused as bytes or a path before the library is reached."""
    for job in (b"run-17", tmp_path):
        with pytest.raises(TypeError):
            uform.Mesh.bootstrap(executor, uform.MeshOptions(directory=tmp_path, job=job))


def test_step_report_starts_empty():
    """A fresh report has counted nothing."""
    stats = uform.TrainerStepStats()
    assert (stats.step, stats.loss, stats.truncated, stats.dropped) == (0, 0.0, 0, 0)
    assert (stats.late_interaction_loss, stats.choices_loss) == (0.0, 0.0)
    with pytest.raises(TypeError):
        uform.TrainerStepStats(1)


def test_training_enums_spell_the_header():
    """The training enums carry the C values and names."""
    assert [member.name for member in uform.Modality] == ["TEXT", "IMAGE", "VIDEO", "AUDIO"]
    assert [member.name for member in uform.Objective] == ["MASKED", "CONTRASTIVE", "JOINT"]
    assert [member.name for member in uform.Negatives] == ["AMBIENT", "NAMED"]
    assert [member.name for member in uform.MultimemTier] == ["SWITCH_REDUCED", "PEER_READ"]
    assert [member.name for member in uform.PairLossForm] == ["DERIVED", "BALANCED", "RANKED", "SIGMOID"]
    assert [member.name for member in uform.LateInteractionSide] == ["NONE", "QUERY", "DOCUMENT", "QUERY_AND_DOCUMENT"]


def test_mesh_and_trainer_need_a_gpu(executor: uform.Executor, tmp_path: Path):
    """On a build without GPUs, the bootstrap and the trainer report `DeviceError`."""
    if "serial" not in uform.capabilities_compiled():
        pytest.skip("the build carries a GPU backend")
    # A path crosses as the OS's bytes, so one that is not UTF-8 reaches the library too.
    for directory in (tmp_path, bytes(tmp_path) + b"/\xff"):
        options = uform.MeshOptions(host_domain_count=1, directory=directory, job="run-17", timeout_seconds=1.5)
        with pytest.raises(uform.DeviceError):
            uform.Mesh.bootstrap(executor, options)
    with pytest.raises(uform.DeviceError):
        uform.Trainer.init_vram(
            executor,
            depth=2,
            hidden=128,
            objective=uform.Objective.JOINT,
            batch=uform.BatchShape(
                max_rows_per_device=4, max_segments_per_device=8, max_columns_per_row=64, max_choices_per_item=4
            ),
            contrastive=uform.Contrastive(matryoshka_dimensions=[64, 128], pair_loss=uform.PairLossForm.BALANCED),
            late_interaction=uform.LateInteraction(weight=0.5, width=32),
            choices=uform.Choices(weight=1.0),
            schedule=uform.Schedule(steps=10),
            seed=42,
        )


def test_trainer_arguments_are_checked(executor: uform.Executor):
    """Arguments of the wrong type are refused before the library is reached."""
    with pytest.raises(TypeError):
        uform.Trainer.init_vram(executor, batch=uform.Schedule())
    with pytest.raises(TypeError):
        uform.Trainer.init_vram(executor, mesh=executor)
    with pytest.raises(TypeError):
        uform.Trainer.init_vram(executor, contrastive=uform.Contrastive(matryoshka_dimensions=8))
    with pytest.raises(TypeError):
        uform.Trainer.init_vram(executor, choices=uform.LateInteraction())
    with pytest.raises(TypeError):
        uform.Trainer()
    with pytest.raises(TypeError):
        uform.Mesh.bootstrap(executor, uform.Schedule())
