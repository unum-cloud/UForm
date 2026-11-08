"""Distributed training and inference pipeline for multi-modal perception, bound over its C ABI.

UForm encodes texts, images, clips and recordings into embeddings with one model per checkpoint,
and trains those models across GPUs. This package binds the whole C interface: the model and its
encode verbs, the mesh and the trainer, and the executors and arenas they run on and write into.

The engine is reached by `dlopen`, so a consumer builds with no linker flags at all:

```mojo
from uform import Arena, Executor, Library, Model, Outputs, TapeView

def main() raises:
    var library = Library()
    var executor = Executor.for_ram(library)
    var model = Model.load_ram(executor, "model.safetensors")
    var arena = Arena(executor)
    var text = "a photo of a cat".as_bytes()
    var offsets: List[UInt] = [0, UInt(len(text))]
    var encoded = model.on(executor).text_encode(TapeView.of(text, offsets), Outputs.EMBEDDINGS, arena)
    print(encoded.embeddings().value().row[DType.float32](0)[0])
```

The receiver chooses where a verb runs and how it completes, and the verb stays the same: a verb
on the model runs on the calling thread, `model.on(executor)` blocks on that executor, and
`model.deferred(executor)` submits the call there and answers a `Deferred` at once.

This package root re-exports every public symbol from the modules that mirror the C headers, so
`from uform import X` resolves whatever module X lives in.

File: mojo/uform/__init__.mojo
Author: Ash Vardanian
Date: November 8, 2026
"""

from uform.encode import (
    EncodeConfig,
    Encoded,
    EncodeStats,
    Outputs,
    TextChooseConfig,
)
from uform.library import Library
from uform.model import Model, ModelDeferredOn, ModelDescription, ModelOn, ModelVerbs
from uform.train import (
    AudioShape,
    BatchShape,
    Choices,
    Contrastive,
    Edges,
    ImageShape,
    LateInteraction,
    LateInteractionSide,
    Masking,
    Mesh,
    MeshDescription,
    MeshOptions,
    Modality,
    MultimemTier,
    Negatives,
    Objective,
    PairLossForm,
    Relations,
    Schedule,
    Trainer,
    TrainerDeferredOn,
    TrainerInitVramConfig,
    TrainerOn,
    TrainerStepConfig,
    TrainerStepStats,
    TrainVerbs,
    VideoShape,
)
from uform.types import (
    Arena,
    ArenaDescription,
    BytesView,
    Completion,
    Continuation,
    CPUClass,
    Deferred,
    Error,
    Executor,
    GPUClass,
    GPUDevice,
    GPUVendor,
    Mood,
    Progress,
    RamOptions,
    Residency,
    Rows,
    RowsTape,
    ScalarKind,
    Status,
    Tape,
    TapeView,
    VramOptions,
)
