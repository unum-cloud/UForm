/**
 *  @file javascript/uform.ts
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief Pocket-Sized Multimodal AI for Content Understanding and Generation, for Node.
 *
 *      The package entry point: it binds the Node addon, and re-exports one module per C header underneath, so a
 *      caller imports `@unum-cloud/uform` and nothing deeper. Browsers import `@unum-cloud/uform/wasm` instead.
 */
import { loadAddon } from "./library.js";
import { backend, bind } from "./types.js";

bind(loadAddon());

export {
  Arena,
  ArenaView,
  Deferred,
  Executor,
  GpuVendor,
  LateInteractionSide,
  Modality,
  Mood,
  MultimemTier,
  Negatives,
  Objective,
  PairLossForm,
  Scalar,
  Status,
  gpuDevices,
} from "./types.js";
export type {
  Answer,
  ExecutorForRamConfig,
  ExecutorForVramConfig,
  ExecutorOnPoolConfig,
  GpuDevice,
  UFormError,
} from "./types.js";
export { EncodeVerbs, Outputs, Probabilities, Rows, States } from "./encode.js";
export type { ChooseConfig, Chosen, EncodeConfig, EncodeStats, Encoded, Pixels, Tape } from "./encode.js";
export { Model, ModelDeferredOn, ModelOn } from "./model.js";
export type { ModelDescription, ModelLoadRamConfig, ModelLoadVramConfig } from "./model.js";
export { Mesh, Trainer, TrainerDeferredOn } from "./train.js";
export type {
  BatchShape,
  Choices,
  Contrastive,
  Edges,
  LateInteraction,
  Masking,
  MeshDescription,
  MeshOptions,
  Schedule,
  Stepped,
  TrainerBatch,
  TrainerConfig,
  TrainerStepConfig,
} from "./train.js";
export type { Offsets, ProgressCallback } from "./library.js";

/** The library's version, as the release tooling stamped it. Calls `uform_version_name`. */
export function version(): string {
  return backend().version();
}

/** The hardware capabilities this binary was compiled with, comma-separated. Calls `uform_capabilities_compiled`. */
export function capabilitiesCompiled(): string {
  return backend().capabilitiesCompiled();
}

/** The hardware capabilities the current machine offers at runtime, comma-separated. Calls `uform_capabilities_available`. */
export function capabilitiesAvailable(): string {
  return backend().capabilitiesAvailable();
}
