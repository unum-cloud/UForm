//! The model lifecycle: load an encoder into RAM or VRAM, describe it, and free it.
//!
//! Residency is in the constructor's name: [`Model::load_ram`] copies a checkpoint into host
//! memory and [`Model::load_vram`] onto devices, each with a `_buffer` twin reading the checkpoint
//! from memory. Quantization is the runtime [`Scalar`] passed at load, as in C, so a model carries
//! no type parameter. One model serves every modality it was trained on.
//!
//! Every verb comes from [`ModelVerbs`] and [`EncodeVerbs`](crate::EncodeVerbs), written once over
//! the receiver: a [`Model`] runs on the calling thread, `model.on(&executor)` blocks on an
//! executor, and `model.deferred(&executor)` answers a [`Deferred`](crate::Deferred) from the same
//! verbs given owned arguments.

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types)]

use core::ffi::c_void;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr;
use std::path::Path;

use crate::types::{
    check, fail, lower_progress, path_view, ram_options, uform_bytes_view_t,
    uform_completion_blocking_k, uform_completion_t, uform_cpu_class_t, uform_executor_t, uform_gpu_class_t,
    uform_outcome_t, uform_progress_t, uform_ram_options_t, uform_scalar_t, uform_vram_options_t, vram_options,
    Blocking, CPUClass, Callbacks, Calling, Completion, Deferring, Executor, GPUClass, Handles, On, ProgressCallback,
    Result, Scalar, Status,
};

// region: Raw Declarations

pub type uform_model_t = *mut c_void;

#[repr(C)]
pub struct uform_model_load_ram_t {
    pub model: *mut uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub path: uform_bytes_view_t,
    pub ram: uform_ram_options_t,
    pub quantization: uform_scalar_t,
    pub progress: uform_progress_t,
}

#[repr(C)]
pub struct uform_model_load_ram_buffer_t {
    pub model: *mut uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub buffer: uform_bytes_view_t,
    pub ram: uform_ram_options_t,
    pub quantization: uform_scalar_t,
    pub progress: uform_progress_t,
}

#[repr(C)]
pub struct uform_model_load_vram_t {
    pub model: *mut uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub path: uform_bytes_view_t,
    pub vram: uform_vram_options_t,
    pub quantization: uform_scalar_t,
    pub progress: uform_progress_t,
}

#[repr(C)]
pub struct uform_model_load_vram_buffer_t {
    pub model: *mut uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub buffer: uform_bytes_view_t,
    pub vram: uform_vram_options_t,
    pub quantization: uform_scalar_t,
    pub progress: uform_progress_t,
}

#[repr(C)]
pub struct uform_model_description_t {
    pub dimensions: usize,
    pub context: usize,
    pub quantization: uform_scalar_t,
    pub memory_usage: usize,
    pub cpu_class: uform_cpu_class_t,
    pub gpu_class: uform_gpu_class_t,
    pub token_width: usize,
    pub choice_temperature_groups: usize,
}

impl Default for uform_model_description_t {
    fn default() -> Self {
        uform_model_description_t {
            dimensions: 0,
            context: 0,
            quantization: 0,
            memory_usage: 0,
            cpu_class: 0,
            gpu_class: 0,
            token_width: 0,
            choice_temperature_groups: 0,
        }
    }
}

#[repr(C)]
pub struct uform_model_describe_t {
    pub model: uform_model_t,
    pub executor: uform_executor_t,
    pub outcome: *mut uform_outcome_t,
    pub completion: uform_completion_t,
    pub description: *mut uform_model_description_t,
}

extern "C" {
    pub fn uform_model_load_ram(arguments: *mut uform_model_load_ram_t);
    pub fn uform_model_load_ram_buffer(arguments: *mut uform_model_load_ram_buffer_t);
    pub fn uform_model_load_vram(arguments: *mut uform_model_load_vram_t);
    pub fn uform_model_load_vram_buffer(arguments: *mut uform_model_load_vram_buffer_t);
    pub fn uform_model_describe(arguments: *mut uform_model_describe_t);
    pub fn uform_model_free(model: uform_model_t);
}

// endregion: Raw Declarations

// region: Configs

/// What a loaded model reports about itself.
///
/// Mirrors `uform_model_description_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct ModelDescription {
    /// Scalars per embedding row; Matryoshka prefixes of it are embeddings too.
    pub dimensions: usize,
    /// Tokens one item may span before it is truncated: bytes of text, patches of an image, tubes
    /// of a clip, frames of a recording.
    pub context: usize,
    /// The scalar the transformer blocks run in.
    pub quantization: Scalar,
    /// Bytes the model holds; activations are the executor's.
    pub memory_usage: usize,
    /// The class a RAM model's tower was packed for; `None` for a VRAM model.
    pub cpu_class: Option<CPUClass>,
    /// The class every replica of a VRAM model runs as; `None` for a RAM model.
    pub gpu_class: Option<GPUClass>,
    /// Scalars per token vector; zero when the checkpoint carries no token head.
    pub token_width: usize,
    /// Question types the choice temperatures were calibrated over: zero without the byte decoder,
    /// one when uncalibrated.
    pub choice_temperature_groups: usize,
}

/// A checkpoint path and how it is copied into a new model in host memory; see [`Model::load_ram`].
///
/// Mirrors `uform_model_load_ram_t`.
pub struct ModelLoadRamConfig<'config> {
    /// A safetensors checkpoint, read in full and closed before the load completes.
    pub path: &'config Path,
    /// The executor the load runs on; `None` loads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// NUMA node IDs to place the weights in; empty derives every node the process may allocate on.
    pub memory_domains: &'config [i32],
    /// The class to pack and dispatch as, admitting every rung whose features it holds; `None`
    /// derives the CPU's own.
    pub cpu_class: Option<CPUClass>,
    /// The scalar the transformer blocks run in: [`Scalar::BF16`] as the checkpoint stores it, or
    /// [`Scalar::E4M3`] quantized at load.
    pub quantization: Scalar,
    /// Reports the tensors placed and stops wherever it says to; a stopped load creates no model.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for ModelLoadRamConfig<'config> {
    fn from(path: &'config Location) -> Self {
        ModelLoadRamConfig {
            path: path.as_ref(),
            executor: None,
            memory_domains: &[],
            cpu_class: None,
            quantization: Scalar::BF16,
            progress: None,
            progress_steps: 0,
        }
    }
}

/// A checkpoint path and how it is copied onto devices; see [`Model::load_vram`].
///
/// Mirrors `uform_model_load_vram_t`.
pub struct ModelLoadVramConfig<'config> {
    /// A safetensors checkpoint, read in full and closed before the load completes.
    pub path: &'config Path,
    /// The executor the upload runs on, one stream per device; `None` uploads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// Positions in what [`gpu_devices`](crate::gpu_devices) lists, each holding a replica; empty
    /// derives one free device.
    pub devices: &'config [i32],
    /// The class every device dispatches as; `None` derives the devices' own, which must agree.
    pub gpu_class: Option<GPUClass>,
    /// The scalar the transformer blocks run in, as for [`ModelLoadRamConfig::quantization`].
    pub quantization: Scalar,
    /// Reports the tensors placed and stops wherever it says to; a stopped load creates no model.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for ModelLoadVramConfig<'config> {
    fn from(path: &'config Location) -> Self {
        ModelLoadVramConfig {
            path: path.as_ref(),
            executor: None,
            devices: &[],
            gpu_class: None,
            quantization: Scalar::BF16,
            progress: None,
            progress_steps: 0,
        }
    }
}

/// A checkpoint held in memory and how it is copied into a new model in host memory; see
/// [`Model::load_ram_buffer`].
///
/// Mirrors `uform_model_load_ram_buffer_t`.
pub struct ModelLoadRamBufferConfig<'config> {
    /// The safetensors checkpoint itself, copied before the load completes.
    pub buffer: &'config [u8],
    /// The executor the load runs on; `None` loads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// NUMA node IDs to place the weights in; empty derives every node the process may allocate on.
    pub memory_domains: &'config [i32],
    /// The class to pack and dispatch as, as for [`ModelLoadRamConfig::cpu_class`].
    pub cpu_class: Option<CPUClass>,
    /// The scalar the transformer blocks run in, as for [`ModelLoadRamConfig::quantization`].
    pub quantization: Scalar,
    /// Reports the tensors placed and stops wherever it says to; a stopped load creates no model.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config> From<&'config [u8]> for ModelLoadRamBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        ModelLoadRamBufferConfig {
            buffer,
            executor: None,
            memory_domains: &[],
            cpu_class: None,
            quantization: Scalar::BF16,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for ModelLoadRamBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self { ModelLoadRamBufferConfig::from(&**bytes) }
}

/// A checkpoint held in memory and how it is copied onto devices; see [`Model::load_vram_buffer`].
///
/// Mirrors `uform_model_load_vram_buffer_t`.
pub struct ModelLoadVramBufferConfig<'config> {
    /// The safetensors checkpoint itself, in host or device memory, copied before the load completes.
    pub buffer: &'config [u8],
    /// The executor the upload runs on, one stream per device; `None` uploads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// Positions in what [`gpu_devices`](crate::gpu_devices) lists, each holding a replica; empty
    /// derives one free device.
    pub devices: &'config [i32],
    /// The class every device dispatches as, as for [`ModelLoadVramConfig::gpu_class`].
    pub gpu_class: Option<GPUClass>,
    /// The scalar the transformer blocks run in, as for [`ModelLoadRamConfig::quantization`].
    pub quantization: Scalar,
    /// Reports the tensors placed and stops wherever it says to; a stopped load creates no model.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config> From<&'config [u8]> for ModelLoadVramBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        ModelLoadVramBufferConfig {
            buffer,
            executor: None,
            devices: &[],
            gpu_class: None,
            quantization: Scalar::BF16,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for ModelLoadVramBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self { ModelLoadVramBufferConfig::from(&**bytes) }
}

// endregion: Configs

// region: Model

/// A loaded encoder: its weights, their packed copies, and on a GPU the session its attention runs
/// on, whose verbs come from [`ModelVerbs`] and [`EncodeVerbs`](crate::EncodeVerbs).
///
/// ```no_run
/// use uform::*;
///
/// let model = Model::load_ram("model.safetensors")?;
/// let executor = Executor::for_ram(&[], None, 0, 0)?;
/// let mut arena = Arena::new_in(&executor)?;
/// let encoded = model.on(&executor).text_encode(&["a caption"], Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
/// if let Some(embeddings) = encoded.embeddings {
///     println!("{} rows of {}", embeddings.count(), embeddings.columns());
/// }
/// # Ok::<(), uform::Error>(())
/// ```
///
/// Mirrors `uform_model_t`; dropping it calls `uform_model_free`.
#[derive(Debug)]
pub struct Model {
    pub(crate) handle: uform_model_t,
}

// Safety: the library shares a model across executors, whose calls take turns on its session.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Drop for Model {
    fn drop(&mut self) { unsafe { uform_model_free(self.handle) } }
}

impl Model {
    /// A model whose handle a constructor is about to fill.
    fn null() -> Self {
        Model {
            handle: ptr::null_mut(),
        }
    }

    /// The receiver whose verbs block on `executor`, after the calls submitted there before them.
    ///
    /// Rust-only.
    pub fn on<'model>(&'model self, executor: &'model Executor) -> On<'model, Model> {
        On {
            handle: self,
            executor,
            completion: PhantomData,
        }
    }

    /// The receiver whose verbs submit their calls to `executor`, answering a
    /// [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
    ///
    /// The library reads a deferred call's inputs after the verb returns, so lending borrowed texts
    /// does not compile:
    ///
    /// ```compile_fail,E0277
    /// use uform::*;
    ///
    /// let model = Model::load_ram("model.safetensors")?;
    /// let executor = Executor::for_ram(&[], None, 0, 0)?;
    /// let mut arena = Arena::new_in(&executor)?;
    /// let texts = vec![String::from("a caption")];
    /// let call = model.deferred(&executor).text_encode(&texts, Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
    /// # Ok::<(), uform::Error>(())
    /// ```
    ///
    /// Owned texts move into the call, and the arena is borrowed as a blocking call borrows it, and
    /// so is the waited result:
    ///
    /// ```no_run
    /// use uform::*;
    ///
    /// let model = Model::load_ram("model.safetensors")?;
    /// let executor = Executor::for_ram(&[], None, 0, 0)?;
    /// let mut arena = Arena::new_in(&executor)?;
    /// let texts = vec![String::from("a caption")];
    /// let call = model.deferred(&executor).text_encode(texts, Outputs::EMBEDDINGS, Scalar::F32, &mut arena)?;
    /// let encoded = call.wait()?;
    /// # Ok::<(), uform::Error>(())
    /// ```
    ///
    /// Rust-only.
    pub fn deferred<'model>(&'model self, executor: &'model Executor) -> On<'model, Model, Deferring> {
        On {
            handle: self,
            executor,
            completion: PhantomData,
        }
    }

    /// Copies a checkpoint file into host memory and packs its products for the CPU; a bare path
    /// loads on the calling thread in BFloat16.
    ///
    /// Calls `uform_model_load_ram`.
    pub fn load_ram<'config>(config: impl Into<ModelLoadRamConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let callbacks = Callbacks::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps, &callbacks);
        let mut model = Model::null();
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_model_load_ram(&mut uform_model_load_ram_t {
                model: &mut model.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: uform_completion_blocking_k,
                path: path_view(config.path)?,
                ram: ram_options(config.memory_domains, config.cpu_class),
                quantization: config.quantization as uform_scalar_t,
                progress,
            })
        };
        callbacks.raise();
        check(outcome).map(|()| model)
    }

    /// Copies a checkpoint held in memory into host memory and packs its products for the CPU.
    ///
    /// Calls `uform_model_load_ram_buffer`.
    pub fn load_ram_buffer<'config>(config: impl Into<ModelLoadRamBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let callbacks = Callbacks::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps, &callbacks);
        let mut model = Model::null();
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_model_load_ram_buffer(&mut uform_model_load_ram_buffer_t {
                model: &mut model.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: uform_completion_blocking_k,
                buffer: config.buffer.into(),
                ram: ram_options(config.memory_domains, config.cpu_class),
                quantization: config.quantization as uform_scalar_t,
                progress,
            })
        };
        callbacks.raise();
        check(outcome).map(|()| model)
    }

    /// Copies a checkpoint file onto the named GPUs and packs its products there.
    ///
    /// Calls `uform_model_load_vram`.
    pub fn load_vram<'config>(config: impl Into<ModelLoadVramConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let callbacks = Callbacks::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps, &callbacks);
        let mut model = Model::null();
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_model_load_vram(&mut uform_model_load_vram_t {
                model: &mut model.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: uform_completion_blocking_k,
                path: path_view(config.path)?,
                vram: vram_options(config.devices, config.gpu_class),
                quantization: config.quantization as uform_scalar_t,
                progress,
            })
        };
        callbacks.raise();
        check(outcome).map(|()| model)
    }

    /// Copies a checkpoint held in memory onto the named GPUs and packs its products there.
    ///
    /// Calls `uform_model_load_vram_buffer`.
    pub fn load_vram_buffer<'config>(config: impl Into<ModelLoadVramBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let callbacks = Callbacks::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps, &callbacks);
        let mut model = Model::null();
        let mut outcome = uform_outcome_t::default();
        unsafe {
            uform_model_load_vram_buffer(&mut uform_model_load_vram_buffer_t {
                model: &mut model.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: uform_completion_blocking_k,
                buffer: config.buffer.into(),
                vram: vram_options(config.devices, config.gpu_class),
                quantization: config.quantization as uform_scalar_t,
                progress,
            })
        };
        callbacks.raise();
        check(outcome).map(|()| model)
    }
}

/// What a verb answers through `Receiver`: what the call produced, or a [`Deferred`](crate::Deferred).
pub(crate) type Answer<'model, Receiver, Verb> =
    <<Receiver as ModelVerbs<'model>>::Completion as Completion>::Answer<'model, Verb>;

/// Every verb of the model header, written once for every receiver: `&Model` blocks on the calling
/// thread, and the [`On`] that [`Model::on`] or [`Model::deferred`] makes runs on an executor.
///
/// A blocking receiver answers what the call produced; a deferring one answers a
/// [`Deferred`](crate::Deferred) that borrows the model. [`EncodeVerbs`](crate::EncodeVerbs)
/// extends it with the forward passes.
///
/// Rust-only.
pub trait ModelVerbs<'model>: Copy {
    /// [`Blocking`] or [`Deferring`].
    type Completion: Completion;

    /// The model this receiver reaches and the executor its calls run on.
    fn handles(&self) -> Handles;

    /// Reports the model's geometry and placement in one call.
    ///
    /// Calls `uform_model_describe`.
    fn describe(self) -> Answer<'model, Self, ModelDescribe> {
        <Self::Completion as Completion>::answer(ModelDescribe {
            handles: self.handles(),
            description: uform_model_description_t::default(),
        })
    }
}

impl<'model> ModelVerbs<'model> for &'model Model {
    type Completion = Blocking;

    fn handles(&self) -> Handles {
        Handles {
            handle: self.handle,
            executor: ptr::null_mut(),
        }
    }
}

impl<'model, Completion: crate::types::Completion> ModelVerbs<'model> for On<'model, Model, Completion> {
    type Completion = Completion;

    fn handles(&self) -> Handles {
        Handles {
            handle: self.handle.handle,
            executor: self.executor.handle,
        }
    }
}

// endregion: Model

// region: Calls

/// A report of the model's geometry; see [`ModelVerbs::describe`].
///
/// Calls `uform_model_describe`.
pub struct ModelDescribe {
    handles: Handles,
    description: uform_model_description_t,
}

// Safety: the only pointer the description holds is the static capabilities name.
unsafe impl Send for ModelDescribe {}

impl Calling for ModelDescribe {
    type Arguments = uform_model_describe_t;
    type Output = ModelDescription;

    fn arguments(
        &mut self,
        outcome: *mut uform_outcome_t,
        completion: uform_completion_t,
        _: &Callbacks,
    ) -> Result<Self::Arguments> {
        Ok(uform_model_describe_t {
            model: self.handles.handle,
            executor: self.handles.executor,
            outcome,
            completion,
            description: &mut self.description,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { uform_model_describe(arguments) }

    fn finish(self) -> Result<ModelDescription> {
        Ok(ModelDescription {
            dimensions: self.description.dimensions,
            context: self.description.context,
            quantization: Scalar::from_raw(self.description.quantization).ok_or(fail(
                Status::Internal,
                "the library reported a quantization this binding cannot name",
            ))?,
            memory_usage: self.description.memory_usage,
            cpu_class: CPUClass::from_raw(self.description.cpu_class),
            gpu_class: GPUClass::from_raw(self.description.gpu_class),
            token_width: self.description.token_width,
            choice_temperature_groups: self.description.choice_temperature_groups,
        })
    }
}

// endregion: Calls

// region: Tests

#[cfg(test)]
pub(crate) mod tests {
    use core::ops::ControlFlow;

    use super::*;

    /// The tiny checkpoint `UFORM_TEST_CHECKPOINT` names, or stand-in bytes only the stub loads as a model.
    pub(crate) fn checkpoint() -> Vec<u8> {
        match std::env::var_os("UFORM_TEST_CHECKPOINT") {
            Some(path) => std::fs::read(path).expect("UFORM_TEST_CHECKPOINT names a readable checkpoint"),
            None => vec![7u8; 256],
        }
    }

    #[test]
    fn a_loaded_model_describes_itself_on_every_receiver() -> Result<()> {
        let checkpoint = checkpoint();
        let model = Model::load_ram_buffer(&checkpoint)?;
        let local = model.describe()?;
        assert_eq!(local.quantization, Scalar::BF16);
        assert!(local.dimensions != 0 && local.context != 0);

        let executor = Executor::for_ram(&[], None, 1, 0)?;
        assert_eq!(model.on(&executor).describe()?, local);
        assert_eq!(model.deferred(&executor).describe()?.wait()?, local);
        Ok(())
    }

    /// Every machine runs the serial class, so pinning it loads, and the description reports it.
    #[test]
    fn a_serial_pin_loads_and_describes_itself() -> Result<()> {
        let checkpoint = checkpoint();
        let config = ModelLoadRamBufferConfig {
            cpu_class: Some(CPUClass::Serial),
            ..ModelLoadRamBufferConfig::from(&checkpoint)
        };
        let description = Model::load_ram_buffer(config)?.describe()?;
        assert_eq!((description.cpu_class, description.gpu_class), (Some(CPUClass::Serial), None));
        Ok(())
    }

    #[test]
    fn descriptions_lay_out_as_the_header_declares() {
        use core::mem::{offset_of, size_of};

        let word = size_of::<usize>();
        let cpu_class = offset_of!(uform_model_description_t, cpu_class);
        assert_eq!(cpu_class, 4 * word);
        assert_eq!(offset_of!(uform_model_description_t, gpu_class), cpu_class + 4);
        assert_eq!(offset_of!(uform_model_description_t, token_width), cpu_class + 8);
        assert_eq!(size_of::<uform_model_description_t>(), cpu_class + 8 + 2 * word);
        assert_eq!(offset_of!(uform_ram_options_t, cpu_class), 2 * word);
        assert_eq!(offset_of!(uform_vram_options_t, gpu_class), 2 * word);
    }

    #[test]
    fn loads_refuse_what_they_cannot_read() {
        let missing = Model::load_ram("/nonexistent/uform/model.safetensors");
        assert_eq!(missing.unwrap_err().status, Status::Io);
        let empty = Model::load_ram_buffer(&[0u8; 0][..]);
        assert!(empty.is_err(), "an empty checkpoint holds no model");
        let config = ModelLoadRamBufferConfig {
            quantization: Scalar::F64,
            ..ModelLoadRamBufferConfig::from(&[1u8; 16][..])
        };
        assert!(Model::load_ram_buffer(config).is_err(), "blocks run in bf16 or e4m3");
    }

    #[test]
    fn a_stopped_load_creates_no_model() {
        let checkpoint = checkpoint();
        let mut stop = |_, _| ControlFlow::Break(());
        let config = ModelLoadRamBufferConfig {
            progress: Some(&mut stop),
            ..ModelLoadRamBufferConfig::from(checkpoint.as_slice())
        };
        assert_eq!(Model::load_ram_buffer(config).unwrap_err().status, Status::Cancelled);
    }
}

// endregion: Tests
