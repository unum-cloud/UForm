//! The traits only this crate implements, and the carriers their signatures name: public within
//! this private module, so callers see them in bounds yet can neither name nor implement them.
//!
//! File: rust/sealed.rs
//! Author: Ash Vardanian
//! Date: November 18, 2026

use core::ffi::c_void;

use crate::types::{Element, Result, uform_completion_t, uform_executor_t, uform_outcome_t, uform_tape_view_t};

/// The handles every call names: the model or trainer it reaches and the executor it runs on.
#[derive(Debug)]
pub struct Handles {
    pub handle: *mut c_void,
    pub executor: uform_executor_t,
}

// Safety: a model is shared across threads, and an executor goes with the calls submitted to it.
unsafe impl Send for Handles {}

/// One verb over one `#[repr(C)]` argument struct: where the arguments point, which C entry
/// point takes them, and what a successful call hands back.
pub trait Calling: Sized {
    /// The argument struct of the verb's C entry point.
    type Arguments;
    /// What a successful call hands back.
    type Output;
    /// The inputs the verb took, in argument order, handed back once the call is final.
    type Inputs;

    /// The arguments over the inputs and outputs this verb holds, which stays in place until
    /// the call they feed is final, or the refusal the binding catches before any call.
    fn arguments(&mut self, outcome: *mut uform_outcome_t, completion: uform_completion_t) -> Result<Self::Arguments>;

    /// Passes `arguments` to the verb's C entry point.
    ///
    /// # Safety
    ///
    /// Everything `arguments` points to stays alive and in place until the
    /// outcome is final.
    unsafe fn call(arguments: *mut Self::Arguments);

    /// What the verb hands back once its outcome is `status`, beside the inputs it took.
    fn finish(self, status: Result<()>) -> (Result<Self::Output>, Self::Inputs);
}

/// How the verbs of a receiver complete, which decides what each of them answers.
pub trait Completion: Sized {
    /// `Result<Output>` when blocking, `Result<Deferred<'handle, Verb>>` when deferring.
    type Deferrable<'handle, Verb: Calling>;

    /// Runs `verb`, or checks it and hands it back unsubmitted.
    fn answer<'handle, Verb: Calling>(verb: Verb) -> Self::Deferrable<'handle, Verb>;
}

/// The inputs a completion takes: a blocking receiver takes references and owned values alike,
/// while a deferring one takes only what it owns, since the library reads its inputs after the
/// verb returns. Both take the arena by `&mut`, which no call reads as an input.
#[diagnostic::on_unimplemented(
    message = "a deferred receiver needs owned inputs",
    label = "the library would still read this reference after the verb returns",
    note = "pass texts, files and samples in a `Vec` or an array of owned or `'static` items, a tape as \
            `(Vec<_>, Vec<usize>)`, a `PathBuf`, `String` or `'static` path, and a config owning its \
            handler and stats"
)]
pub trait Accepts<Input> {}

/// A tape the library reads without a copy, and the batch axis it spans.
pub struct Batch {
    pub tape: uform_tape_view_t,
    pub count: usize,
}

/// The tape a verb lays separate items into end to end, kept beside the inputs until the
/// call is final.
pub struct Gathered<Value> {
    pub data: Vec<Value>,
    pub offsets: Vec<usize>,
}

impl<Value> Default for Gathered<Value> {
    fn default() -> Self {
        Gathered { data: Vec::new(), offsets: Vec::new() }
    }
}

/// A ragged batch a verb reads its items from: separate items, gathered into `gathered`, or a
/// tape of data and offsets, read in place.
pub trait TapeView<Value: Element> {
    /// The tape and its count, refused when the offsets do not fit the data.
    fn tape(&self, gathered: &mut Gathered<Value>) -> Result<Batch>;
}
