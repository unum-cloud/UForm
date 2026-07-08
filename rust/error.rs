//! The crate's error story: loading and parsing return [`Result`] — a bad file or an absent
//! tensor is the environment's fault and must surface to the caller — while the forward paths
//! panic via `expect("invariant: …")`, because every shape they touch was fixed and checked at
//! load time, so a failure there is a bug in this crate, not bad input.

use std::fmt;
use std::io;

pub enum Error {
    /// The checkpoint file could not be opened or mapped.
    Io(io::Error),
    /// A tensor the model geometry requires is absent from the checkpoint.
    MissingTensor(String),
    /// A tensor is stored in a dtype the requested [`crate::Precision`] cannot pack.
    UnexpectedDtype { name: String, found: &'static str, expected: &'static str },
    /// The SafeTensors header is truncated, out of bounds, or not the expected JSON shape.
    MalformedHeader(String),
    /// The checkpoint lacks the pool and head tiers — it is a Base, not an Embedder.
    NotAnEmbedder(String),
    /// A NumKong kernel or allocation failed while packing weights at load.
    Kernel(String),
}

pub type Result<T> = core::result::Result<T, Error>;

impl Error {
    pub(crate) fn kernel<Failure: fmt::Debug>(failure: Failure) -> Self {
        Error::Kernel(format!("{failure:?}"))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(error) => write!(formatter, "checkpoint io: {error}"),
            Error::MissingTensor(name) => write!(formatter, "missing tensor {name}"),
            Error::UnexpectedDtype { name, found, expected } => {
                write!(formatter, "{name} is stored as {found}, this precision packs {expected}")
            }
            Error::MalformedHeader(reason) => write!(formatter, "malformed header: {reason}"),
            Error::NotAnEmbedder(name) => {
                write!(formatter, "{name} absent — a Base checkpoint, not an Embedder")
            }
            Error::Kernel(reason) => write!(formatter, "kernel failure at load: {reason}"),
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error::Io(error)
    }
}
