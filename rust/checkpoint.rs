//! The SafeTensors checkpoint reader: an 8-byte little-endian header length, a JSON header of
//! `name → {dtype, shape, data_offsets}`, then the contiguous little-endian payload — exactly
//! what `cuda_core/checkpoint.hpp`'s `checkpoint_to_save_t::save` writes. The file is
//! memory-mapped once; tensors are handed out as [`TensorPayload`] — zero-copy rank-2 views in
//! their **stored** dtype — so each weight format packs its own representation without a decode
//! round-trip; the few consumers that genuinely want converted storage (norm gains, the
//! byte-embedding table, the pool query) opt into one explicit copy via
//! [`TensorPayload::try_decode`].

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use numkong::{bf16, CastDtype, Matrix, MatrixView, StorageElement};

use crate::error::{Error, Result};

#[derive(Clone, Copy, PartialEq)]
enum StorageDtype {
    Bf16,
    F32,
    I8,
}

struct Entry {
    storage: StorageDtype,
    shape: [usize; 2],
    start: usize,
    stop: usize,
}

/// One tensor's payload in its stored dtype — a zero-copy `[rows, cols]` view over the mmap;
/// vectors keep their `[1, cols]` rank-2 shape.
pub enum TensorPayload<'checkpoint> {
    Bf16(MatrixView<'checkpoint, bf16>),
    F32(MatrixView<'checkpoint, f32>),
    I8(MatrixView<'checkpoint, i8>),
}

impl TensorPayload<'_> {
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorPayload::Bf16(view) => view.shape(),
            TensorPayload::F32(view) => view.shape(),
            TensorPayload::I8(view) => view.shape(),
        }
    }

    /// Decode into an owned matrix of the requested scalar — the one explicit, caller-visible
    /// copy a consumer opts into when it genuinely wants converted storage: norm gains, the
    /// embedding table, staging a mismatched checkpoint into a precision's operand dtype.
    pub fn try_decode<Destination: Clone + CastDtype + StorageElement + Default>(
        &self,
    ) -> Option<Matrix<Destination>> {
        let mut decoded = Matrix::<Destination>::try_zeros(self.shape()).ok()?;
        match self {
            TensorPayload::Bf16(view) => view.try_cast_into(&mut decoded).ok()?,
            TensorPayload::F32(view) => view.try_cast_into(&mut decoded).ok()?,
            TensorPayload::I8(view) => view.try_cast_into(&mut decoded).ok()?,
        }
        Some(decoded)
    }
}

/// The memory-mapped checkpoint and its parsed header.
pub struct SafeTensors {
    map: Mmap,
    entries: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
    payload: usize,
}

impl SafeTensors {
    /// Map the file and parse its header, bounds-checking every offset against the mapping —
    /// a truncated or malformed file returns an error, never panics.
    ///
    /// # Errors
    /// [`Error::Io`] when the file cannot be opened or mapped; [`Error::MalformedHeader`] when
    /// the header is truncated, exceeds the file, names an unsupported dtype or rank, or points
    /// a tensor outside the payload.
    pub fn open(path: &Path) -> Result<Self> {
        let malformed = |reason: &str, name: &str| {
            Error::MalformedHeader(if name.is_empty() { reason.into() }
                                   else { format!("{reason} for {name}") })
        };
        let map = unsafe { Mmap::map(&File::open(path)?)? };
        if map.len() < 8 {
            return Err(malformed("file shorter than the 8-byte header length", ""));
        }
        let header_bytes = u64::from_le_bytes(map[..8].try_into().expect("invariant: length checked above")) as usize;
        let payload = 8usize.checked_add(header_bytes).filter(|end| *end <= map.len())
            .ok_or_else(|| malformed("header length exceeds the file", ""))?;
        let header: serde_json::Value = serde_json::from_slice(&map[8..payload])
            .map_err(|error| Error::MalformedHeader(error.to_string()))?;
        let payload_bytes = map.len() - payload;
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        for (name, value) in header.as_object().into_iter().flatten() {
            if name == "__metadata__" {
                for (key, entry) in value.as_object().into_iter().flatten() {
                    if let Some(text) = entry.as_str() {
                        metadata.insert(key.clone(), text.to_string());
                    }
                }
                continue;
            }
            let storage = match value["dtype"].as_str() {
                Some("BF16") => StorageDtype::Bf16,
                Some("F32") => StorageDtype::F32,
                Some("I8") => StorageDtype::I8,
                _ => return Err(malformed("unsupported dtype", name)),
            };
            let dims: Vec<usize> = value["shape"].as_array().into_iter().flatten()
                .filter_map(|d| d.as_u64()).map(|d| d as usize).collect();
            let shape = match dims.as_slice() {
                [cols] => [1, *cols],
                [rows, cols] => [*rows, *cols],
                _ => return Err(malformed("rank above 2", name)),
            };
            let offsets = value["data_offsets"].as_array()
                .filter(|offsets| offsets.len() == 2)
                .ok_or_else(|| malformed("missing data_offsets", name))?;
            let start = offsets[0].as_u64().ok_or_else(|| malformed("bad data_offsets", name))? as usize;
            let stop = offsets[1].as_u64().ok_or_else(|| malformed("bad data_offsets", name))? as usize;
            if start > stop || stop > payload_bytes {
                return Err(malformed("data_offsets outside the payload", name));
            }
            entries.insert(name.clone(), Entry { storage, shape, start, stop });
        }
        Ok(Self { map, entries, metadata, payload })
    }

    /// One `__metadata__` value, parsed — the checkpoint self-describes its hyperparameters
    /// this way: `step`, `head_width`, `rotary_base`, `rms_epsilon`.
    pub fn metadata<Value: core::str::FromStr>(&self, key: &str) -> Option<Value> {
        self.metadata.get(key).and_then(|text| text.parse().ok())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Option<[usize; 2]> {
        self.entries.get(name).map(|entry| entry.shape)
    }

    /// One tensor as a zero-copy view in its stored dtype.
    pub fn view(&self, name: &str) -> Option<TensorPayload<'_>> {
        let entry = self.entries.get(name)?;
        Some(unsafe {
            match entry.storage {
                StorageDtype::Bf16 => TensorPayload::Bf16(self.raw_view(entry)),
                StorageDtype::F32 => TensorPayload::F32(self.raw_view(entry)),
                StorageDtype::I8 => TensorPayload::I8(self.raw_view(entry)),
            }
        })
    }

    /// # Safety
    /// `Scalar` must match the entry's storage dtype; the view borrows the mmap.
    unsafe fn raw_view<Scalar: StorageElement>(&self, entry: &Entry) -> MatrixView<'_, Scalar> {
        let bytes = &self.map[self.payload + entry.start..self.payload + entry.stop];
        let element = core::mem::size_of::<Scalar>() as isize;
        let strides = [entry.shape[1] as isize * element, element];
        MatrixView::from_raw_parts(bytes.as_ptr() as *const Scalar, &entry.shape, &strides)
    }
}
