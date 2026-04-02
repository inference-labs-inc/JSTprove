// Copyright 2024-2025 Irreducible Inc.

#![warn(rustdoc::missing_crate_level_docs)]

//! Utility modules used in Binius.

pub mod array_2d;
pub mod bitwise;
pub mod checked_arithmetics;
pub mod env;
pub mod iter;
pub mod mem;
#[cfg(feature = "platform-diagnostics")]
pub mod platform_diagnostics;
pub mod rand;
pub mod random_access_sequence;
pub mod rayon;
pub mod serialization;
pub mod sorting;
pub mod sparse_index;
pub mod strided_array;

pub use bytes;
pub use serialization::{DeserializeBytes, SerializationError, SerializeBytes};
