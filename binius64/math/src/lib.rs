// Copyright 2024-2025 Irreducible Inc.

#![warn(rustdoc::missing_crate_level_docs)]

//! Mathematical primitives used in Binius, built atop the `binius-field` crate.
//!
//! This crate provides a variety of mathematical primitives used in Binius, including:
//!
//! * Multilinear polynomials
//! * Univariate polynomials
//! * Matrix operations
//! * Additive number-theoretic transform
//! * Error-correcting codes

pub mod batch_invert;
pub mod binary_subspace;
pub mod bit_reverse;
pub mod field_buffer;
pub mod fold;
pub mod inner_product;
pub mod line;
pub mod matrix;
pub mod multilinear;
pub mod ntt;
pub mod reed_solomon;
pub mod span;
pub mod tensor_algebra;
#[cfg(feature = "test-utils")]
pub mod test_utils;
pub mod univariate;

pub use binary_subspace::BinarySubspace;
pub use field_buffer::{AsSlicesMut, FieldBuffer, FieldSlice, FieldSliceMut};
pub use matrix::Matrix;
pub use reed_solomon::ReedSolomonCode;
