// Copyright 2025 Irreducible Inc.
//! Core datatypes common to prover and verifier of Binius64.
//!
//! Most imporantly it hosts the definition of a [`ConstraintSystem`].

#![warn(rustdoc::missing_crate_level_docs)]

pub mod constraint_system;
pub mod consts;
pub mod error;
pub mod verify;
pub mod word;

pub use constraint_system::*;
pub use error::ConstraintSystemError;
pub use word::Word;
