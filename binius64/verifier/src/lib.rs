// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! High-level proof verification for Binius64 constraint systems.
//!
//! This crate provides the main [`Verifier`] struct for verifying zero-knowledge proofs
//! that a witness satisfies a constraint system. It is the verifier-side counterpart to
//! `binius_prover`.
//!
//! # When to use this crate
//!
//! Use this crate when you have a constraint system and a proof, and need to verify
//! that the proof is valid. The verifier does not need access to the witness.
//!
//! # Key types
//!
//! - [`Verifier`] - Main verification interface; call [`Verifier::setup`] with a constraint system,
//!   then [`Verifier::verify`] with a proof and public inputs
//! - [`VerificationError`] - Error type returned when proof verification fails
//!
//! # Design philosophy
//!
//! Verifier code is optimized for simplicity, security, and readability rather than
//! performance. It uses only scalar fields (not packed fields), avoids parallelization,
//! and prefers simple data structures.
//!
//! # Related crates
//!
//! - `binius_prover` - Proving counterpart
//! - `binius_core` - Constraint system definitions
//! - `binius_spartan_verifier` - Spartan-based verification (alternative backend)

#![warn(rustdoc::missing_crate_level_docs)]

pub mod and_reduction;
pub mod config;
mod error;
pub mod protocols;
pub mod ring_switch;
mod verify;

pub use binius_hash as hash;
pub use binius_iop::{fri, merkle_tree};
pub use binius_transcript as transcript;
pub use error::*;
pub use verify::*;
