// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! High-level proof generation for Binius64 constraint systems.
//!
//! This crate provides the main [`Prover`] struct for generating zero-knowledge proofs
//! that a witness satisfies a constraint system. It is the prover-side counterpart to
//! `binius_verifier`.
//!
//! # When to use this crate
//!
//! Use this crate when you have a compiled constraint system and witness and need to
//! generate a proof. For building circuits and compiling them to constraint systems,
//! see `binius_frontend`.
//!
//! # Key types
//!
//! - [`Prover`] - Main proving interface; call [`Prover::setup`] with a verifier, then
//!   [`Prover::prove`] with witness data
//! - [`KeyCollection`] - Precomputed keys for shift reduction (can be serialized for reuse)
//!
//! # Related crates
//!
//! - `binius_verifier` - Verification counterpart
//! - `binius_frontend` - Circuit construction and compilation
//! - `binius_spartan_prover` - Spartan-based proving (alternative backend)

#![warn(rustdoc::missing_crate_level_docs)]

pub mod and_reduction;
mod error;
pub mod fold_word;
pub mod protocols;
mod prove;
pub mod ring_switch;

pub use binius_field::arch::OptimalPackedB128;
pub use binius_hash as hash;
pub use binius_iop_prover::{fri, merkle_tree};
pub use error::*;
pub use protocols::shift::KeyCollection;
pub use prove::*;
