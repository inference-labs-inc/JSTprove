// Copyright 2026 The Binius Developers

//! Interactive Oracle Protocol (IOP) components for Binius64 proof generation.
//!
//! This crate provides the prover-side implementations of IOPs used in the Binius64
//! proof system, including polynomial commitment schemes (BaseFold), FRI protocols,
//! and Merkle tree construction.
//!
//! # When to use this crate
//!
//! This crate is primarily used internally by `binius_prover`. Direct use is needed
//! when implementing custom proving logic or when working with the IOP layer directly.
//!
//! # Key types
//!
//! - [`basefold`] - BaseFold polynomial commitment scheme proving
//! - [`fri`] - FRI (Fast Reed-Solomon Interactive Oracle Proof) proving
//! - [`merkle_tree`] - Merkle tree commitment construction
//! - [`channel`] - IOP prover channel traits for abstracting oracle interactions
//!
//! # Related crates
//!
//! - `binius_iop` - Verifier-side IOP implementations
//! - `binius_prover` - High-level proving that uses this crate

#![warn(rustdoc::missing_crate_level_docs)]

pub mod basefold;
pub mod basefold_channel;
pub mod basefold_compiler;
pub mod basefold_zk_channel;
pub mod channel;
pub mod fri;
pub mod merkle_tree;
pub mod naive_channel;
