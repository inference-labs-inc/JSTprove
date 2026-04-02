// Copyright 2026 The Binius Developers

//! Interactive Oracle Protocol (IOP) components for Binius64 verification.
//!
//! This crate provides the verifier-side implementations of IOPs used in the Binius64
//! proof system, including polynomial commitment schemes (BaseFold), FRI protocols,
//! and Merkle tree verification.
//!
//! # When to use this crate
//!
//! This crate is primarily used internally by `binius_verifier`. Direct use is needed
//! when implementing custom verification logic or when working with the IOP layer directly.
//!
//! # Key types
//!
//! - [`basefold`] - BaseFold polynomial commitment scheme verification
//! - [`fri`] - FRI (Fast Reed-Solomon Interactive Oracle Proof) verification
//! - [`merkle_tree`] - Merkle tree commitment verification
//! - [`channel`] - IOP verifier channel traits for abstracting oracle interactions
//!
//! # Related crates
//!
//! - `binius_iop_prover` - Prover-side IOP implementations
//! - `binius_verifier` - High-level verification that uses this crate

#![warn(rustdoc::missing_crate_level_docs)]

pub mod basefold;
pub mod basefold_channel;
pub mod basefold_compiler;
pub mod basefold_zk_channel;
pub mod channel;
pub mod fri;
pub mod merkle_tree;
pub mod naive_channel;
pub mod size_tracking_channel;
