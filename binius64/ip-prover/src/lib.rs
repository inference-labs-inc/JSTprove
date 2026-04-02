// Copyright 2026 The Binius Developers

//! Interactive Polynomial (IP) protocol proving for Binius64.
//!
//! This crate provides the prover-side implementations of interactive polynomial
//! protocols used in Binius64, including sumcheck, prodcheck, and fractional
//! addition check provers.
//!
//! # When to use this crate
//!
//! This crate is primarily used internally by `binius_prover` and `binius_iop_prover`.
//! Direct use is needed when implementing custom proving logic or working with
//! the IP protocol layer directly.
//!
//! # Key types
//!
//! - [`sumcheck`] - Sumcheck protocol proving
//! - [`prodcheck`] - Product check protocol proving
//! - [`fracaddcheck`] - Fractional addition check proving
//! - [`channel`] - IP prover channel traits
//!
//! # Related crates
//!
//! - [`binius_ip`] - Verifier-side IP implementations and shared data structures
//! - `binius_iop_prover` - Higher-level IOP provers built on IP

#![warn(rustdoc::missing_crate_level_docs)]

pub mod channel;
pub mod fracaddcheck;
pub mod prodcheck;
pub mod sumcheck;
