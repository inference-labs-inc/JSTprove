// Copyright 2026 The Binius Developers

#![warn(rustdoc::missing_crate_level_docs)]

//! Cryptographic hash functions and compression functions for Binius.
//!
//! This crate provides hash function implementations used throughout the Binius proof system,
//! including both standard hash functions (SHA-256) and specialized binary field hash functions
//! (Vision).

pub mod compress;
pub mod parallel_compression;
pub mod parallel_digest;
mod serialization;
pub mod vision;
pub mod vision_4;
pub mod vision_6;

pub use compress::{CompressionFunction, PseudoCompressionFunction};
pub use parallel_compression::{ParallelCompressionAdaptor, ParallelPseudoCompression};
pub use parallel_digest::{MultiDigest, ParallelDigest, ParallelMultidigestImpl};
pub use serialization::*;

/// The standard digest is SHA-256.
pub type StdDigest = sha2::Sha256;
pub type StdCompression = compress::sha256::Sha256Compression;
