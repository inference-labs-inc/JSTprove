// Copyright 2025 Irreducible Inc.

//! Standard library of circuit gadgets for Binius64.
//!
//! This crate provides pre-built circuit gadgets for common cryptographic and computational
//! operations, including hash functions (SHA256, SHA512, Blake2, Keccak), elliptic curve
//! operations (ECDSA, secp256k1), and utility gadgets.
//!
//! # When to use this crate
//!
//! Use this crate when building circuits with [`binius_frontend`] and you need implementations
//! of common cryptographic primitives. These gadgets are optimized for Binius64's constraint
//! system.
//!
//! # Related crates
//!
//! - [`binius_frontend`] - Circuit construction API that these gadgets build on

#![warn(rustdoc::missing_crate_level_docs)]

pub mod base64;
pub mod bignum;
pub mod bitcoin;
pub mod blake2b;
pub mod blake2s;
pub mod bytes;
pub mod concat;
pub mod ecdsa;
pub mod fixed_byte_vec;
pub mod float64;
pub mod hash_based_sig;
pub mod hmac;
pub mod jwt_claims;
pub mod keccak;
pub mod multiplexer;
pub mod popcount;
pub mod ripemd;
pub mod rs256;
pub mod secp256k1;
pub mod sha256;
pub mod sha512;
pub mod skein512;
pub mod slice;
