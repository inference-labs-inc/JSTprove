// Copyright 2025 Irreducible Inc.
//! SIMD binary field operations for the POLYVAL field.
//!
//! This module implements multiplication in the GF(2^128) binary field used by the POLYVAL
//! universal hash function, which is part of the AES-GCM-SIV authenticated encryption mode
//! defined in RFC 8452.
//!
//! The POLYVAL field uses the reduction polynomial x^128 + x^127 + x^126 + x^121 + 1,
//! which is the bit-reversal of the polynomial used in GHASH/GCM. This choice makes
//! POLYVAL more efficient on little-endian architectures.

pub mod clmul;
pub mod soft64;

/// The multiplicative identity in polynomial Montgomery form
pub const MONTGOMERY_ONE: u128 = 0xc2000000000000000000000000000001u128;

// Re-export mul_clmul for backward compatibility
#[allow(unused_imports)]
pub use clmul::mul as mul_clmul;
